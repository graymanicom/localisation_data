from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

import pandas as pd

from src.swapping.matcher import build_variant_lookup, find_cluster_spans


@dataclass(frozen=True)
class MatchRecord:
    pair_id: str | None
    source: str | None
    language: str
    original_sentence: str
    matched_surface: str
    cluster_id: str
    seed_surface: str
    kind: str
    semantic_type: str
    induced_from_source: str
    start: int
    end: int


def _token_len(text: str) -> int:
    return len([t for t in str(text).split() if t.strip()])


def _safe_json_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return json.loads(value)


def _normalise(text: str) -> str:
    return str(text).casefold().strip()


def _surface_regex(surface: str) -> str:
    return r"(?<!\w)" + re.escape(str(surface)) + r"(?!\w)"


def _count_surface_occurrences(sentence: str, surface: str) -> int:
    if not sentence or not surface:
        return 0
    pattern = _surface_regex(surface)
    return len(re.findall(pattern, str(sentence), flags=re.IGNORECASE | re.UNICODE))


def _surface_length_ok(
    matched_surface: str,
    replacement_surface: str,
    max_token_length_diff: int,
) -> bool:
    return abs(_token_len(replacement_surface) - _token_len(matched_surface)) <= max_token_length_diff


def _is_singleton_allowed(surface: str, kind: str, replacement_filter) -> bool:
    toks = [t for t in _normalise(surface).split() if t.strip()]

    if len(toks) != 1:
        return True

    token = toks[0]

    if kind == "document":
        return token in {x.casefold() for x in replacement_filter.document_singleton_whitelist}

    if kind == "institution":
        return token in {x.casefold() for x in replacement_filter.institution_singleton_whitelist}

    return True


def _has_singleton_document_context(
    sentence: str,
    surface: str,
    language: str,
    replacement_filter,
) -> bool:
    """
    Extra guard for one-token document matches.

    This prevents cases like:
    - id est
    - id, ego and superego

    from being treated as identity-document examples.
    """
    toks = [t for t in _normalise(surface).split() if t.strip()]
    if len(toks) != 1:
        return True

    context_terms = replacement_filter.document_singleton_context_terms or {}
    terms = {t.casefold() for t in context_terms.get(language, [])}

    # If no language-specific context terms are configured, do not block.
    if not terms:
        return True

    sent = _normalise(sentence)
    return any(term in sent for term in terms)


def _semantic_replacement_allowed(
    kind: str,
    original_semantic_type: str,
    replacement_semantic_type: str,
    replacement_filter,
) -> bool:
    matrix = replacement_filter.allowed_semantic_replacements or {}
    kind_matrix = matrix.get(kind, {})

    if not kind_matrix:
        return original_semantic_type != replacement_semantic_type

    allowed_targets = kind_matrix.get(original_semantic_type, [])
    return replacement_semantic_type in allowed_targets


def _cluster_support_ok(row: pd.Series, kind: str, replacement_filter, matched: bool = False) -> bool:
    if replacement_filter is None or not replacement_filter.enabled:
        return True

    thresholds = (
        replacement_filter.min_matched_total_count_by_kind
        if matched
        else replacement_filter.min_total_count_by_kind
    )

    min_count = int(thresholds.get(kind, 1))
    total_count = int(row.get("total_count", 0))

    return total_count >= min_count

GENERIC_INSTITUTION_SURFACES = {
    "service centre",
    "services centre",
    "customer care",
    "call centre",
    "help desk",
    "information centre",
    "support centre",
    "office",
    "centre",
    "center",
}


GENERIC_INSTITUTION_SUBSTRINGS = {
    "service centre",
    "call centre",
    "help desk",
    "customer care",
}


ALLOWED_GENERIC_INSTITUTION_CONTEXTS = {
    "sassa service centre",
    "home affairs service centre",
    "department of home affairs service centre",
    "thusong service centre",
    "sars service centre",
}


def _is_generic_institution_surface(surface: str, kind: str) -> bool:
    """
    Reject generic institutional infrastructure phrases unless they are anchored
    to a recognised institution.

    The aim is not to ban all service-centre references, but to prevent generic
    or fragmentary phrases from being used as institution replacements.
    """
    if kind != "institution":
        return False

    s = str(surface).casefold().strip()
    s = " ".join(s.split())

    if not s:
        return True

    if s in ALLOWED_GENERIC_INSTITUTION_CONTEXTS:
        return False

    if s in GENERIC_INSTITUTION_SURFACES:
        return True

    if any(bad in s for bad in GENERIC_INSTITUTION_SUBSTRINGS):
        if not any(ok in s for ok in ALLOWED_GENERIC_INSTITUTION_CONTEXTS):
            return True

    return False


def _extract_match_records(
    pairs_df: pd.DataFrame,
    swap_inventory_df: pd.DataFrame,
    language_col: str = "language",
    sentence_col: str = "other_sentence",
) -> list[MatchRecord]:
    variant_lookup = build_variant_lookup(swap_inventory_df)
    records: list[MatchRecord] = []

    for _, row in pairs_df.iterrows():
        lang = str(row[language_col])
        sent = str(row[sentence_col])

        matches = find_cluster_spans(sent, lang=lang, variant_lookup=variant_lookup)

        for m in matches:
            records.append(
                MatchRecord(
                    pair_id=row.get("pair_id"),
                    source=row.get("source"),
                    language=lang,
                    original_sentence=sent,
                    matched_surface=m["matched_surface"],
                    cluster_id=m["cluster_id"],
                    seed_surface=m["seed_surface"],
                    kind=m["kind"],
                    semantic_type=m["semantic_type"],
                    induced_from_source=m["induced_from_source"],
                    start=m["start"],
                    end=m["end"],
                )
            )

    return records


def _candidate_pool_for_invalid_substitution(
    match: MatchRecord,
    swap_inventory_df: pd.DataFrame,
    allowed_kinds: set[str] | None = None,
) -> pd.DataFrame:
    pool = swap_inventory_df.copy()

    pool = pool[pool["lang"] == match.language]
    pool = pool[pool["kind"] == match.kind]
    pool = pool[pool["semantic_type"] != match.semantic_type]
    pool = pool[pool["cluster_id"] != match.cluster_id]

    if allowed_kinds is not None:
        pool = pool[pool["kind"].isin(allowed_kinds)]

    return pool


def _choose_invalid_replacements(
    match: MatchRecord,
    swap_inventory_df: pd.DataFrame,
    rng: random.Random,
    n_choices: int = 1,
    allowed_kinds: set[str] | None = None,
    max_token_length_diff: int = 1,
    replacement_filter=None,
) -> list[tuple[str, str, str]]:
    pool = _candidate_pool_for_invalid_substitution(
        match=match,
        swap_inventory_df=swap_inventory_df,
        allowed_kinds=allowed_kinds,
    )

    if pool.empty:
        return []

    matched_rows = swap_inventory_df[swap_inventory_df["cluster_id"] == match.cluster_id]
    if matched_rows.empty:
        return []

    matched_row = matched_rows.iloc[0]

    if replacement_filter is not None and replacement_filter.enabled:
        if not _cluster_support_ok(matched_row, match.kind, replacement_filter, matched=True):
            return []

        if not _is_singleton_allowed(match.matched_surface, match.kind, replacement_filter):
            return []

        if _is_generic_institution_surface(match.matched_surface, match.kind):
            return []

        if (
            match.kind == "document"
            and replacement_filter.require_document_context_for_singletons
            and not _has_singleton_document_context(
                match.original_sentence,
                match.matched_surface,
                match.language,
                replacement_filter,
            )
        ):
            return []

    candidate_rows: list[tuple[str, str, str]] = []

    for _, row in pool.iterrows():
        replacement_semantic_type = str(row["semantic_type"])

        if replacement_filter is not None and replacement_filter.enabled:
            if not _cluster_support_ok(row, match.kind, replacement_filter, matched=False):
                continue

            if not _semantic_replacement_allowed(
                kind=match.kind,
                original_semantic_type=match.semantic_type,
                replacement_semantic_type=replacement_semantic_type,
                replacement_filter=replacement_filter,
            ):
                continue

        # Safer default: canonical replacements only.
        replacement_surface = str(row["canonical_surface"])
        if _is_generic_institution_surface(replacement_surface, match.kind):
            continue

        if replacement_filter is not None and replacement_filter.enabled:
            if not _is_singleton_allowed(replacement_surface, match.kind, replacement_filter):
                continue

        if not _surface_length_ok(
            match.matched_surface,
            replacement_surface,
            max_token_length_diff=max_token_length_diff,
        ):
            continue

        candidate_rows.append(
            (
                replacement_surface,
                str(row["cluster_id"]),
                replacement_semantic_type,
            )
        )

    if not candidate_rows:
        return []

    deduped = list(dict.fromkeys(candidate_rows))

    if len(deduped) <= n_choices:
        rng.shuffle(deduped)
        return deduped

    return rng.sample(deduped, n_choices)


def generate_invalid_substitutions(
    pairs_df: pd.DataFrame,
    swap_inventory_df: pd.DataFrame,
    language_col: str = "language",
    sentence_col: str = "other_sentence",
    n_per_match: int = 1,
    seed: int = 42,
    allowed_kinds: set[str] | None = None,
    max_token_length_diff: int = 1,
    replacement_filter=None,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []

    match_records = _extract_match_records(
        pairs_df=pairs_df,
        swap_inventory_df=swap_inventory_df,
        language_col=language_col,
        sentence_col=sentence_col,
    )

    for match in match_records:
        if allowed_kinds is not None and match.kind not in allowed_kinds:
            continue

        # Avoid cheap invalids caused by one occurrence changed and another left unchanged.
        if _count_surface_occurrences(match.original_sentence, match.matched_surface) != 1:
            continue

        replacements = _choose_invalid_replacements(
            match=match,
            swap_inventory_df=swap_inventory_df,
            rng=rng,
            n_choices=n_per_match,
            allowed_kinds=allowed_kinds,
            max_token_length_diff=max_token_length_diff,
            replacement_filter=replacement_filter,
        )

        for replacement_surface, replacement_cluster_id, replacement_semantic_type in replacements:
            swapped = (
                match.original_sentence[: match.start]
                + replacement_surface
                + match.original_sentence[match.end :]
            )

            rows.append(
                {
                    "pair_id": match.pair_id,
                    "source": match.source,
                    "language": match.language,
                    "label": 0,
                    "substitution_mode": "invalid_cross_cluster",
                    "original_sentence": match.original_sentence,
                    "matched_surface": match.matched_surface,
                    "replacement_surface": replacement_surface,
                    "text": swapped,
                    "matched_cluster_id": match.cluster_id,
                    "replacement_cluster_id": replacement_cluster_id,
                    "seed_surface": match.seed_surface,
                    "kind": match.kind,
                    "original_semantic_type": match.semantic_type,
                    "replacement_semantic_type": replacement_semantic_type,
                    "induced_from_source": match.induced_from_source,
                }
            )

    return pd.DataFrame(rows)


def cap_invalid_transitions(
    invalid_df: pd.DataFrame,
    max_per_transition: int | None = 50,
    seed: int = 42,
    semantic_type_overrides: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Cap overrepresented invalid transition types.

    Default grouping:
      language + kind + original_semantic_type + replacement_semantic_type

    semantic_type_overrides lets us impose stricter caps for noisy semantic types.
    Example:
      {"application_form": 15}

    If either original or replacement semantic type appears in the override dict,
    the stricter cap is used for that transition group.
    """
    if invalid_df.empty or max_per_transition is None:
        return invalid_df.copy()

    semantic_type_overrides = semantic_type_overrides or {}

    group_cols = [
        "language",
        "kind",
        "original_semantic_type",
        "replacement_semantic_type",
    ]

    missing = [c for c in group_cols if c not in invalid_df.columns]
    if missing:
        raise ValueError(f"Cannot cap transitions. Missing columns: {missing}")

    bad = invalid_df[group_cols].isna().any(axis=1).sum()
    if bad:
        print("[debug] rows with missing cap metadata:")
        print(invalid_df[invalid_df[group_cols].isna().any(axis=1)].head(10))
        raise ValueError(
            f"Cannot cap transitions because {bad} invalid rows have missing metadata."
        )

    capped_parts = []

    for key, group in invalid_df.groupby(group_cols, dropna=False, sort=False):
        language, kind, original_semantic_type, replacement_semantic_type = key

        cap = max_per_transition

        original_semantic_type = str(original_semantic_type)
        replacement_semantic_type = str(replacement_semantic_type)

        override_caps = []

        if original_semantic_type in semantic_type_overrides:
            override_caps.append(semantic_type_overrides[original_semantic_type])

        if replacement_semantic_type in semantic_type_overrides:
            override_caps.append(semantic_type_overrides[replacement_semantic_type])

        if override_caps:
            cap = min([cap] + override_caps)

        if len(group) > cap:
            group = group.sample(n=cap, random_state=seed)

        capped_parts.append(group)

    if not capped_parts:
        return invalid_df.iloc[0:0].copy()

    capped = pd.concat(capped_parts, ignore_index=True)
    capped = capped[invalid_df.columns.tolist()].copy()

    return capped


def generate_classifier_dataset(
    pairs_df: pd.DataFrame,
    swap_inventory_df: pd.DataFrame,
    language_col: str = "language",
    sentence_col: str = "other_sentence",
    n_invalid_per_match: int = 1,
    seed: int = 42,
    allowed_kinds: set[str] | None = None,
    max_token_length_diff: int = 1,
    replacement_filter=None,
    max_invalid_per_transition: int | None = 50,
    max_invalid_per_transition_overrides: dict[str, int] | None = None,
    keep_only_valids_with_invalids: bool = True,
) -> pd.DataFrame:
    invalid_df = generate_invalid_substitutions(
        pairs_df=pairs_df,
        swap_inventory_df=swap_inventory_df,
        language_col=language_col,
        sentence_col=sentence_col,
        n_per_match=n_invalid_per_match,
        seed=seed,
        allowed_kinds=allowed_kinds,
        max_token_length_diff=max_token_length_diff,
        replacement_filter=replacement_filter,
    )

    print("[debug] invalid rows before cap:", len(invalid_df))
    print("[debug] invalid columns before cap:", invalid_df.columns.tolist())

    required_invalid_cols = [
        "language",
        "kind",
        "original_semantic_type",
        "replacement_semantic_type",
    ]

    if not invalid_df.empty:
        missing = [c for c in required_invalid_cols if c not in invalid_df.columns]
        if missing:
            raise ValueError(
                f"Invalid dataframe is missing required transition columns before cap: {missing}"
            )

        bad = invalid_df[required_invalid_cols].isna().any(axis=1).sum()
        if bad:
            print("[debug] bad invalid rows before cap:")
            print(
                invalid_df[
                    invalid_df[required_invalid_cols].isna().any(axis=1)
                ].head(10)
            )
            raise ValueError(
                f"Invalid dataframe has {bad} rows with missing transition metadata before cap."
            )

    invalid_df = cap_invalid_transitions(
        invalid_df,
        max_per_transition=max_invalid_per_transition,
        seed=seed,
        semantic_type_overrides=max_invalid_per_transition_overrides,
    )

    print("[debug] invalid rows after cap:", len(invalid_df))
    print("[debug] invalid columns after cap:", invalid_df.columns.tolist())

    if not invalid_df.empty:
        missing = [c for c in required_invalid_cols if c not in invalid_df.columns]
        if missing:
            raise ValueError(
                f"Invalid dataframe is missing required transition columns after cap: {missing}"
            )

        bad = invalid_df[required_invalid_cols].isna().any(axis=1).sum()
        if bad:
            print("[debug] bad invalid rows after cap:")
            print(
                invalid_df[
                    invalid_df[required_invalid_cols].isna().any(axis=1)
                ].head(10)
            )
            raise ValueError(
                f"Invalid dataframe has {bad} rows with missing transition metadata after cap."
            )

    valid_df = pairs_df.copy()

    if keep_only_valids_with_invalids:
        if invalid_df.empty:
            valid_df = valid_df.iloc[0:0].copy()
        else:
            valid_pair_ids = set(invalid_df["pair_id"].dropna())
            valid_df = valid_df[valid_df["pair_id"].isin(valid_pair_ids)].copy()

    valid_df = valid_df.rename(columns={sentence_col: "text"})
    valid_df["original_sentence"] = valid_df["text"]
    valid_df["label"] = 1
    valid_df["substitution_mode"] = "original_valid"
    valid_df["matched_surface"] = None
    valid_df["replacement_surface"] = None
    valid_df["matched_cluster_id"] = None
    valid_df["replacement_cluster_id"] = None
    valid_df["seed_surface"] = None
    valid_df["kind"] = None
    valid_df["original_semantic_type"] = None
    valid_df["replacement_semantic_type"] = None
    valid_df["induced_from_source"] = None

    cols = [
    "pair_id",
    "source",
    "language",
    "label",
    "substitution_mode",
    "original_sentence",
    "text",
    "matched_surface",
    "replacement_surface",
    "matched_cluster_id",
    "replacement_cluster_id",
    "seed_surface",
    "kind",
    "original_semantic_type",
    "replacement_semantic_type",
    "induced_from_source",
    ]

    valid_df = valid_df[[c for c in cols if c in valid_df.columns]].copy()
    invalid_df = invalid_df[[c for c in cols if c in invalid_df.columns]].copy()

    dataset_df = pd.concat([valid_df, invalid_df], ignore_index=True)

    invalid_final = dataset_df[dataset_df["label"] == 0]

    if not invalid_final.empty:
        bad = invalid_final[required_invalid_cols].isna().any(axis=1).sum()
        if bad:
            print("[debug] final invalid rows with missing metadata:")
            print(
                invalid_final[
                    invalid_final[required_invalid_cols].isna().any(axis=1)
                ].head(10)
            )
            raise ValueError(
                f"Final dataset has {bad} invalid rows with missing transition metadata."
            )

        print("[debug] final invalid transition counts:")
        print(
            invalid_final.groupby(required_invalid_cols)
            .size()
            .sort_values(ascending=False)
            .head(20)
        )

    return dataset_df