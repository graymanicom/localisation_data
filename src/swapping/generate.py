from __future__ import annotations

import json
import random

import pandas as pd

from src.swapping.matcher import build_variant_lookup, find_cluster_spans


def make_cluster_index(swap_inventory_df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        row["cluster_id"]: row
        for _, row in swap_inventory_df.iterrows()
    }


def choose_replacement_variants(
    cluster_row: pd.Series,
    current_surface: str,
    rng: random.Random,
    n_choices: int = 2,
) -> list[str]:
    """
    Pick up to n_choices replacement variants from the same cluster,
    excluding the currently matched surface.
    """
    variants = json.loads(cluster_row["swappable_variants_json"])
    current_len = len(str(current_surface).split())

    candidates = [
        v for v in variants
        if v != current_surface and abs(len(str(v).split()) - current_len) <= 2
    ]

    if not candidates:
        return []

    if len(candidates) <= n_choices:
        rng.shuffle(candidates)
        return candidates

    return rng.sample(candidates, n_choices)


def swap_sentence_once(sentence: str, start: int, end: int, replacement: str) -> str:
    return sentence[:start] + replacement + sentence[end:]


def generate_swapped_sentences(
    pairs_df: pd.DataFrame,
    swap_inventory_df: pd.DataFrame,
    language_col: str,
    sentence_col: str,
    side_name: str,
    n_per_match: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate up to n_per_match alternative swaps for each matched span.
    """
    rng = random.Random(seed)

    variant_lookup = build_variant_lookup(swap_inventory_df)
    cluster_index = make_cluster_index(swap_inventory_df)

    rows: list[dict] = []

    for _, row in pairs_df.iterrows():
        lang = str(row[language_col])
        sent = str(row[sentence_col])

        matches = find_cluster_spans(sent, lang=lang, variant_lookup=variant_lookup)
        if not matches:
            continue

        for match in matches:
            cluster_id = match["cluster_id"]
            cluster_row = cluster_index.get(cluster_id)
            if cluster_row is None:
                continue

            replacements = choose_replacement_variants(
                cluster_row=cluster_row,
                current_surface=match["matched_surface"],
                rng=rng,
                n_choices=n_per_match,
            )
            if not replacements:
                continue

            for replacement in replacements:
                swapped = swap_sentence_once(
                    sent,
                    start=match["start"],
                    end=match["end"],
                    replacement=replacement,
                )

                rows.append(
                    {
                        "pair_id": row.get("pair_id"),
                        "source": row.get("source"),
                        "side": side_name,
                        "language": lang,
                        "original_sentence": sent,
                        "matched_surface": match["matched_surface"],
                        "replacement_surface": replacement,
                        "swapped_sentence": swapped,
                        "cluster_id": cluster_id,
                        "seed_surface": match["seed_surface"],
                        "kind": match["kind"],
                        "semantic_type": match["semantic_type"],
                        "induced_from_source": match["induced_from_source"],
                    }
                )

    return pd.DataFrame(rows)


def generate_bidirectional_swaps(
    pairs_df: pd.DataFrame,
    english_swap_inventory_df: pd.DataFrame,
    other_swap_inventory_df: pd.DataFrame,
    n_per_match: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate swaps on both English and non-English sides.

    Assumptions:
    - English side uses language code 'eng'
    - Non-English side uses the existing 'language' column
    """
    work = pairs_df.copy()

    # English-side dataframe view
    eng_df = work.copy()
    eng_df["swap_language"] = "eng"
    eng_df["swap_sentence"] = eng_df["english"]

    english_swaps = generate_swapped_sentences(
        pairs_df=eng_df,
        swap_inventory_df=english_swap_inventory_df,
        language_col="swap_language",
        sentence_col="swap_sentence",
        side_name="english",
        n_per_match=n_per_match,
        seed=seed,
    )

    # Non-English-side dataframe view
    other_df = work.copy()
    other_df["swap_language"] = other_df["language"]
    other_df["swap_sentence"] = other_df["other_sentence"]

    other_swaps = generate_swapped_sentences(
        pairs_df=other_df,
        swap_inventory_df=other_swap_inventory_df,
        language_col="swap_language",
        sentence_col="swap_sentence",
        side_name="other",
        n_per_match=n_per_match,
        seed=seed,
    )

    return pd.concat([english_swaps, other_swaps], ignore_index=True)