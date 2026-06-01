from __future__ import annotations

import json
from typing import Callable
from src.lexicon.induce import looks_like_valid_institution_candidate
import pandas as pd


def tokenise(text: str) -> list[str]:
    return [t for t in str(text).split() if t.strip()]


def is_swappable_surface(surface: str, kind: str, stopwords: set[str]) -> bool:
    toks = tokenise(surface)
    if not toks:
        return False

    content_toks = [t for t in toks if t not in stopwords]
    if len(content_toks) == 0:
        return False

    # Institutions should generally be multiword unless acronym-like.
    if kind == "institution":
        if len(toks) < 2 and not str(surface).isupper():
            return False

    # Avoid leading/trailing stopword fragments.
    if toks[0] in stopwords or toks[-1] in stopwords:
        return False

    # Avoid very long clause-like surfaces.
    if len(toks) > 5:
        return False

    return True


def build_swap_inventory(
    clusters_df: pd.DataFrame,
    stopword_files: dict[str, str],
    load_stopwords_fn: Callable[[str], set[str]],
    min_variants: int = 2,
    config=None,
) -> pd.DataFrame:
    """
    Convert clusters into swap-ready inventories.
    """
    if clusters_df.empty:
        return clusters_df.copy()

    stopword_cache = {
        lang: load_stopwords_fn(path)
        for lang, path in stopword_files.items()
    }

    rows = []

    for _, row in clusters_df.iterrows():
        lang = row["lang"]
        kind = row["kind"]
        stopwords = stopword_cache.get(lang, set())

        variants = json.loads(row["variants_json"])
        swappable_variants = []

        for v in variants:
            if not is_swappable_surface(v, kind=kind, stopwords=stopwords):
                continue

            if kind == "institution" and config is not None:
                if not looks_like_valid_institution_candidate(
                    v,
                    lang=lang,
                    config=config,
                    stopwords=stopwords,
                ):
                    continue

            swappable_variants.append(v)

        canonical_surface = str(row["canonical_surface"])
        canonical_ok = is_swappable_surface(
            canonical_surface,
            kind=kind,
            stopwords=stopwords,
        )



        if kind == "institution" and config is not None:
            canonical_ok = canonical_ok and looks_like_valid_institution_candidate(
                canonical_surface,
                lang=lang,
                config=config,
                stopwords=stopwords,
            )

        if not canonical_ok:
            continue

        if len(swappable_variants) < min_variants:
            continue

        out = dict(row)
        out["swappable_variants_json"] = json.dumps(swappable_variants, ensure_ascii=False)
        out["swappable_variant_count"] = len(swappable_variants)
        rows.append(out)

    return pd.DataFrame(rows)