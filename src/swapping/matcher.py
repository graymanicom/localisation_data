from __future__ import annotations

import json
import re

import pandas as pd


def build_variant_lookup(swap_inventory_df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Build a per-language lookup of cluster variants.
    """
    lookup: dict[str, list[dict]] = {}

    if swap_inventory_df.empty:
        return lookup

    for _, row in swap_inventory_df.iterrows():
        lang = row["lang"]
        variants = json.loads(row["swappable_variants_json"])

        for variant in variants:
            lookup.setdefault(lang, []).append(
                {
                    "cluster_id": row["cluster_id"],
                    "variant": variant,
                    "canonical_surface": row["canonical_surface"],
                    "seed_surface": row["seed_surface"],
                    "kind": row["kind"],
                    "semantic_type": row["semantic_type"],
                    "induced_from_source": row["induced_from_source"],
                }
            )

    # Longest-first matching to reduce partial overlaps.
    for lang in lookup:
        lookup[lang] = sorted(
            lookup[lang],
            key=lambda x: len(str(x["variant"]).split()),
            reverse=True,
        )

    return lookup


def _surface_regex(surface: str) -> str:
    """
    Phrase-bounded regex for matching surfaces inside a sentence.
    """
    surface = re.escape(str(surface))
    return r"(?<!\w)" + surface + r"(?!\w)"


def find_cluster_spans(sentence: str, lang: str, variant_lookup: dict[str, list[dict]]) -> list[dict]:
    """
    Find non-overlapping variant matches in a sentence.
    """
    sent = str(sentence)
    matches: list[dict] = []

    for item in variant_lookup.get(lang, []):
        variant = str(item["variant"])
        if not variant:
            continue

        pattern = _surface_regex(variant)
        for m in re.finditer(pattern, sent, flags=re.UNICODE):
            matches.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "matched_surface": variant,
                    **item,
                }
            )

    # Longest-first, then left-to-right.
    matches = sorted(
        matches,
        key=lambda x: (x["start"], -(x["end"] - x["start"])),
    )

    kept = []
    occupied: list[tuple[int, int]] = []

    for m in matches:
        overlap = False
        for s, e in occupied:
            if not (m["end"] <= s or m["start"] >= e):
                overlap = True
                break

        if not overlap:
            kept.append(m)
            occupied.append((m["start"], m["end"]))

    return kept