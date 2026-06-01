from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _normalise_text(text: str) -> str:
    text = str(text).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_seed_surface(text: str, seed_surfaces: list[str]) -> bool:
    norm = _normalise_text(text)

    for seed in seed_surfaces:
        seed_norm = _normalise_text(seed)
        if not seed_norm:
            continue

        # word/phrase boundary matching
        pattern = r"(?<!\w)" + re.escape(seed_norm) + r"(?!\w)"
        if re.search(pattern, norm, flags=re.UNICODE):
            return True

    return False


def build_candidate_sentence_pool(
    pairs_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Filter raw aligned pairs down to candidate sentences suitable for substitution.

    This prevents classifier data being built from exam questions, generic school text,
    fragments, or non-institutional sentences.
    """
    cf = config.candidate_filter

    if not cf.enabled:
        return pairs_df.copy()

    df = pairs_df.copy()

    # Basic null and length checks.
    df = df[df["english"].notna()]
    df = df[df["other_sentence"].notna()]

    df["english_len"] = df["english"].astype(str).str.len()
    df["other_len"] = df["other_sentence"].astype(str).str.len()

    df = df[
        (df["english_len"] >= cf.min_english_chars)
        & (df["english_len"] <= cf.max_english_chars)
        & (df["other_len"] >= cf.min_other_chars)
        & (df["other_len"] <= cf.max_other_chars)
    ].copy()

    # Remove obvious educational/exam/instructional text.
    for pattern in cf.exclude_english_patterns:
        df = df[
            ~df["english"].astype(str).str.contains(
                pattern,
                case=False,
                regex=True,
                na=False,
            )
        ].copy()

    # Require English seed hit from allowed kinds.
    if cf.require_any_seed_hit:
        allowed_seed_df = seed_df[
            (seed_df["lang"] == "eng")
            & (seed_df["kind"].isin(cf.allowed_seed_kinds))
        ].copy()

        seed_surfaces = allowed_seed_df["surface"].astype(str).dropna().unique().tolist()

        df = df[
            df["english"].astype(str).map(
                lambda text: _contains_seed_surface(text, seed_surfaces)
            )
        ].copy()

    df = df.drop(columns=["english_len", "other_len"], errors="ignore")
    return df.reset_index(drop=True)