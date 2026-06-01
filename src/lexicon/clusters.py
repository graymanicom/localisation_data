from __future__ import annotations

import json
import math
from typing import Callable

import pandas as pd


def tokenise(text: str) -> list[str]:
    return [t for t in str(text).split() if t.strip()]


def content_tokens(text: str, stopwords: set[str]) -> list[str]:
    return [t for t in tokenise(text) if t not in stopwords]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def add_adjusted_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the original induction score, but add a frequency-aware adjusted score.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["adjusted_score"] = out["score"] * out["count"].map(
        lambda c: math.log1p(max(int(c), 0))
    )
    return out


def suppress_subphrases(df: pd.DataFrame, score_ratio: float = 0.9) -> pd.DataFrame:
    """
    Remove weaker strict subphrases when a longer phrase with similar or better
    score exists for the same source/seed/lang/kind/semantic_type.
    """
    if df.empty:
        return df.copy()

    work = df.copy()
    if "adjusted_score" not in work.columns:
        work = add_adjusted_score(work)

    group_cols = ["induced_from_source", "seed_surface", "lang", "kind", "semantic_type"]
    kept_groups = []

    for _, group in work.groupby(group_cols, dropna=False):
        group = group.copy()
        group["token_len"] = group["surface"].map(lambda s: len(tokenise(s)))
        group = group.sort_values(
            by=["adjusted_score", "score", "count", "token_len"],
            ascending=[False, False, False, False],
        )

        kept: list[dict] = []
        for row in group.to_dict("records"):
            surf = row["surface"]
            adj = float(row["adjusted_score"])
            shadowed = False

            for prev in kept:
                prev_surf = prev["surface"]
                prev_adj = float(prev["adjusted_score"])

                if surf != prev_surf and surf in prev_surf:
                    if adj <= prev_adj * score_ratio:
                        shadowed = True
                        break

            if not shadowed:
                kept.append(row)

        kept_groups.append(pd.DataFrame(kept))

    out = (
        pd.concat(kept_groups, ignore_index=True)
        if kept_groups
        else work.iloc[0:0].copy()
    )
    return out.drop(columns=["token_len"], errors="ignore")


def variant_similarity(
    surface_a: str,
    surface_b: str,
    stopwords: set[str],
    min_jaccard: float = 0.5,
) -> bool:
    """
    Decide whether two surfaces are lexical variants of the same concept.
    """
    a = str(surface_a).strip()
    b = str(surface_b).strip()
    if not a or not b:
        return False

    if a == b:
        return True

    # Superphrase relation.
    if a in b or b in a:
        return True

    a_content = set(content_tokens(a, stopwords))
    b_content = set(content_tokens(b, stopwords))

    if not a_content or not b_content:
        return False

    if a_content == b_content:
        return True

    if jaccard(a_content, b_content) >= min_jaccard:
        return True

    return False


def choose_canonical_surface(cluster_df: pd.DataFrame, stopwords: set[str]) -> str:
    """
    Pick one canonical form for the cluster.
    """
    if cluster_df.empty:
        return ""

    work = cluster_df.copy()
    if "adjusted_score" not in work.columns:
        work = add_adjusted_score(work)

    work["token_len"] = work["surface"].map(lambda s: len(tokenise(s)))
    work["content_len"] = work["surface"].map(lambda s: len(content_tokens(s, stopwords)))

    best = work.sort_values(
        by=["adjusted_score", "count", "content_len", "token_len"],
        ascending=[False, False, False, True],
    ).iloc[0]

    return str(best["surface"])


def build_variant_clusters(
    df: pd.DataFrame,
    stopword_files: dict[str, str],
    load_stopwords_fn: Callable[[str], set[str]],
    min_jaccard: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build source-specific variant clusters.

    Returns:
      cluster_members_df: each candidate row with cluster_id attached
      clusters_df: one row per cluster with JSON-encoded variants
    """
    if df.empty:
        return (
            df.iloc[0:0].copy(),
            pd.DataFrame(
                columns=[
                    "cluster_id",
                    "induced_from_source",
                    "seed_surface",
                    "lang",
                    "kind",
                    "semantic_type",
                    "canonical_surface",
                    "variants_json",
                    "cluster_size",
                    "max_score",
                    "max_adjusted_score",
                    "total_count",
                ]
            ),
        )

    work = df.copy()
    if "adjusted_score" not in work.columns:
        work = add_adjusted_score(work)

    stopword_cache = {
        lang: load_stopwords_fn(path)
        for lang, path in stopword_files.items()
    }

    group_cols = ["induced_from_source", "seed_surface", "lang", "kind", "semantic_type"]
    cluster_member_frames: list[pd.DataFrame] = []
    cluster_rows: list[dict] = []
    cluster_counter = 0

    for group_key, group in work.groupby(group_cols, dropna=False):
        source_name, seed_surface, lang, kind, semantic_type = group_key
        stopwords = stopword_cache.get(lang, set())

        group = group.sort_values(
            by=["adjusted_score", "score", "count"],
            ascending=[False, False, False],
        ).copy()

        clusters: list[list[dict]] = []

        for row in group.to_dict("records"):
            assigned = False
            for cluster in clusters:
                anchor = cluster[0]["surface"]
                if variant_similarity(
                    row["surface"],
                    anchor,
                    stopwords,
                    min_jaccard=min_jaccard,
                ):
                    cluster.append(row)
                    assigned = True
                    break

            if not assigned:
                clusters.append([row])

        for cluster in clusters:
            cluster_counter += 1
            cluster_id = f"cluster::{source_name}::{lang}::{seed_surface}::{cluster_counter}"

            cluster_df = pd.DataFrame(cluster)
            canonical_surface = choose_canonical_surface(cluster_df, stopwords)

            member_df = cluster_df.copy()
            member_df["cluster_id"] = cluster_id
            cluster_member_frames.append(member_df)

            variants = (
                cluster_df.sort_values(
                    by=["adjusted_score", "score", "count"],
                    ascending=[False, False, False],
                )["surface"]
                .drop_duplicates()
                .tolist()
            )

            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "induced_from_source": source_name,
                    "seed_surface": seed_surface,
                    "lang": lang,
                    "kind": kind,
                    "semantic_type": semantic_type,
                    "canonical_surface": canonical_surface,
                    "variants_json": json.dumps(variants, ensure_ascii=False),
                    "cluster_size": len(variants),
                    "max_score": float(cluster_df["score"].max()),
                    "max_adjusted_score": float(cluster_df["adjusted_score"].max()),
                    "total_count": int(cluster_df["count"].sum()),
                }
            )

    cluster_members_df = (
        pd.concat(cluster_member_frames, ignore_index=True)
        if cluster_member_frames
        else work.iloc[0:0].copy()
    )
    clusters_df = pd.DataFrame(cluster_rows)

    return cluster_members_df, clusters_df