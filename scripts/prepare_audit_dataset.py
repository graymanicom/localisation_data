from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path

import pandas as pd


REQUIRED_INVALID_COLUMNS = [
    "pair_id",
    "source",
    "language",
    "kind",
    "original_semantic_type",
    "replacement_semantic_type",
    "matched_surface",
    "replacement_surface",
    "text",
    "matched_cluster_id",
    "replacement_cluster_id",
]


def surface_occurrence_count(sentence: str, surface: str) -> int:
    """
    Count phrase-bounded occurrences of a matched surface in a sentence.

    This removes cases where only one occurrence is swapped, creating cheap
    inconsistency negatives such as:
      permit ... permit  ->  id ... permit
    """
    if pd.isna(sentence) or pd.isna(surface):
        return 0

    sentence = str(sentence)
    surface = str(surface)

    if not sentence.strip() or not surface.strip():
        return 0

    pattern = r"(?<!\w)" + re.escape(surface) + r"(?!\w)"
    return len(re.findall(pattern, sentence, flags=re.IGNORECASE | re.UNICODE))


def remove_metadata_broken_rows(invalid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only invalid rows with all metadata needed for audit and later analysis.
    """
    missing_cols = [c for c in REQUIRED_INVALID_COLUMNS if c not in invalid_df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required columns: {missing_cols}")

    before = len(invalid_df)

    clean = invalid_df.copy()
    clean = clean.dropna(subset=REQUIRED_INVALID_COLUMNS)

    for col in REQUIRED_INVALID_COLUMNS:
        clean = clean[clean[col].astype(str).str.strip() != ""]

    after = len(clean)
    print(f"[metadata] kept {after}/{before} invalid rows")

    return clean.reset_index(drop=True)


def remove_repeated_match_swaps(invalid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the matched surface occurs more than once in the original
    sentence.

    These can create invalids that are detectable through internal inconsistency,
    not localisation knowledge.
    """
    before = len(invalid_df)

    df = invalid_df.copy()

    if "original_sentence" not in df.columns:
        raise ValueError("Input is missing original_sentence column.")

    df["matched_surface_occurrences"] = df.apply(
        lambda r: surface_occurrence_count(
            r["original_sentence"],
            r["matched_surface"],
        ),
        axis=1,
    )

    clean = df[df["matched_surface_occurrences"] == 1].copy()

    after = len(clean)
    print(f"[repeated-match] kept {after}/{before} invalid rows")

    return clean.reset_index(drop=True)


def create_audit_dataset(invalid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one human-auditable row per invalid substitution candidate.
    """
    audit = pd.DataFrame(
        {
            "audit_id": [str(uuid.uuid4()) for _ in range(len(invalid_df))],
            "pair_id": invalid_df["pair_id"],
            "source": invalid_df["source"],
            "language": invalid_df["language"],
            "kind": invalid_df["kind"],
            "original_semantic_type": invalid_df["original_semantic_type"],
            "replacement_semantic_type": invalid_df["replacement_semantic_type"],
            "matched_surface": invalid_df["matched_surface"],
            "replacement_surface": invalid_df["replacement_surface"],
            "matched_cluster_id": invalid_df["matched_cluster_id"],
            "replacement_cluster_id": invalid_df["replacement_cluster_id"],
            "original_sentence": invalid_df["original_sentence"],
            "rule_swapped_sentence": invalid_df["text"],

            # To be filled later by LLM-repair script.
            "llm_repaired_sentence": "",
            "llm_repair_status": "not_run",
            "llm_model": "",

            # To be filled by human audit app.
            "auditor_name": "",
            "original_valid": "",
            "rule_grammatical": "",
            "rule_invalid_sa_context": "",
            "llm_grammatical": "",
            "llm_invalid_sa_context": "",
            "preferred_version": "",
            "auditor_notes": "",
        }
    )

    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/ins_doc_filtered_replacement_capped_v2/substitutions/classifier_dataset.csv",
        help="Path to classifier_dataset.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/ins_doc_filtered_replacement_capped_v2/audit/audit_items_rule.csv",
        help="Path to save audit dataset CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, low_memory=False)

    invalid = df[df["label"] == 0].copy()
    print(f"[input] invalid rows: {len(invalid)}")

    invalid = remove_metadata_broken_rows(invalid)
    invalid = remove_repeated_match_swaps(invalid)

    audit = create_audit_dataset(invalid)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)

    print(f"[audit] saved: {output_path.resolve()}")
    print(f"[audit] rows: {len(audit)}")

    if len(audit):
        print("\n[audit] by language:")
        print(audit["language"].value_counts())

        print("\n[audit] by kind:")
        print(audit["kind"].value_counts())

        print("\n[audit] top transitions:")
        print(
            audit.groupby(
                ["language", "kind", "original_semantic_type", "replacement_semantic_type"]
            )
            .size()
            .sort_values(ascending=False)
            .head(20)
        )


if __name__ == "__main__":
    main()