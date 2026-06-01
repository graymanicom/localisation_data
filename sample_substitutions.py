import pandas as pd
from pathlib import Path
import random

# -------- CONFIG --------
# path to classifier_dataset.csv
INPUT_PATH = "outputs/ins_doc_filtered_replacement_capped_v2/substitutions/classifier_dataset.csv"
OUTPUT_PATH = "outputs/ins_doc_filtered_replacement_capped_v2/substitutions/sample_inspection.csv"

N_VALID = 5
N_INVALID = 5
SEED = 42
# ------------------------

random.seed(SEED)


def sample_per_language(df: pd.DataFrame) -> pd.DataFrame:
    samples = []

    for lang, lang_df in df.groupby("language"):
        valid_df = lang_df[lang_df["label"] == 1]
        invalid_df = lang_df[lang_df["label"] == 0]

        valid_sample = valid_df.sample(
            n=min(N_VALID, len(valid_df)),
            random_state=SEED
        )

        invalid_sample = invalid_df.sample(
            n=min(N_INVALID, len(invalid_df)),
            random_state=SEED
        )

        combined = pd.concat([valid_sample, invalid_sample])
        combined["sample_language"] = lang

        samples.append(combined)

    return pd.concat(samples, ignore_index=True)


def main():
    df = pd.read_csv(INPUT_PATH)

    print("Total rows:", len(df))
    print(df["label"].value_counts())
    print(df["language"].value_counts())

    sample_df = sample_per_language(df)

    # Keep only the most relevant columns for inspection
    cols = [
        "language",
        "label",
        "substitution_mode",
        "original_sentence",
        "text",
        "matched_surface",
        "replacement_surface",
        "kind",
        "original_semantic_type",
        "replacement_semantic_type",
    ]

    sample_df = sample_df[[c for c in cols if c in sample_df.columns]]

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSample saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()