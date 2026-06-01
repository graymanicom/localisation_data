from __future__ import annotations
from pathlib import Path
import typer
from src.config import load_config
from src.pipeline import LocalisationPipeline
from shutil import copy2
import yaml

import os
print("Running from:", os.getcwd())

app = typer.Typer(add_completion=False, help="Multilingual localisation pair pipeline.")

@app.command()
def ingest(
    config_path: str = typer.Option("config/settings.yaml", help="Path to YAML config file."),
) -> None:
    """Ingest Autshumato and Vuk'uzenzele into one paired-sentence CSV."""
    config = load_config(config_path)
    pipeline = LocalisationPipeline(config)
    pairs_df = pipeline.ingest()
    out_path = pipeline.save_ingested_pairs(pairs_df)
    typer.echo(f"Saved {len(pairs_df):,} rows to {out_path}")


@app.command()
def lexicon(
    config_path: str = typer.Option("config/settings.yaml", help="Path to YAML config file."),
) -> None:
    """Induce lexicons from both corpora and write comparison outputs."""
    config = load_config(config_path)
    pipeline = LocalisationPipeline(config)
    pairs_df = pipeline.ingest()
    lexicons = pipeline.induce_lexicons(pairs_df)
    saved = pipeline.save_lexicon_outputs(lexicons)
    for name, path in saved.items():
        typer.echo(f"[{name}] -> {path}")

@app.command()
def substitutions(config_path: str = typer.Option("config/settings.yaml", help="Path to YAML config file.")) -> None:

    config = load_config(config_path)
    pipeline = LocalisationPipeline(config)

    pairs_df = pipeline.ingest()
    lexicons = pipeline.induce_lexicons(pairs_df)
    saved_lexicons = pipeline.save_lexicon_outputs(lexicons)
    for name, path in saved_lexicons.items():
        typer.echo(f"[{name}] -> {path}")

    seed_df = lexicons["seed"]
    candidate_df = pipeline.build_candidate_sentence_pool(pairs_df, seed_df)

    out_base = Path(config.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    if config.candidate_filter.save_candidates:
        candidate_path = out_base / "candidates.csv"
        candidate_df.to_csv(candidate_path, index=False)
        typer.echo(f"[candidates] -> {candidate_path.resolve()}")

    dataset_df = pipeline.generate_substitution_dataset(candidate_df, lexicons)

    # --- Output directory ---
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Save config snapshot ---

    def save_resolved_config(config, path: Path):
        with open(path, "w") as f:
            yaml.safe_dump(config.model_dump(), f)

    copy2(config_path, out_dir / "settings_raw.yaml")
    save_resolved_config(config, out_dir / "settings_resolved.yaml")

    # --- Save dataset ---
    sub_dir = out_dir / "substitutions"
    sub_dir.mkdir(parents=True, exist_ok=True)

    out_path = sub_dir / "classifier_dataset.csv"

    print("classifier rows:", len(dataset_df))
    print(dataset_df["label"].value_counts(dropna=False))
    print(dataset_df["substitution_mode"].value_counts(dropna=False))

    invalid = dataset_df[dataset_df["label"] == 0]
    if not invalid.empty:
        print(
            invalid.groupby(
                ["language", "kind", "original_semantic_type", "replacement_semantic_type"]
            ).size().sort_values(ascending=False).head(20)
        )
    dataset_df.to_csv(out_path, index=False)

    typer.echo(f"[classifier_dataset] -> {out_path.resolve()}")


@app.command()
def filter(
    config_path: str = typer.Option("config/settings.yaml", help="Path to YAML config file."),
) -> None:
    """Run ingestion, lexicon induction, and the revised pair-filtering stage."""
    config = load_config(config_path)
    pipeline = LocalisationPipeline(config)

    pairs_df = pipeline.ingest()
    pipeline.save_ingested_pairs(pairs_df)

    lexicons = pipeline.induce_lexicons(pairs_df)
    pipeline.save_lexicon_outputs(lexicons)

    filtered_df = pipeline.filter_pairs(pairs_df, lexicons["combined"])
    out_path = pipeline.save_filtered_pairs(filtered_df)

    kept = int(filtered_df["keep"].sum()) if "keep" in filtered_df else 0
    typer.echo(f"Saved {len(filtered_df):,} rows to {out_path}; kept {kept:,} rows.")


@app.command()
def all(
    config_path: str = typer.Option("config/settings.yaml", help="Path to YAML config file."),
) -> None:
    """Run the full pipeline end-to-end.

    This is the main entrypoint most users will want. It performs:
      1. Ingestion of aligned pairs into one dataframe and CSV.
      2. Lexicon induction from Autshumato and Vuk'uzenzele separately.
      3. Lexicon comparison and combined lexicon export.
      4. Revised filtering: English high-precision screen + paired-language validation.
    """
    filter(config_path=config_path)


if __name__ == "__main__":
    app()
