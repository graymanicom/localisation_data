from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.filtering.pipeline import PairFilterPipeline
from src.filtering.candidates import build_candidate_sentence_pool
from src.ingest.autshumato import ingest_autshumato
from src.ingest.vukuzenzele import ingest_vukuzenzele
from src.lexicon.induce import combine_lexicons, compare_lexicon_sources, _load_stopwords, induce_lexicon_from_pairs, top_candidates_by_language_and_kind, suppress_subphrases, select_canonical_forms, add_adjusted_score, build_swappable_lexicon
from src.lexicon.schema import load_seed_lexicon
from src.utils.io import ensure_dir
from src.lexicon.clusters import add_adjusted_score, suppress_subphrases, build_variant_clusters
from src.lexicon.swap_inventory import build_swap_inventory
from src.swapping.generate import generate_bidirectional_swaps
from src.swapping.invalid_substitutions import generate_classifier_dataset
from src.lexicon.nguni_documents import normalise_nguni_document_inventory


class LocalisationPipeline:
    def __init__(self, config: ProjectConfig):
        self.config = config
        ensure_dir(config.output_dir)

    def ingest(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        if self.config.autshumato.enabled:
            frames.append(ingest_autshumato(self.config.autshumato))
        if self.config.vukuzenzele.enabled:
            frames.append(ingest_vukuzenzele(self.config.vukuzenzele))
        if not frames:
            raise ValueError("No data sources are enabled.")
        pairs_df = pd.concat(frames, ignore_index=True)
        pairs_df = pairs_df.drop_duplicates(subset=["english", "other_sentence", "language", "source"])
        return pairs_df

    def save_ingested_pairs(self, pairs_df: pd.DataFrame, filename: str = "paired_sentences.csv") -> Path:
        out_path = Path(self.config.output_dir) / filename
        pairs_df.to_csv(out_path, index=False)
        return out_path
    
    def induce_lexicons(self, pairs_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        seed_df = load_seed_lexicon(self.config.lexicon.seed_yaml)

        aut_df, aut_stats = induce_lexicon_from_pairs(
            pairs_df=pairs_df,
            seed_df=seed_df,
            config=self.config.lexicon,
            source_name="autshumato",
        )
        vuk_df, vuk_stats = induce_lexicon_from_pairs(
            pairs_df=pairs_df,
            seed_df=seed_df,
            config=self.config.lexicon,
            source_name="vukuzenzele_hf",
        )

        aut_df = add_adjusted_score(aut_df)
        vuk_df = add_adjusted_score(vuk_df)

        if self.config.lexicon.prune.enabled and self.config.lexicon.prune.suppress_subphrases:
            aut_df = suppress_subphrases(
                aut_df,
                score_ratio=self.config.lexicon.prune.subphrase_score_ratio,
            )
            vuk_df = suppress_subphrases(
                vuk_df,
                score_ratio=self.config.lexicon.prune.subphrase_score_ratio,
            )

        aut_cluster_members, aut_clusters = build_variant_clusters(
            aut_df,
            stopword_files=self.config.lexicon.stopword_files,
            load_stopwords_fn=_load_stopwords,
            min_jaccard=0.5,
        )
        vuk_cluster_members, vuk_clusters = build_variant_clusters(
            vuk_df,
            stopword_files=self.config.lexicon.stopword_files,
            load_stopwords_fn=_load_stopwords,
            min_jaccard=0.5,
        )

        aut_swap_inventory = build_swap_inventory(
            aut_clusters,
            stopword_files=self.config.lexicon.stopword_files,
            load_stopwords_fn=_load_stopwords,
            min_variants=1,
            config=self.config.lexicon,
        )
        aut_swap_inventory = normalise_nguni_document_inventory(aut_swap_inventory)

        vuk_swap_inventory = build_swap_inventory(
            vuk_clusters,
            stopword_files=self.config.lexicon.stopword_files,
            load_stopwords_fn=_load_stopwords,
            min_variants=1,
            config=self.config.lexicon,
        )
        vuk_swap_inventory = normalise_nguni_document_inventory(vuk_swap_inventory)

        aut_top50 = top_candidates_by_language_and_kind(aut_df, top_n=50)
        vuk_top50 = top_candidates_by_language_and_kind(vuk_df, top_n=50)

        combined = combine_lexicons(seed_df, [aut_df, vuk_df])
        comparison = compare_lexicon_sources(
            aut_df,
            vuk_df,
            output_path=self.config.lexicon.compare_output,
        )

        return {
            "seed": seed_df,

            "autshumato_induced": aut_df,
            "autshumato_stats": aut_stats,
            "autshumato_top50_by_lang_kind": aut_top50,
            "autshumato_cluster_members": aut_cluster_members,
            "autshumato_clusters": aut_clusters,
            "autshumato_swap_inventory": aut_swap_inventory,

            "vukuzenzele_induced": vuk_df,
            "vukuzenzele_stats": vuk_stats,
            "vukuzenzele_top50_by_lang_kind": vuk_top50,
            "vukuzenzele_cluster_members": vuk_cluster_members,
            "vukuzenzele_clusters": vuk_clusters,
            "vukuzenzele_swap_inventory": vuk_swap_inventory,

            "combined": combined,
            "comparison": comparison,
        }

    def save_lexicon_outputs(self, lexicons: dict[str, pd.DataFrame]) -> dict[str, Path]:
        saved: dict[str, Path] = {}
        for name, df in lexicons.items():
            out_path = Path(self.config.output_dir) / f"{name}.csv"
            df.to_csv(out_path, index=False)
            saved[name] = out_path
        return saved

    def filter_pairs(self, pairs_df: pd.DataFrame, combined_lexicon_df: pd.DataFrame) -> pd.DataFrame:
        filter_pipeline = PairFilterPipeline(self.config, combined_lexicon_df)
        return filter_pipeline.filter_pairs(pairs_df)

    def save_filtered_pairs(self, filtered_df: pd.DataFrame, filename: str = "filtered_pairs.csv") -> Path:
        out_path = Path(self.config.output_dir) / filename
        filtered_df.to_csv(out_path, index=False)
        return out_path
    
    def generate_substitution_dataset(self, pairs_df: pd.DataFrame, lexicon_outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate a classifier dataset:
        - original sentences = valid
        - cross-cluster substitutions = invalid

        Current policy:
        - Use Autshumato swap inventory only.
        - Restrict to semantically cleaner kinds for now.
        """
        aut_swap_inventory = lexicon_outputs["autshumato_swap_inventory"]
        swap_inventory_df = aut_swap_inventory[aut_swap_inventory["lang"] != "eng"].copy()

        allowed_kinds = set(self.config.substitutions.allowed_kinds)

        dataset_df = generate_classifier_dataset(
            pairs_df=pairs_df,
            swap_inventory_df=swap_inventory_df,
            language_col="language",
            sentence_col="other_sentence",
            n_invalid_per_match=self.config.substitutions.n_invalid_per_match,
            seed=42,
            allowed_kinds=set(self.config.substitutions.allowed_kinds),
            max_token_length_diff=self.config.substitutions.max_token_length_diff,
            replacement_filter=self.config.substitutions.replacement_filter,
            max_invalid_per_transition=self.config.substitutions.max_invalid_per_transition,
            max_invalid_per_transition_overrides=self.config.substitutions.max_invalid_per_transition_overrides,
            keep_only_valids_with_invalids=self.config.substitutions.keep_only_valids_with_invalids,
        )
        return dataset_df
    
    def build_candidate_sentence_pool(self, pairs_df: pd.DataFrame, seed_df: pd.DataFrame) -> pd.DataFrame:
        return build_candidate_sentence_pool( pairs_df=pairs_df, seed_df=seed_df, config=self.config)
