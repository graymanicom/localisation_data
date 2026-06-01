from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from src.utils.io import load_yaml
from dataclasses import dataclass, field
from typing import Dict, List

class AutshumatoPairConfig(BaseModel):
    language: str
    english_path: str
    other_path: str


class AutshumatoConfig(BaseModel):
    enabled: bool = True
    pairs: list[AutshumatoPairConfig] = Field(default_factory=list)


class VukuzenzeleConfig(BaseModel):
    enabled: bool = True
    repo_id: str = "dsfsi/vukuzenzele-sentence-aligned"
    subsets: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=lambda: ["train", "eval", "test"])
    min_alignment_score: float = 0.0
    prefer_cached_offline: bool = False


class LLMFilterConfig(BaseModel):
    enabled: bool = False
    model: str = "gpt-5.4"
    batch_size: int = 20
    reasoning_effort: str = "low"
    temperature: float = 0.0
    max_output_tokens: int = 300
    low_confidence_rule_band: tuple[int, int] = (2, 4)


class EnglishFilteringConfig(BaseModel):
    min_words: int = 8
    max_words: int = 32
    min_chars: int = 25
    max_chars: int = 320
    max_digit_ratio: float = 0.25
    max_upper_ratio_short: float = 0.85
    short_text_upper_ratio_len: int = 12
    allow_sources: list[str] = Field(default_factory=list)
    require_institution_hit: bool = True
    require_action_hit: bool = True
    llm: LLMFilterConfig = Field(default_factory=LLMFilterConfig)


class PairedLanguageFilteringConfig(BaseModel):
    min_words: int = 4
    max_words: int = 40
    require_nonempty: bool = True
    relation_window: int = 8
    min_validation_score: int = 1
    fasttext_model_path: str | None = None


class FilteringConfig(BaseModel):
    english: EnglishFilteringConfig = Field(default_factory=EnglishFilteringConfig)
    paired_language: PairedLanguageFilteringConfig = Field(default_factory=PairedLanguageFilteringConfig)

@dataclass
class LexiconSourceThresholds:
    min_candidate_count: int = 2
    min_candidate_score: float = 0.1
    top_k_per_seed_lang: int = 50

@dataclass
class LexiconPruneConfig:
    enabled: bool = True
    min_tokens_by_kind: Dict[str, int] = field(default_factory=lambda: {
        "institution": 2,
        "action": 1,
        "document": 1,
        "locality": 1,
        "argument": 1,
        "access_frame": 1,
    })

    max_tokens_by_kind: Dict[str, int] = field(default_factory=lambda: {
        "institution": 5,
        "action": 4,
        "document": 4,
        "locality": 4,
        "argument": 4,
        "access_frame": 4,
    })

    drop_leading_stopword: bool = True
    drop_trailing_stopword: bool = True
    suppress_subphrases: bool = True
    subphrase_score_ratio: float = 0.9

class InstitutionFilterConfig(BaseModel):
    enabled: bool = True
    min_tokens: int = 2
    max_tokens: int = 5
    allow_single_token_acronyms: bool = True
    allowed_singletons: list[str] = Field(default_factory=lambda: ["sassa", "sars", "saps", "dha"])
    required_head_terms: dict[str, list[str]] = Field(default_factory=dict)

class CandidateFilterConfig(BaseModel):
    enabled: bool = True
    min_english_chars: int = 25
    max_english_chars: int = 280
    min_other_chars: int = 10
    max_other_chars: int = 280
    exclude_english_patterns: list[str] = Field(default_factory=list)
    require_any_seed_hit: bool = True
    allowed_seed_kinds: list[str] = Field(default_factory=lambda: ["institution", "document"])
    save_candidates: bool = True

class ReplacementFilterConfig(BaseModel):
    enabled: bool = True

    min_total_count_by_kind: dict[str, int] = Field(
        default_factory=lambda: {"institution": 10, "document": 5}
    )
    min_matched_total_count_by_kind: dict[str, int] = Field(
        default_factory=lambda: {"institution": 10, "document": 5}
    )

    document_singleton_whitelist: list[str] = Field(
        default_factory=lambda: [
            "passport",
            "phasepoto",
            "paspoort",
            "permit",
            "laesense",
            "licence",
            "license",
        ]
    )

    institution_singleton_whitelist: list[str] = Field(
        default_factory=lambda: ["sassa", "sars", "saps", "dha"]
    )

    allowed_semantic_replacements: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    use_canonical_replacement_only: bool = True

    require_document_context_for_singletons: bool = True
    document_singleton_context_terms: dict[str, list[str]] = Field(default_factory=dict)


class SubstitutionConfig(BaseModel):
    allowed_kinds: list[str] = Field(default_factory=lambda: ["institution", "document"])
    n_invalid_per_match: int = 1
    max_token_length_diff: int = 1
    max_invalid_per_transition: int | None = 50
    max_invalid_per_transition_overrides: dict[str, int] = Field(default_factory=dict)
    keep_only_valids_with_invalids: bool = True
    replacement_filter: ReplacementFilterConfig = Field(default_factory=ReplacementFilterConfig)
    

class LexiconConfig(BaseModel):
    seed_yaml: str = "config/seed_lexicon.yaml"
    max_ngram: int = 4
    min_candidate_count: int = 2
    min_candidate_score: float = 1.5
    min_target_token_chars: int = 2
    stopword_files: dict[str, str] = Field(default_factory=dict)
    source_thresholds: dict[str, LexiconSourceThresholds] = field(default_factory=dict)
    prune: LexiconPruneConfig = field(default_factory=LexiconPruneConfig)
    blocklist: dict = field(default_factory=dict)
    institution_filter: InstitutionFilterConfig = Field(default_factory=InstitutionFilterConfig)
    compare_output: str = "outputs/lexicon_comparison.csv"

class ProjectConfig(BaseModel):
    project_name: str = "localisation-pipeline"
    output_dir: str = "outputs"
    english_column: str = "english"
    other_column: str = "other_sentence"
    language_column: str = "language"
    source_column: str = "source"
    autshumato: AutshumatoConfig = Field(default_factory=AutshumatoConfig)
    vukuzenzele: VukuzenzeleConfig = Field(default_factory=VukuzenzeleConfig)
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    lexicon: LexiconConfig = Field(default_factory=LexiconConfig)
    candidate_filter: CandidateFilterConfig = Field(default_factory=CandidateFilterConfig)
    substitutions: SubstitutionConfig

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    
def load_config(path: str | Path) -> ProjectConfig:
    raw: dict[str, Any] = load_yaml(path)

    lex_raw = raw.get("lexicon", {})

    source_thresholds = {
        name: LexiconSourceThresholds(**vals)
        for name, vals in lex_raw.get("source_thresholds", {}).items()
    }

    prune_cfg = LexiconPruneConfig(**lex_raw.get("prune", {}))

    lexicon_cfg = LexiconConfig(
        seed_yaml=lex_raw.get("seed_yaml", "config/seed_lexicon.yaml"),
        stopword_files=lex_raw.get("stopword_files", {}),
        max_ngram=lex_raw.get("max_ngram", 4),
        min_target_token_chars=lex_raw.get("min_target_token_chars", 2),
        source_thresholds=source_thresholds,
        prune=prune_cfg,
        blocklist=lex_raw.get("blocklist", {}),
        institution_filter=InstitutionFilterConfig(**lex_raw.get("institution_filter", {})),
        compare_output=lex_raw.get(
            "compare_output",
            "outputs/lexicon/lexicon_comparison.csv",
        ),
    )

    raw["lexicon"] = lexicon_cfg
    raw["candidate_filter"] = CandidateFilterConfig(**raw.get("candidate_filter", {}))
    raw["substitutions"] = SubstitutionConfig(**raw.get("substitutions", {}))
    
    return ProjectConfig.model_validate(raw)

