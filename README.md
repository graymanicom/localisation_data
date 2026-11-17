# Localisation Data

This repository contains the data construction pipeline for a multilingual South African localisation evaluation dataset.

The project investigates whether language models encode localisation knowledge: the institutional, bureaucratic, and sociocultural knowledge required to distinguish contextually appropriate statements from statements that remain grammatically plausible but are locally incorrect.

The repository accompanies the paper:

> Localisation and infrastructural inheritance in African language models

---

## Overview

The pipeline constructs multilingual sentence pairs from South African government corpora and generates localisation-sensitive counterfactuals.

For example:

**Valid**

> Submit your birth certificate to the Department of Home Affairs.

**Counterfactual**

> Submit your marriage certificate to the Department of Home Affairs.

The counterfactual remains grammatical but may become institutionally inappropriate.

The goal is to create evaluation data that probes localisation knowledge rather than general language modelling ability.

---

## Languages

The current pipeline supports:

* Afrikaans (`afr`)
* isiXhosa (`xho`)
* isiZulu (`zul`)
* siSwati (`ssw`)
* Sesotho (`sot`)
* Setswana (`tsn`)
* Sepedi (`nso`)
* Tshivenda (`ven`)
* Xitsonga (`tso`)

English is used during lexicon induction and alignment but is not included in the final evaluation set.

---

## Data Sources

### Autshumato

The primary source of multilingual government translations.

Citation:

> Eiselen, R., & Puttkammer, M. J. (2025). Datasets for South African Languages: Bilingual Aligned and Monolingual Datasets from the Autshumato Project. *Journal of Open Humanities Data*, 11, 14.

### Vuk'uzenzele

A multilingual corpus of South African government communication articles.

Citation:

> Data Science for Social Impact Research Group (2023). The Vuk'uzenzele South African Multilingual Corpus.

---

## Pipeline

### 1. Lexicon Construction

A seed lexicon of South African institutions and official documents is defined in:

```text
config/seed_lexicon.yaml
```

Examples include:

* Department of Home Affairs
* SASSA
* SARS
* SAPS
* birth certificates
* passports
* police clearances
* identity documents

The pipeline induces language-specific variants from aligned corpora.

---

### 2. Candidate Selection

Sentences are retained only if they contain localisation-relevant institutions or official documents.

Examples lacking localisation content are removed.

---

### 3. Counterfactual Generation

The system creates invalid localisation examples by replacing institutions or documents with semantically incompatible alternatives.

Examples:

```text
birth certificate → marriage certificate
passport → death certificate
affidavit → police clearance
```

The objective is to preserve grammaticality while disrupting contextual validity.

---

### 4. Quality Control

The pipeline applies:

* duplicate removal
* metadata validation
* semantic transition constraints
* generic institution filtering
* canonical document normalisation
* transition frequency caps

These controls prioritise precision over dataset size.

---

### 5. LLM Repair

Automatically generated counterfactuals may be grammatically awkward.

A large language model can be used to make minimal grammatical repairs while preserving the intentionally incorrect substitution.

Both the rule-generated and repaired versions are retained for human review.

---

### 6. Human Audit

The audit interface supports human assessment of:

* validity of the original sentence
* grammaticality of the counterfactual
* contextual invalidity of the counterfactual
* grammaticality of the repaired version
* preference between rule-based and repaired versions

The archived dataset used in the study has no completed human-audit fields for the selected repair records. Successful language-model repair must not be interpreted as human validation of grammaticality or contextual invalidity.

---

## Repository Structure

```text
localisation_data/
├── config/
├── scripts/
├── src/
├── human audit/
├── outputs/
├── raw_data/
├── reproducibility/
└── README.md
```

Large datasets, intermediate artefacts, and generated outputs are excluded from version control.

---

## Installation

```bash
uv sync --locked
```

Optional development dependencies:

```bash
uv sync --locked --extra dev
```

---

## Typical Workflow

Generate lexicons:

```bash
uv run python main.py lexicon --config-path config/settings.yaml
```

Generate substitutions:

```bash
uv run python main.py substitutions --config-path config/settings.yaml
```

Create audit dataset:

```bash
uv run python scripts/prepare_audit_dataset.py
```

Generate LLM repairs:

```bash
uv run python scripts/add_llm_repairs.py
```

Run human audit:

```text
Open "human audit/localisation_audit_app_with_consent.html"
```

---

## Reproducibility

The repository contains implementations for:

* lexicon induction
* candidate extraction
* counterfactual generation
* filtering
* grammatical repair
* audit dataset construction

See [reproducibility and provenance](reproducibility/README.md) for the dated archive inventory, dataset provenance and historical limitations. A repository clone does not include raw corpora, saved language-model repairs or generated results. Access to the local archives is required to verify their payloads; otherwise corpora must be obtained from their original sources. Language-model requests may produce different repairs when repeated.

Generated datasets and intermediate outputs are intentionally excluded from version control.

---

## Research Use

This repository is intended for research into:

* localisation evaluation
* multilingual NLP
* African language technologies
* mechanistic interpretability
* digital sovereignty
* sociotechnical studies of AI localisation
