# Reproducibility and data provenance

This directory documents the institutional-validity experiments for *Localisation and infrastructural inheritance in African language models*. Run manifests identify the datasets, analysis scripts, recovered commands and saved outputs. SHA-256 inventories support verification of archived files.

The manifests are retrospective records assembled on 17 November 2026. Unknown execution-time code revisions, dependency versions and checkpoint revisions are recorded as null. A matching source-file hash establishes content identity; it does not establish which checkout was executed.

## Repository contents

- `runs/`: provenance records for the study's experiments.
- `archives/snapshot-2026-11-17.json`: archive and member checksums for the dated file snapshot.
- `archive.py` and `archive_sources.json`: local archive creation and verification.
- `run_manifest.template.json`: an unpopulated record for future experiments.

The dataset manifest links corpus-derived substitutions and saved language-model repairs to preparation of the paired classifier dataset in `afroxlmr_comparison`.

## Data and archive availability

Archive payloads are excluded from Git under `private/`. A repository clone includes their inventories, but does not include the datasets, experimental results or model weights. Verification of payloads requires access to the corresponding local archives; no public download location is configured.

The snapshot includes local raw corpora, the `ins_doc_filtered_replacement_capped_v2` construction outputs, source/environment files and supporting provenance evidence. Local archive directories use owner-only permissions and archive files use owner read/write permissions. These permissions provide local access control; they do not provide encryption or an off-machine backup. Historical archive member names are retained to preserve provenance.

Verify an available snapshot without extracting it:

```sh
python3 reproducibility/archive.py verify snapshot-2026-11-17
```

Create a snapshot with a new identifier:

```sh
python3 reproducibility/archive.py create snapshot-YYYY-MM-DD
```

Each identifier must be unique. Existing archives and manifests are never overwritten. Archive contents are selected by `archive_sources.json`; environment files, common credential-file extensions, Python caches and macOS metadata are excluded. Restore archives into a separate directory and verify member hashes before use.

## Reproducing the analyses

Consult the run manifests before selecting an input or result directory. Input, output and source-code paths are relative to the repository root; archive-manifest links are relative to the run-manifest directory. Archive inventories resolve payload paths relative to `reproducibility/`.

The downstream preparation script is `src/prepare_classifier_dataset_with_llm.py` in `afroxlmr_comparison`. It samples one repaired invalid example per original sentence and assigns splits by pair ID, using seed 13. Its reconstruction matches the archived 3,094-row classifier input byte-for-byte. Reissuing language-model requests is not a substitute for the saved repair responses: service outputs may differ.

The lockfile supports consistent dependency installation with `uv sync --locked`. It records the archived repository state, and does not establish the environment used for the original experiments. Remote model caches and remote corpus revisions are not fully archived or pinned.

## Recording future runs

Copy `run_manifest.template.json` to a new file under `runs/`. Record the command, explicit parameters, code revision, input hashes, model/tokenizer revisions, environment and completion status. Add output paths and hashes after execution. Save trained classifier heads, predictions with example IDs, dataset/split snapshots, training histories and any feature scalers or representation arrays required to recompute derived metrics.

The retrospective classifier and feature commands omit an explicit seed. Seed 13 is inferred from matching source defaults; it is explicit and independently reproduced for dataset preparation. Historical limitations are retained in each manifest.
