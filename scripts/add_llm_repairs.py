from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
import time

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    OpenAI = None


LANGUAGE_NAMES = {
    "afr": "Afrikaans",
    "nso": "Sepedi / Northern Sotho",
    "sot": "Sesotho",
    "tsn": "Setswana",
    "ssw": "siSwati",
    "ven": "Tshivenda",
    "tso": "itsonga",
    "xho": "isiXhosa",
    "zul": "isiZulu",
}


def build_prompt(row: pd.Series) -> str:
    lang_code = str(row["language"])
    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)

    return f"""
You are repairing a counterfactual sentence for linguistic audit.

Language: {lang_name}

Original valid sentence:
{row["original_sentence"]}

Rule-swapped invalid sentence:
{row["rule_swapped_sentence"]}

The intended replacement was:
{row["matched_surface"]} -> {row["replacement_surface"]}

Metadata:
- kind: {row["kind"]}
- original semantic type: {row["original_semantic_type"]}
- replacement semantic type: {row["replacement_semantic_type"]}

Task:
Rewrite the rule-swapped sentence so that it is grammatically natural in the same language.

Strict constraints:
1. Make the smallest possible grammatical change.
2. The repaired sentence MUST contain this exact replacement phrase: {row["replacement_surface"]}. Do not remove it. Do not replace it with the original phrase. Do not shorten it.
3. Do not translate the sentence into English.
4. Do not make the sentence semantically valid.
5. Preserve the intended institutional/document mismatch.
6. Return only the repaired sentence. No explanation.
7. If you cannot make the sentence grammatical while preserving the exact replacement phrase, return the rule-swapped sentence unchanged.
""".strip()


def repair_with_ollama(
    model: str,
    row: pd.Series,
    ollama_url: str = "http://localhost:11434/api/generate",
    timeout: int = 120,
) -> tuple[str, str]:
    prompt = build_prompt(row)

    try:
        response = requests.post(
            ollama_url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 256,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = str(data.get("response", "")).strip()
        replacement = str(row["replacement_surface"]).casefold().strip()

        if replacement and replacement not in text.casefold():
            return text, "rejected_missing_replacement"
        
        matched = str(row["matched_surface"]).casefold().strip()

        if matched and matched in text.casefold() and replacement not in text.casefold():
            return text, "rejected_reverted_to_original"

        # Light cleanup in case model adds quotes.
        text = text.strip().strip('"').strip("'").strip()

        if not text:
            return "", "error: empty_response"

        return text, "ok"

    except Exception as exc:
        return "", f"error: {type(exc).__name__}: {exc}"


def repair_with_openai(
    model: str,
    row: pd.Series,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> tuple[str, str]:
    if OpenAI is None:
        return "", "error: openai package not installed"

    if not os.getenv("OPENAI_API_KEY"):
        return "", "error: OPENAI_API_KEY is not set"

    client = OpenAI()
    prompt = build_prompt(row)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                temperature=0.2,
            )

            time.sleep(0.5)

            text = response.output_text.strip()
            return text, "ok"

        except RateLimitError as exc:
            wait = sleep_seconds * (2 ** (attempt - 1))
            print(f"[rate-limit] retrying in {wait:.1f}s")
            time.sleep(wait)

        except Exception as exc:
            if attempt == max_retries:
                return "", f"error: {type(exc).__name__}: {exc}"

            time.sleep(sleep_seconds * attempt)

    return "", "error: unknown"


def repair_sentence(
    backend: str,
    model: str,
    row: pd.Series,
    ollama_url: str,
) -> tuple[str, str]:
    if backend == "ollama":
        return repair_with_ollama(
            model=model,
            row=row,
            ollama_url=ollama_url,
        )

    if backend == "openai":
        return repair_with_openai(
            model=model,
            row=row,
        )

    return "", f"error: unknown backend {backend}"


def ensure_repair_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["llm_repaired_sentence", "llm_repair_status", "llm_model", "llm_backend"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype("object")

    df["llm_repair_status"] = df["llm_repair_status"].replace(
        {"not_run": "", "nan": ""}
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/ins_doc_filtered_replacement_capped_v2/audit/audit_items_rule.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/ins_doc_filtered_replacement_capped_v2/audit/audit_items_llm.csv",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        default="ollama",
    )
    parser.add_argument(
        "--model",
        default="aya-expanse:8b",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )
    parser.add_argument(
        "--only-language",
        default=None,
        help="Optional language code, e.g. xho or zul.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, low_memory=False)
    df = ensure_repair_columns(df)

    if output_path.exists():
        existing = pd.read_csv(output_path, low_memory=False)
        existing = ensure_repair_columns(existing)

        done_ids = set(
            existing.loc[
                existing["llm_repair_status"].astype(str).eq("ok"),
                "audit_id",
            ]
        )

        merge_cols = [
            "audit_id",
            "llm_repaired_sentence",
            "llm_repair_status",
            "llm_model",
            "llm_backend",
        ]

        df = df.merge(
            existing[merge_cols],
            on="audit_id",
            how="left",
            suffixes=("", "_existing"),
        )

        for col in ["llm_repaired_sentence", "llm_repair_status", "llm_model", "llm_backend"]:
            existing_col = f"{col}_existing"
            if existing_col in df.columns:
                df[col] = df[existing_col].combine_first(df[col])
                df = df.drop(columns=[existing_col])
    else:
        done_ids = set()

    df = ensure_repair_columns(df)

    if args.only_language:
        run_mask = df["language"].astype(str).eq(args.only_language)
    else:
        run_mask = pd.Series(True, index=df.index)

    run_mask &= ~df["audit_id"].isin(done_ids)

    run_indices = df[run_mask].index.tolist()

    if args.limit is not None:
        run_indices = run_indices[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rows in input: {len(df)}")
    print(f"Rows already repaired: {len(done_ids)}")
    print(f"Rows to repair now: {len(run_indices)}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")

    for n, idx in enumerate(run_indices, start=1):
        row = df.loc[idx]

        repaired, status = repair_sentence(
            backend=args.backend,
            model=args.model,
            row=row,
            ollama_url=args.ollama_url,
        )

        df.at[idx, "llm_repaired_sentence"] = repaired
        df.at[idx, "llm_repair_status"] = status
        df.at[idx, "llm_model"] = args.model
        df.at[idx, "llm_backend"] = args.backend

        print(f"[repair] {n}/{len(run_indices)} audit_id={row['audit_id']} status={status}")
        if repaired:
            print(repaired[:200])

        if n % 10 == 0 or n == len(run_indices):
            df.to_csv(output_path, index=False)
            print(f"[{n}/{len(run_indices)}] saved -> {output_path}")

    df.to_csv(output_path, index=False)
    print(f"[done] -> {output_path.resolve()}")


if __name__ == "__main__":
    main()