from pathlib import Path
import json
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.swapping.matcher import build_variant_lookup, find_cluster_spans

BASE = Path("outputs/ins_doc_filtered_replacement_capped_v2")
BASE = Path("outputs/ins_doc_filtered_replacement_capped_v2")

candidates_path = BASE / "candidates.csv"
possible_paths = [
    BASE / "autshumato_swap_inventory.csv",
    BASE / "outputs/autshumato_swap_inventory.csv",
    Path("outputs/lexicon/autshumato_swap_inventory.csv"),
]

swap_inventory_path = None
for p in possible_paths:
    if p.exists():
        swap_inventory_path = p
        break

if swap_inventory_path is None:
    raise FileNotFoundError(
        "Could not find autshumato_swap_inventory.csv. "
        "Run: find outputs -name '*swap_inventory*.csv' "
        "and update swap_inventory_path in scripts/diagnose_xhosa_loss.py."
    )
candidates = pd.read_csv(candidates_path, low_memory=False)
swap_inventory = pd.read_csv(swap_inventory_path, low_memory=False)

xho_candidates = candidates[candidates["language"] == "xho"].copy()
xho_inventory = swap_inventory[swap_inventory["lang"] == "xho"].copy()

print("xho candidates:", len(xho_candidates))
print("xho swap inventory rows:", len(xho_inventory))

print("\nxho inventory by kind:")
print(xho_inventory["kind"].value_counts(dropna=False))

print("\nxho inventory by semantic type:")
print(xho_inventory["semantic_type"].value_counts(dropna=False))

variant_lookup = build_variant_lookup(swap_inventory)

match_rows = []
for _, row in xho_candidates.iterrows():
    matches = find_cluster_spans(
        str(row["other_sentence"]),
        lang="xho",
        variant_lookup=variant_lookup,
    )

    for m in matches:
        match_rows.append({
            "pair_id": row["pair_id"],
            "other_sentence": row["other_sentence"],
            "matched_surface": m["matched_surface"],
            "cluster_id": m["cluster_id"],
            "kind": m["kind"],
            "semantic_type": m["semantic_type"],
        })

matches_df = pd.DataFrame(match_rows)

print("\nxho candidate rows with matches:", matches_df["pair_id"].nunique() if not matches_df.empty else 0)
print("xho total matches:", len(matches_df))

if not matches_df.empty:
    print("\nxho matches by kind:")
    print(matches_df["kind"].value_counts())

    print("\nxho matches by semantic type:")
    print(matches_df["semantic_type"].value_counts())

    out = BASE / "diagnostics_xho_matches.csv"
    matches_df.to_csv(out, index=False)
    print("\nsaved:", out)