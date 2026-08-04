"""
Step 1 -- Measurement Harness: structural verification (checkpoints 1-4, 6).

Runs in seconds and trains nothing, by design. Fold geometry is the part of
this harness where a silent error would be most damaging and least visible --
a leaked test patient or a fold with 2 India-healthy patients would not throw,
it would just quietly produce a number that looks fine and is wrong. Every
check below therefore FAILS LOUDLY rather than warning.

Checkpoint 5 (the label-shuffle negative control) requires actual training and
lives in run_cv_harness.py --shuffle-control. Checkpoints 7-9 (plausibility,
precision gate, frozen baseline artifact) are evaluated in
aggregate_baseline.py once predictions exist.

Usage:
    python classification/step1_cv_harness/validate_harness.py
    python classification/step1_cv_harness/validate_harness.py --tissue palpebral
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_config import (  # noqa: E402
    MIN_RARE_CELL_PER_FOLD,
    N_REPEATS,
    N_SPLITS,
    OUTPUTS_DIR,
    SEED,
    STRATUM_COL,
)
from cv_data import TISSUE_TYPES, build_folds, load_held_out, load_pool, pool_composition  # noqa: E402


class CheckFailure(AssertionError):
    pass


def _report(name: str, passed: bool, detail: str) -> dict:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    return {"check": name, "passed": bool(passed), "detail": detail}


def check_1_test_isolation(folds: list, held_out_ids: set) -> dict:
    """The sealed 33-patient test set must be absent from every fold, in both
    the train and the val position, across all repeats. This is the one error
    that would invalidate not just Step 1 but Step 6 as well -- once a test
    patient has been trained on, the held-out evaluation is spent."""
    leaked = set()
    for f in folds:
        for key in ("outer_train_ids", "outer_val_ids", "inner_train_ids", "inner_val_ids"):
            leaked |= held_out_ids & set(f[key])
    return _report(
        "1. test-set isolation",
        not leaked,
        f"0 of {len(held_out_ids)} held-out test patients appear in any fold"
        if not leaked
        else f"LEAKED {len(leaked)} test patients into folds: {sorted(leaked)[:10]}",
    )


def check_2_partition_integrity(folds: list, pool_ids: set, n_splits: int) -> dict:
    """Within each repeat the outer val folds must be an exact partition of
    the pool: every patient held out exactly once, never twice. This is what
    makes 'pooled out-of-fold' a well-defined estimator -- if a patient
    appeared in two val folds, they would contribute twice to the AUC; if in
    none, the pooled vector would be incomplete."""
    problems = []
    by_repeat = {}
    for f in folds:
        by_repeat.setdefault(f["repeat"], []).append(f)
    for repeat, repeat_folds in sorted(by_repeat.items()):
        seen = []
        for f in repeat_folds:
            seen.extend(f["outer_val_ids"])
            overlap = set(f["outer_train_ids"]) & set(f["outer_val_ids"])
            if overlap:
                problems.append(f"repeat {repeat} fold {f['fold']}: train/val overlap ({len(overlap)})")
            inner_overlap = set(f["inner_train_ids"]) & set(f["inner_val_ids"])
            if inner_overlap:
                problems.append(f"repeat {repeat} fold {f['fold']}: inner train/val overlap")
            # The inner early-stopping split must be drawn ONLY from the outer
            # training portion. If it ever touched the outer val fold, early
            # stopping would be selecting the epoch using the evaluation data
            # -- exactly the bias this harness exists to remove.
            inner_leak = (set(f["inner_train_ids"]) | set(f["inner_val_ids"])) & set(f["outer_val_ids"])
            if inner_leak:
                problems.append(f"repeat {repeat} fold {f['fold']}: inner split touches outer val fold")
        if len(seen) != len(set(seen)):
            problems.append(f"repeat {repeat}: a patient was held out more than once")
        if set(seen) != pool_ids:
            problems.append(f"repeat {repeat}: val folds do not cover the pool exactly")
        if len(repeat_folds) != n_splits:
            problems.append(f"repeat {repeat}: {len(repeat_folds)} folds, expected {n_splits}")
    return _report(
        "2. partition integrity",
        not problems,
        f"each of {len(by_repeat)} repeats partitions all {len(pool_ids)} pool patients exactly once"
        if not problems
        else "; ".join(problems[:5]),
    )


def check_3_rare_cell_coverage(folds: list, pool_df, minimum: int = MIN_RARE_CELL_PER_FOLD) -> dict:
    """India-healthy (23 in the pool) and Italy-anemic (20) are the two cells
    that constrain the per-country AUCs. A fold that received only 1-2 of
    either would produce a degenerate or wildly unstable per-fold
    contribution. Checked on the outer val fold AND on the inner val split,
    since an inner split with no minority examples makes early stopping
    meaningless."""
    lookup = dict(zip(pool_df["patient_id"], pool_df[STRATUM_COL]))
    watched = ["India_0", "Italy_1"]
    worst = {c: (10**9, None) for c in watched}
    problems = []
    for f in folds:
        for key in ("outer_val_ids", "inner_val_ids"):
            for cell in watched:
                n = sum(1 for pid in f[key] if lookup[pid] == cell)
                floor = minimum if key == "outer_val_ids" else 1
                if key == "outer_val_ids" and n < worst[cell][0]:
                    worst[cell] = (n, f"r{f['repeat']}f{f['fold']}")
                if n < floor:
                    problems.append(f"r{f['repeat']}f{f['fold']} {key} has {n}x {cell} (min {floor})")
    detail = ", ".join(f"min {cell} per outer val fold = {worst[cell][0]} (at {worst[cell][1]})" for cell in watched)
    return _report(
        "3. rare-cell coverage",
        not problems,
        detail if not problems else "; ".join(problems[:5]),
    )


def check_4_pair_counts(pool_df, tissue_type: str) -> dict:
    """Reports the discordant-pair counts the pooled estimator will actually
    use, and the multiple by which they beat the single-split baseline (40
    India / 60 Italy). Expected counts are NOT hardcoded for
    forniceal_palpebral: 6 patients dataset-wide lack that crop, and which
    cells they fall into is a fact about the data, so it is computed and
    printed rather than asserted against an assumption."""
    comp = pool_composition(pool_df)
    india_pairs = comp["India"]["n_discordant_pairs"]
    italy_pairs = comp["Italy"]["n_discordant_pairs"]
    single_split = {"India": 40, "Italy": 60}
    passed = india_pairs > single_split["India"] * 10  # sanity floor, not a tuned threshold
    detail = (
        f"pool n={comp['n_total']} cells={comp['cells']} | "
        f"India {comp['India']['n_anemic']}x{comp['India']['n_healthy']}={india_pairs} pairs "
        f"({india_pairs / single_split['India']:.1f}x the single split's 40) | "
        f"Italy {comp['Italy']['n_anemic']}x{comp['Italy']['n_healthy']}={italy_pairs} pairs "
        f"({italy_pairs / single_split['Italy']:.1f}x the single split's 60)"
    )
    return _report(f"4. pair counts ({tissue_type})", passed, detail)


def check_6_reproducibility(pool_df, n_splits: int, n_repeats: int, seed: int) -> dict:
    """Same seed must produce byte-identical fold assignments. Without this,
    Steps 3-5 could not reuse these folds, and the paired bootstrap -- the
    only comparison with enough power to detect a realistic intervention
    effect on 23 India-healthy patients -- would be impossible."""
    a = build_folds(pool_df, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    b = build_folds(pool_df, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    identical = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    return _report(
        "6. fold reproducibility",
        identical,
        f"two independent build_folds(seed={seed}) calls produced identical assignments"
        if identical
        else "fold assignments DIFFER between two identically-seeded calls",
    )


def validate(tissue_type: str, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS, seed: int = SEED) -> dict:
    print(f"\n=== Structural verification: tissue_type={tissue_type} "
          f"(k={n_splits}, repeats={n_repeats}, seed={seed}) ===")
    pool_df = load_pool(tissue_type)
    held_out_df = load_held_out(tissue_type)
    folds = build_folds(pool_df, n_splits=n_splits, n_repeats=n_repeats, seed=seed)

    pool_ids = set(pool_df["patient_id"])
    held_out_ids = set(held_out_df["patient_id"])
    overlap = pool_ids & held_out_ids
    if overlap:
        raise CheckFailure(f"Pool and test split overlap before folds are even built: {sorted(overlap)[:10]}")

    results = [
        check_1_test_isolation(folds, held_out_ids),
        check_2_partition_integrity(folds, pool_ids, n_splits),
        check_3_rare_cell_coverage(folds, pool_df),
        check_4_pair_counts(pool_df, tissue_type),
        check_6_reproducibility(pool_df, n_splits, n_repeats, seed),
    ]
    all_passed = all(r["passed"] for r in results)
    print(f"  --> {'ALL STRUCTURAL CHECKS PASSED' if all_passed else 'STRUCTURAL CHECKS FAILED'}")
    return {
        "tissue_type": tissue_type,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "seed": seed,
        "pool_composition": pool_composition(pool_df),
        "n_held_out": len(held_out_ids),
        "checks": results,
        "all_passed": all_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 1 harness structural verification (checkpoints 1-4, 6).")
    parser.add_argument("--tissue", choices=TISSUE_TYPES, default=None, help="Default: validate both tissue types.")
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tissues = [args.tissue] if args.tissue else TISSUE_TYPES
    reports = [validate(t, args.folds, args.repeats, args.seed) for t in tissues]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "structural_verification.json"
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nSaved structural verification report to {out_path}")

    if not all(r["all_passed"] for r in reports):
        print("\nRefusing to clear Step 1: structural verification failed. Do not run training until this passes.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
