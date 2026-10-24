"""
What this script does
  1. Runs HyFD (Desbordante's implementation) on each CSV and records #FDs and runtime.
     Optionally also runs baselines (--baselines FUN FDep ...), which are algorithms from the paper.
  2. Cross-checks the FD count of every small dataset with an independent brute-force
     discoverer written in this file (--naive), to validate the library's output.
  3. Prints a comparison table against the numbers reported in the paper's Table 1.

Null semantics: the paper uses null = null. Nulls (empty cells, '?', 'NA', 'NULL') are replaced
by one fixed token before running anything, so every tool sees them as an ordinary equal value.

Setup:    pip install desbordante pandas numpy
Usage:    python reproduce_hyfd.py --selftest
          python reproduce_hyfd.py --data_dir ./data --datasets iris chess abalone --naive
          python reproduce_hyfd.py --data_dir ./data --datasets iris abalone --baselines FUN FDep --runs 3 --out results.csv

Data: one CSV per dataset in --data_dir, named <dataset>.csv (e.g. iris.csv). The datasets of Table 1
are listed on the HPI repeatability page for FD discovery (hpi.de/naumann/projects/repeatability/
data-profiling/fds.html). Use --no_header for files without a header row (e.g. raw UCI files).
IMPORTANT: check that #columns and #rows printed for each file match Table 1 of the paper.
"""
import argparse
import csv
import itertools
import os
import statistics
import tempfile
import time

import numpy as np
import pandas as pd

NULL_TOKEN = "__NULL__"
NULL_STRINGS = {"", "?", "na", "n/a", "nan", "null", "none"}

# Table 1 of the paper: dataset -> (cols, rows, #FDs, {algorithm: seconds}); None = ML/TL (memory/time limit)
# Algorithm keys are the paper's names. Desbordante names map to them in ALG_TO_PAPER below.
def _t(tane, fun, fdmine, dfd, depminer, fastfds, fdep, hyfd):
    return {"TANE": tane, "FUN": fun, "FD_Mine": fdmine, "DFD": dfd, "Dep-Miner": depminer,
            "FastFDs": fastfds, "Fdep": fdep, "HyFD": hyfd}

PAPER = {
    "iris": (5, 150, 4, _t(1.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1)),
    "balance-scale": (5, 625, 1, _t(1.2, 0.1, 0.2, 0.3, 0.3, 0.3, 0.2, 0.1)),
    "chess": (7, 28056, 1, _t(2.9, 1.1, 3.8, 1.0, 174.6, 164.2, 125.5, 0.2)),
    "abalone": (9, 4177, 137, _t(2.1, 0.6, 1.8, 1.1, 3.0, 2.9, 3.8, 0.2)),
    "nursery": (9, 12960, 1, _t(4.1, 1.8, 7.1, 0.9, 121.2, 118.9, 46.8, 0.5)),
    "breast-cancer": (11, 699, 46, _t(2.3, 0.6, 2.2, 0.8, 1.1, 1.1, 0.5, 0.2)),
    "bridges": (13, 108, 142, _t(2.2, 0.6, 4.2, 0.9, 0.5, 0.6, 0.2, 0.1)),
    "echocardiogram": (13, 132, 527, _t(1.6, 0.4, 69.9, 1.2, 0.5, 0.5, 0.2, 0.1)),
    "adult": (14, 48842, 78, _t(67.4, 111.6, 531.5, 5.9, 6039.2, 6033.8, 860.2, 1.1)),
    "letter": (17, 20000, 61, _t(260.0, 529.0, 7204.8, 6.0, 1090.0, 1015.5, 291.3, 3.4)),
    "ncvoter": (19, 1000, 758, _t(4.3, 4.0, None, 5.1, 11.4, 1.9, 1.1, 0.4)),
    "hepatitis": (20, 155, 8250, _t(12.2, 175.9, None, 326.7, 5576.5, 9.5, 0.8, 0.6)),
    "horse": (27, 368, 128727, _t(457.0, None, None, None, None, 385.8, 7.2, 7.1)),
    "fd-reduced-30": (30, 250000, 89571, _t(41.1, 77.7, None, None, 377.2, 382.4, None, 513.0)),
    "plista": (63, 1000, 178152, _t(None, None, None, None, None, None, 26.9, 21.8)),
    "flight": (109, 1000, 982631, _t(None, None, None, None, None, None, 216.5, 53.4)),
}
# Desbordante class name -> name used in the paper (TANE is not available as an exact-FD algorithm in Desbordante)
ALG_TO_PAPER = {"HyFD": "HyFD", "FUN": "FUN", "FdMine": "FD_Mine", "DFD": "DFD",
                "Depminer": "Dep-Miner", "FastFDs": "FastFDs", "FDep": "Fdep"}


# ---------------------------------------------------------------------------------------------
# Data loading (null = null)
# ---------------------------------------------------------------------------------------------
def load_table(path, header=True):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(65536)
    try:
        sep = csv.Sniffer().sniff(sample.split("\n")[0] + "\n", delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    df = pd.read_csv(path, sep=sep, header=0 if header else None, dtype=str,
                     keep_default_na=False, na_filter=False, encoding_errors="replace")
    if not header:
        df.columns = [f"c{i}" for i in range(df.shape[1])]
    clean = lambda v: NULL_TOKEN if str(v).strip().lower() in NULL_STRINGS else v
    return df.map(clean) if hasattr(df, "map") else df.applymap(clean)


def to_temp_csv(df):
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------------------------
# Independent brute-force discovery of all minimal, non-trivial FDs (for validation only)
# ---------------------------------------------------------------------------------------------
def naive_fds(df, max_cols=16):
    """Exact minimal FDs X -> A (single RHS), including empty-LHS FDs for constant columns."""
    m = df.shape[1]
    if m > max_cols:
        raise ValueError(f"naive discovery limited to {max_cols} columns (has {m})")
    codes = np.stack([pd.factorize(df.iloc[:, j])[0] for j in range(m)], axis=1)

    def holds(lhs, a):
        if len(lhs) == 0:
            return len(np.unique(codes[:, a])) == 1
        key = np.unique(codes[:, list(lhs)], axis=0, return_inverse=True)[1].ravel()
        pairs = np.unique(np.stack([key, codes[:, a]], axis=1), axis=0)
        return len(pairs) == key.max() + 1

    result = set()
    for a in range(m):
        others = [c for c in range(m) if c != a]
        found = []
        for k in range(len(others) + 1):
            for lhs in itertools.combinations(others, k):
                s = set(lhs)
                if any(f <= s for f in found):  # a generalisation already holds -> not minimal
                    continue
                if holds(lhs, a):
                    found.append(s)
                    result.add((tuple(sorted(lhs)), a))
    return result


# ---------------------------------------------------------------------------------------------
# Desbordante wrappers
# ---------------------------------------------------------------------------------------------
def get_algorithm(name):
    import desbordante
    ns = desbordante.fd.algorithms
    match = [n for n in dir(ns) if n.lower() == name.lower() and not n.startswith("_")]
    cls = getattr(ns, match[0], None) if match else None
    if cls is None:
        raise RuntimeError(f"Desbordante has no FD algorithm called {name}; available: "
                           f"{[n for n in dir(ns) if not n.startswith('_')]}")
    return cls()


def run_desbordante(name, csv_path):
    """Return (runtime_seconds, set_of_fds or None, fd_count). Runtime covers execute() only."""
    algo = get_algorithm(name)
    try:
        algo.load_data(table=(csv_path, ",", True))
    except Exception:
        algo.load_data(table=pd.read_csv(csv_path, dtype=str, keep_default_na=False))
    t0 = time.perf_counter()
    algo.execute()
    elapsed = time.perf_counter() - t0
    fds = algo.get_fds()
    parsed = set()
    try:
        for fd in fds:
            parsed.add((tuple(sorted(fd.lhs_indices)), fd.rhs_index))
    except AttributeError:
        parsed = None  # library version exposes FDs differently; counts still work
    return elapsed, parsed, len(fds)


# ---------------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------------
def fmt(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def run_dataset(name, df, args):
    path = to_temp_csv(df)
    row = {"dataset": name, "cols": df.shape[1], "rows": df.shape[0]}
    try:
        fds_hyfd = None
        for alg in ["HyFD"] + list(args.baselines):
            times, n = [], None
            for _ in range(args.runs):
                t, fds, n = run_desbordante(alg, path)
                times.append(t)
                if alg == "HyFD":
                    fds_hyfd = fds
            row[f"{alg}_fds"] = n
            row[f"{alg}_s"] = statistics.median(times)
        row["hyfd_fds"] = row["HyFD_fds"]
        if args.naive and df.shape[1] <= args.naive_max_cols:
            ref = naive_fds(df, args.naive_max_cols)
            row["naive_fds"] = len(ref)
            if fds_hyfd is not None:
                row["same_fd_set"] = (fds_hyfd == ref)
    finally:
        os.remove(path)
    return row


def print_table(rows, baselines):
    algs = ["HyFD"] + list(baselines)
    hdr = ["dataset", "cols", "rows", "FDs ours", "FDs paper"]
    for a in algs:
        hdr += [f"{a} s ours", f"{a} s paper"]
    hdr += ["naive FDs", "same set"]
    print(" | ".join(f"{h:>14}" for h in hdr))
    for r in rows:
        p = PAPER.get(r["dataset"])
        cells = [r["dataset"], r["cols"], r["rows"], r.get("HyFD_fds"), p[2] if p else "-"]
        for a in algs:
            paper_t = p[3].get(ALG_TO_PAPER.get(a, a)) if p else None
            cells += [fmt(r.get(f"{a}_s"), 2), fmt(paper_t) if (p and paper_t is not None) else ("limit" if p else "-")]
        cells += [r.get("naive_fds", "-"), r.get("same_fd_set", "-")]
        print(" | ".join(f"{str(c):>14}" for c in cells))
        
        counts = {a: r.get(f"{a}_fds") for a in algs}
        if len(set(counts.values())) > 1:
            print(f"   NOTE: algorithms disagree on the FD count: {counts}")


def selftest(args):
    # (1) The paper's Section 5 example: Class(Teacher, Subject)
    cls = pd.DataFrame({"Teacher": ["Brown", "Walker", "Brown", "Miller", "Brown"],
                        "Subject": ["Math", "Math", "English", "English", "Math"]})
    print("Class(Teacher, Subject) minimal FDs (brute force):", naive_fds(cls), "(expected: empty set)")
    # (2) Random table with planted FDs: C = f(A, B), D = g(A)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"A": rng.integers(0, 6, 400), "B": rng.integers(0, 6, 400),
                       "E": rng.integers(0, 40, 400)})
    df["C"] = (df.A * 7 + df.B) % 11
    df["D"] = df.A % 3
    df = df.astype(str)
    ref = naive_fds(df)
    names = list(df.columns)
    print("planted-FD table, brute force:",
          sorted(f"{{{','.join(names[i] for i in l)}}}->{names[a]}" for l, a in ref))
    try:
        path = to_temp_csv(df)
        t, fds, n = run_desbordante("HyFD", path)
        os.remove(path)
        print(f"Desbordante HyFD: {n} FDs in {t:.3f}s; naive: {len(ref)} FDs; "
              f"identical sets: {fds == ref if fds is not None else 'n/a (count equal: %s)' % (n == len(ref))}")
    except ImportError:
        print("Desbordante not installed (pip install desbordante); brute-force part works.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--datasets", nargs="+", default=[], help="names without .csv, e.g. iris adult")
    ap.add_argument("--no_header", nargs="*", default=[], help="datasets whose CSV has no header row")
    ap.add_argument("--baselines", nargs="*", default=[],
                    help="extra Desbordante algorithms to compare with HyFD, e.g. FUN FDep DFD FastFDs "
                         "Depminer FdMine (TANE is not available as exact-FD algorithm in Desbordante)")
    ap.add_argument("--naive", action="store_true", help="cross-check FD counts with brute force")
    ap.add_argument("--naive_max_cols", type=int, default=14)
    ap.add_argument("--runs", type=int, default=1, help="repeat each run; median time is reported")
    ap.add_argument("--out", default=None, help="write results to this CSV file")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
    rows = []
    for name in args.datasets:
        path = os.path.join(args.data_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        df = load_table(path, header=name not in args.no_header)
        print(f"running {name} ({df.shape[1]} cols x {df.shape[0]} rows) ...", flush=True)
        rows.append(run_dataset(name, df, args))
    if rows:
        print()
        print_table(rows, args.baselines)
        if args.out:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
