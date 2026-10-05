"""Run every important split-brain configuration with one command, then print a CER table.

    nohup python run_split_brain_suite.py --datasetPath /data/hossein/mm_project/CORP_data_release --gpus 0,1 > suite.log 2>&1 &
    python run_split_brain_suite.py --summary          # table only (also printed at the end of the suite)

Runs are independent processes of start_split_brain.py (one per GPU at a time).
A run whose final_cer.json exists is skipped; an interrupted run resumes from its checkpoint.
Logs: <root>/<name>.log, results: <root>/<name>/final_cer.json, table: <root>/summary.csv
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

# (name, flags for start_split_brain.py, tier, what it tests)
RUNS = [
    # ---- tier 1: core comparison (conv encoder, e2e, no noise unless stated) ----
    ("none",            "--split none",                                    1, "baseline: one encoder, CTC only"),
    ("none_in08",       "--split none --input_noise_sd 0.8",               1, "baseline + input noise (standard augmentation)"),
    ("split_noaux",     "--aux none",                                      1, "split encoders, CTC only (architecture effect)"),
    ("T",               "--aux T",                                         1, "split-brain: predict other array's activity"),
    ("L",               "--aux L",                                         1, "split-brain: predict other array's latent"),
    ("M",               "--aux M",                                         1, "split-brain: activity + latent"),
    ("M_freeze",        "--mode pretrain_freeze --aux M",                  1, "split-brain pretrain -> frozen encoders -> CTC"),
    ("M_in08",          "--aux M --input_noise_sd 0.8",                    1, "M + input noise"),
    ("M_emb05",         "--aux M --emb_noise_sd 0.5",                      1, "M + embedding noise (CTC path)"),
    # ---- tier 2: ablations and other encoders ----
    ("M_random",        "--aux M --split random",                          2, "random channel split instead of arrays"),
    ("M_freeze_linear", "--mode pretrain_freeze --aux M --decoder linear", 2, "linear CTC probe on frozen features"),
    ("M_finetune",      "--mode pretrain_finetune --aux M",                2, "split-brain pretrain -> full fine-tune"),
    ("gru_none",        "--arch gru --enc_layers 2 --split none",          2, "GRU encoder baseline"),
    ("gru_M",           "--arch gru --enc_layers 2 --aux M",               2, "GRU encoder + split-brain"),
    ("conformer_none",  "--arch conformer --split none",                   2, "Conformer encoder baseline"),
    ("conformer_M",     "--arch conformer --aux M",                        2, "Conformer encoder + split-brain"),
]

COLS = [("seen", "seen"), ("no_recal", "unseen_norecal"), ("recal", "unseen_recal"),
        ("unseen_all", "unseen_all"), ("eval_single_pool", "eval_single")]


def parse():
    p = argparse.ArgumentParser(description="Run the split-brain experiment suite")
    p.add_argument("--datasetPath", type=str, default="/data/hossein/mm_project/CORP_data_release")
    p.add_argument("--root", type=str, default="sb_runs", help="all run folders and logs go here")
    p.add_argument("--gpus", type=str, default="0", help="comma separated GPU ids, one run per GPU at a time")
    p.add_argument("--tier", type=int, default=2, help="1 = core runs only, 2 = core + ablations")
    p.add_argument("--only", type=str, default="", help="comma separated run names to run (default: all of the tier)")
    p.add_argument("--extra", type=str, default="", help="extra flags appended to every run, e.g. '--seed 1'")
    p.add_argument("--summary", action="store_true", help="only print the result table")
    p.add_argument("--dry_run", action="store_true", help="print the commands without running them")
    return p.parse_args()


def selected_runs(args):
    runs = [r for r in RUNS if r[2] <= args.tier]
    if args.only:
        names = set(args.only.split(","))
        unknown = names - {r[0] for r in RUNS}
        if unknown:
            sys.exit(f"unknown run names: {sorted(unknown)}; available: {[r[0] for r in RUNS]}")
        runs = [r for r in RUNS if r[0] in names]
    return runs


def command(args, name, flags):
    here = os.path.dirname(os.path.abspath(__file__))
    return ([sys.executable, os.path.join(here, "start_split_brain.py"),
             "--datasetPath", args.datasetPath, "--out_dir", os.path.join(args.root, name)]
            + shlex.split(flags) + shlex.split(args.extra))


def done(args, name):
    return os.path.exists(os.path.join(args.root, name, "final_cer.json"))


def last_match(path, pattern):
    if not os.path.exists(path):
        return None
    last = None
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                last = m
    return last


def summary(args):
    rows = []
    for name, flags, tier, desc in RUNS:
        res_path = os.path.join(args.root, name, "final_cer.json")
        if not os.path.exists(res_path):
            continue
        with open(res_path) as f:
            r = json.load(f)
        log = os.path.join(args.root, name + ".log")
        tg = last_match(log, r"T_gain=(-?[\d.]+)")
        es = last_match(log, r"emb_std=(-?[\d.]+)")
        rows.append([name] + [r[k] for k, _ in COLS]
                    + [float(tg.group(1)) if tg else float("nan"), float(es.group(1)) if es else float("nan"), desc])
    if not rows:
        print(f"no finished runs in {args.root}/")
        return
    head = ["run"] + [c for _, c in COLS] + ["T_gain", "emb_std", "description"]
    print("\nCER (greedy CTC, no LM) -- lower is better")
    print(f"{head[0]:<16s}" + "".join(f"{h:>15s}" for h in head[1:-1]) + "  " + head[-1])
    for row in rows:
        print(f"{row[0]:<16s}" + "".join(f"{v:>15.4f}" for v in row[1:-1]) + "  " + row[-1])
    split_check = None
    for name, *_ in RUNS:
        split_check = split_check or last_match(os.path.join(args.root, name + ".log"), r"\[split check\].*")
    if split_check:
        print("\n" + split_check.group(0))
    with open(os.path.join(args.root, "summary.csv"), "w") as f:
        f.write(",".join(head) + "\n")
        for row in rows:
            f.write(",".join([row[0]] + [f"{v:.6f}" for v in row[1:-1]] + [f'"{row[-1]}"']) + "\n")
    print(f"saved {args.root}/summary.csv")


def main():
    args = parse()
    os.makedirs(args.root, exist_ok=True)
    if args.summary:
        summary(args)
        return

    queue = [r for r in selected_runs(args) if not done(args, r[0])]
    skipped = [r[0] for r in selected_runs(args) if done(args, r[0])]
    if skipped:
        print(f"already finished, skipping: {', '.join(skipped)}")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if args.dry_run:
        for i, (name, flags, _, desc) in enumerate(queue):
            print(f"CUDA_VISIBLE_DEVICES={gpus[i % len(gpus)]} {shlex.join(command(args, name, flags))}  # {desc}")
        return

    running = {}  # gpu -> (name, process, log file)
    failed = []
    try:
        while queue or running:
            for gpu in gpus:
                if gpu not in running and queue:
                    name, flags, _, desc = queue.pop(0)
                    log = open(os.path.join(args.root, name + ".log"), "a")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
                    proc = subprocess.Popen(command(args, name, flags), stdout=log, stderr=subprocess.STDOUT, env=env)
                    running[gpu] = (name, proc, log)
                    print(f"[{time.strftime('%H:%M:%S')}] start {name:<16s} on GPU {gpu}  ({desc})", flush=True)
            for gpu, (name, proc, log) in list(running.items()):
                if proc.poll() is not None:
                    log.close()
                    del running[gpu]
                    ok = proc.returncode == 0 and done(args, name)
                    if not ok:
                        failed.append(name)
                    print(f"[{time.strftime('%H:%M:%S')}] {'done' if ok else 'FAILED'} {name} "
                          f"(see {args.root}/{name}.log)", flush=True)
            time.sleep(10)
    except KeyboardInterrupt:
        for name, proc, log in running.values():
            proc.terminate()
            log.close()
        print("interrupted; rerun the same command to resume")
        raise
    if failed:
        print(f"failed runs: {', '.join(failed)}")
    summary(args)


if __name__ == "__main__":
    main()
