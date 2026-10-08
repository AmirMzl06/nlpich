"""Sanity check: do the unseen-day (online evaluation) sentences also appear in the training data?

If many test sentences were seen during training, a large bidirectional decoder can partly
memorise them and the CER is optimistic compared with papers that test on new sentences.

    python check_sentence_overlap.py --datasetPath /data/hossein/mm_project/CORP_data_release
    python check_sentence_overlap.py --datasetPath ... --out_dir sb_runs/none_in08   # + CER on new vs repeated sentences
"""
import argparse
import os
import pickle
import re

from start_split_brain import SEED_DIR, NO_RECAL_DIR, RECAL_DIR, load_train_items, load_eval_items, cer_of


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasetPath", type=str, default="/data/hossein/mm_project/CORP_data_release")
    p.add_argument("--out_dir", type=str, default="", help="run folder with evalStats.pkl (optional)")
    args = p.parse_args()

    train_sents = {norm(it[1]) for it in load_train_items(args.datasetPath)}
    eval_items, eval_tags = load_eval_items(args.datasetPath)
    print(f"\ntraining sentences (unique): {len(train_sents)}")

    groups = {}
    for (x, s, d), (split, name) in zip(eval_items, eval_tags):
        groups.setdefault(split, []).append(norm(s) in train_sents)
    for split, flags in groups.items():
        print(f"{split:>9s}: {len(flags):5d} trials, {sum(flags):5d} ({100 * sum(flags) / max(len(flags), 1):5.1f}%) "
              f"have a sentence that is also in the training data")

    stats_path = os.path.join(args.out_dir, "evalStats.pkl") if args.out_dir else ""
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            per_trial = pickle.load(f)  # same order as eval_items
        assert len(per_trial) == len(eval_items), "evalStats.pkl does not match this dataset"
        print(f"\nCER of {args.out_dir} on unseen days, split by sentence novelty:")
        for split in ("no_recal", "recal"):
            seen_s = [r for r, (it, t) in zip(per_trial, zip(eval_items, eval_tags))
                      if t[0] == split and norm(it[1]) in train_sents]
            new_s = [r for r, (it, t) in zip(per_trial, zip(eval_items, eval_tags))
                     if t[0] == split and norm(it[1]) not in train_sents]
            print(f"{split:>9s}: repeated sentences CER {cer_of(seen_s):.4f} ({len(seen_s)} trials) | "
                  f"new sentences CER {cer_of(new_s):.4f} ({len(new_s)} trials)")


if __name__ == "__main__":
    main()
