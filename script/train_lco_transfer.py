#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train Lco transfer main model (Tsp2→Lco) → model/Lco_main.hd5

Loads Tsp2 main weights, then fine-tunes on Lco (finetune_lr=3.5e-4 by default).
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

import model_common as mc
from model_common import PROJ, ROOT, SCRIPT_DIR

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from keras import backend as K

DEFAULT_INIT = os.path.join(ROOT, "model/Tsp2_main.hd5")
DEFAULT_OUT = os.path.join(ROOT, "model/Lco_main.hd5")


def main():
    p = argparse.ArgumentParser(description="Train Lco Tsp2→Lco transfer main model")
    p.add_argument("--csv", default="")
    p.add_argument(
        "--cache",
        default=os.path.join(ROOT, "data/Lco_feat_data.pkl"),
    )
    p.add_argument("--init-weights", default=DEFAULT_INIT)
    p.add_argument("--out-model", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--finetune-lr", type=float, default=0.00035)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=12.0)
    args = p.parse_args()

    csv_path = args.csv or mc.resolve_csv(
        os.path.join(ROOT, "data/Lco-12d_merged_fasta_gRNA_results_legacy_dp_KOD_corrected.csv"),
        os.path.join(PROJ, "Lco_data/Lco-12d_merged_fasta_gRNA_results_legacy_dp_KOD_corrected.csv"),
    )
    if not os.path.isfile(args.init_weights):
        raise FileNotFoundError(
            "Missing Tsp2 init weights: %s (run train_tsp2.py first)" % args.init_weights
        )

    df, X_seq, X_bio = mc.load_xy_arrays(
        csv_path,
        "Lco",
        min_count=100,
        tabular="minimal",
        skip_cache=args.cache if os.path.isfile(args.cache) else "",
        label_col="percentage",
    )
    y_raw, y = mc.prepare_targets(df, "smoothed_frac", y_raw_col="percentage")
    bins = mc.stratify_bins(y_raw)
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=bins
    )
    seq_len, n_bio = int(X_seq.shape[1]), int(X_bio.shape[1])
    out_act = "sigmoid"

    print(
        "Lco transfer | init=%s | seed=%d n_train=%d n_test=%d"
        % (os.path.basename(args.init_weights), args.seed, len(tr_idx), len(te_idx)),
        flush=True,
    )
    K.clear_session()
    m = mc.train_rnn(
        X_seq[tr_idx],
        X_bio[tr_idx],
        y[tr_idx],
        seq_len,
        n_bio,
        args.out_model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.finetune_lr,
        output_activation=out_act,
        loss_mode="combined",
        init_weights=args.init_weights,
        loss_alpha=args.alpha,
        loss_beta=args.beta,
    )
    pred = m.predict([X_seq[te_idx], X_bio[te_idx]], batch_size=96, verbose=0).ravel()
    sp = float(stats.spearmanr(y_raw[te_idx], pred).correlation)
    print("Hold-out Spearman=%.6f" % sp, flush=True)

    meta = {
        "model_name": "Tsp2→Lco (main)",
        "target": "smoothed_frac",
        "label_col": "percentage",
        "output_activation": out_act,
        "loss": "combined",
        "alpha": args.alpha,
        "beta": args.beta,
        "seq_len": seq_len,
        "n_bio": n_bio,
        "tabular": "minimal",
        "csv": os.path.abspath(csv_path),
        "cache": os.path.abspath(args.cache) if os.path.isfile(args.cache) else None,
        "pretrain_weights": os.path.abspath(args.init_weights),
        "finetune_lr_used": float(args.finetune_lr),
        "seed": args.seed,
        "test_size": args.test_size,
        "spearman_best_reported": sp,
    }
    mc.save_train_meta(args.out_model.replace(".hd5", "_train_meta.pkl"), meta)


if __name__ == "__main__":
    main()
