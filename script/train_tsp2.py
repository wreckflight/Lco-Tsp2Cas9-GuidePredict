#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train Tsp2 main model (BiLSTM+bio) → model/Tsp2_main.hd5

Default recipe matches the published Tsp2 main checkpoint meta
(smoothed_frac + sigmoid + combined loss, seed=42, 15% hold-out).
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


def main():
    p = argparse.ArgumentParser(description="Train Tsp2 main BiLSTM+bio model")
    p.add_argument(
        "--csv",
        default="",
        help="Tsp2 KOD-corrected CSV (default: search data/ then project Tsp_data/)",
    )
    p.add_argument(
        "--cache",
        default=os.path.join(ROOT, "data/Tsp2_feat_data.pkl"),
    )
    p.add_argument(
        "--out-model",
        default=os.path.join(ROOT, "model/Tsp2_main.hd5"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=12.0)
    args = p.parse_args()

    csv_path = args.csv or mc.resolve_csv(
        os.path.join(ROOT, "data/Tsp-12d_merged_fasta_gRNA_results_legacy_dp_KOD_corrected.csv"),
        os.path.join(PROJ, "Tsp_data/Tsp-12d_merged_fasta_gRNA_results_legacy_dp_KOD_corrected.csv"),
    )

    df, X_seq, X_bio = mc.load_xy_arrays(
        csv_path,
        "Tsp2",
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
        "Tsp2 main | seed=%d n_train=%d n_test=%d" % (args.seed, len(tr_idx), len(te_idx)),
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
        lr=args.lr,
        output_activation=out_act,
        loss_mode="combined",
        loss_alpha=args.alpha,
        loss_beta=args.beta,
    )
    pred = m.predict([X_seq[te_idx], X_bio[te_idx]], batch_size=96, verbose=0).ravel()
    sp = float(stats.spearmanr(y_raw[te_idx], pred).correlation)
    print("Hold-out Spearman=%.6f" % sp, flush=True)

    meta = {
        "model_name": "Tsp2 (main)",
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
        "pretrain_weights": None,
        "seed": args.seed,
        "test_size": args.test_size,
        "spearman_best_reported": sp,
    }
    mc.save_train_meta(args.out_model.replace(".hd5", "_train_meta.pkl"), meta)


if __name__ == "__main__":
    main()
