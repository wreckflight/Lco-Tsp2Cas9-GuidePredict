#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train Lco BiLSTM+bio scratch (no transfer) → model/Lco_scratch.hd5

Matches the figures_latest BiLSTM_bio_scratch hold-out recipe
(combined loss alpha=0.4 beta=8, linear head, inner 10% val split).
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
from keras.callbacks import EarlyStopping

DEFAULT_OUT = os.path.join(ROOT, "model/Lco_scratch.hd5")


def _train_scratch(Xs_tr, Xb_tr, y_tr, y_raw_tr, seq_len, n_bio, seed, out_hd5, out_act):
    mc.alpha, mc.beta = 0.4, 8.0
    np.random.seed(seed)
    m = mc.make_lco_model(
        seq_len,
        n_bio,
        em_dim=56,
        rnn_units=80,
        fc_units=400,
        fc_layers=3,
        lr=0.001,
        output_activation=out_act,
        loss_mode="combined",
    )
    va_bins = mc.stratify_bins(y_raw_tr)
    tr_i, va_i = train_test_split(
        np.arange(len(y_raw_tr)),
        test_size=0.1,
        random_state=seed,
        stratify=va_bins,
    )
    es = EarlyStopping(monitor="val_loss", patience=7, verbose=0)
    gb = mc.GetBest(out_hd5, monitor="val_loss", verbose=0, mode="min")
    m.fit(
        [Xs_tr[tr_i], Xb_tr[tr_i]],
        y_tr[tr_i],
        validation_data=([Xs_tr[va_i], Xb_tr[va_i]], y_tr[va_i]),
        batch_size=96,
        epochs=45,
        shuffle=False,
        callbacks=[gb, es],
        verbose=2,
    )
    os.makedirs(os.path.dirname(out_hd5) or ".", exist_ok=True)
    m.save(out_hd5)
    print("Wrote", out_hd5)
    return m


def main():
    p = argparse.ArgumentParser(description="Train Lco BiLSTM+bio scratch (no transfer)")
    p.add_argument("--csv", default="")
    p.add_argument(
        "--cache",
        default=os.path.join(ROOT, "data/Lco_feat_data.pkl"),
    )
    p.add_argument("--out-model", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.15)
    args = p.parse_args()

    csv_path = args.csv or mc.resolve_csv(
        os.path.join(ROOT, "data/Lco-12d_merged_fasta_gRNA_results_legacy_dp_KOD_corrected.csv"),
        os.path.join(PROJ, "Lco_data/Lco-12d_merged_fasta_gRNA_results_legacy_dp_KOD_corrected.csv"),
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
    # Scratch hold-out figure uses linear head for smoothed_frac (not sigmoid).
    out_act = "linear"

    print(
        "Lco scratch | seed=%d n_train=%d n_test=%d act=%s"
        % (args.seed, len(tr_idx), len(te_idx), out_act),
        flush=True,
    )
    K.clear_session()
    m = _train_scratch(
        X_seq[tr_idx],
        X_bio[tr_idx],
        y[tr_idx],
        y_raw[tr_idx],
        seq_len,
        n_bio,
        args.seed,
        args.out_model,
        out_act,
    )
    pred = m.predict([X_seq[te_idx], X_bio[te_idx]], batch_size=96, verbose=0).ravel()
    sp = float(stats.spearmanr(y_raw[te_idx], pred).correlation)
    print("Hold-out Spearman=%.6f" % sp, flush=True)

    meta = {
        "model_name": "BiLSTM (seq+bio, scratch)",
        "plot_title": "BiLSTM+bio scratch",
        "target": "smoothed_frac",
        "label_col": "percentage",
        "output_activation": out_act,
        "loss": "combined",
        "alpha": 0.4,
        "beta": 8.0,
        "seq_len": seq_len,
        "n_bio": n_bio,
        "tabular": "minimal",
        "csv": os.path.abspath(csv_path),
        "cache": os.path.abspath(args.cache) if os.path.isfile(args.cache) else None,
        "pretrain_weights": None,
        "seed": args.seed,
        "test_size": args.test_size,
        "spearman_best_reported": sp,
        "recipe": "BiLSTM+bio scratch hold-out (seed=42)",
    }
    mc.save_train_meta(args.out_model.replace(".hd5", "_train_meta.pkl"), meta)


if __name__ == "__main__":
    main()
