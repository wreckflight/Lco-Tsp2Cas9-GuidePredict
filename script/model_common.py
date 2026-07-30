#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared pieces for Tsp2 / Lco transfer / Lco scratch training scripts."""
from __future__ import print_function

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

import keras
from keras.callbacks import Callback, EarlyStopping
from keras.layers import (
    Input,
    Embedding,
    SpatialDropout1D,
    Bidirectional,
    LSTM,
    Flatten,
    Dense,
    Dropout,
)
from keras.models import Model
from keras.optimizers import Nadam

from prediction_util import make_data
from feature_util import featurize_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
PROJ = os.path.dirname(ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Combined loss weights (set before compile / fit)
alpha = 0.3
beta = 12.0

DEFAULT_DROPS = {"emb": 0.2, "rnn": 0.5, "rnn_rec": 0.1, "fc": 0.38}


def pairwise_hinge(y_true, y_pred, margin=0.1):
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    diff_true = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
    diff_pred = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)
    sign = tf.sign(diff_true)
    hinge = tf.nn.relu(margin - sign * diff_pred)
    mask = tf.linalg.band_part(tf.ones_like(hinge), 0, -1) - tf.linalg.band_part(
        tf.ones_like(hinge), 0, 0
    )
    loss_sum = tf.reduce_sum(hinge * mask)
    pair_cnt = tf.reduce_sum(mask)
    return loss_sum / (pair_cnt + 1e-8)


def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rank = pairwise_hinge(y_true, y_pred)
    return alpha * mse + beta * rank


class GetBest(Callback):
    """Keep best val_loss weights in memory and restore at train end."""

    def __init__(
        self, filepath=None, monitor="val_loss", save_best=False, verbose=0, mode="auto", period=1
    ):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.save_best = save_best
        self.filepath = filepath
        self.best_epochs = 0
        self.epochs_since_last_save = 0
        if mode not in ["auto", "min", "max"]:
            warnings.warn("GetBest mode %s is unknown, fallback to auto." % mode, RuntimeWarning)
            mode = "auto"
        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        current = logs.get(self.monitor)
        if current is None:
            return
        if self.monitor_op(current, self.best):
            if self.verbose > 0:
                print(
                    "\nEpoch %05d: %s improved from %0.5f to %0.5f, storing weights."
                    % (epoch + 1, self.monitor, self.best, current)
                )
            self.best = current
            self.best_epochs = epoch + 1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print(
                "Using epoch %05d with %s: %0.5f."
                % (self.best_epochs, self.monitor, self.best)
            )
        self.model.set_weights(self.best_weights)


def stratify_bins(y, n_bins=20):
    s = pd.Series(y)
    bins, _ = pd.qcut(
        s, q=min(n_bins, len(s) // 5), labels=False, retbins=True, duplicates="drop"
    )
    return bins.fillna(0).astype(int)


def build_bio_matrix(feature_dict):
    part_1 = feature_dict["dG_features"].values
    part_2 = feature_dict["gc_above_10"].values
    part_3 = feature_dict["gc_below_10"].values
    part_4 = feature_dict["gc_count"].values
    part_5 = feature_dict["Tm"].values
    return np.concatenate([part_1, part_2, part_3, part_4, part_5], axis=1)


def build_tabular_matrix(feature_dict, wide=False):
    base = build_bio_matrix(feature_dict).astype(np.float32)
    if not wide:
        return base
    extra_keys = ["_nuc_pd_Order1", "_nuc_pi_Order1", "_ba_pd_Order1", "_ba_pi_Order1"]
    blocks = [base]
    for key in extra_keys:
        if key not in feature_dict:
            raise KeyError("wide tabular needs %s" % key)
        blocks.append(feature_dict[key].values.astype(np.float32))
    return np.hstack(blocks)


def make_lco_model(
    seq_len,
    n_bio,
    em_dim=56,
    rnn_units=80,
    fc_units=400,
    fc_layers=3,
    drops=None,
    lr=0.001,
    output_activation="linear",
    loss_mode="combined",
):
    if drops is None:
        drops = DEFAULT_DROPS
    sequence_input = Input(name="seq_input", shape=(seq_len,))
    x = Embedding(7, em_dim, input_length=seq_len)(sequence_input)
    x = SpatialDropout1D(drops["emb"])(x)
    lstm = LSTM(
        rnn_units,
        dropout=drops["rnn"],
        kernel_regularizer="l2",
        recurrent_regularizer="l2",
        recurrent_dropout=drops["rnn_rec"],
        return_sequences=True,
    )
    x = Bidirectional(lstm)(x)
    x = Flatten()(x)
    biological_input = Input(name="bio_input", shape=(n_bio,))
    x = keras.layers.concatenate([x, biological_input])
    for _ in range(fc_layers):
        x = Dense(fc_units, activation="elu")(x)
        x = Dropout(drops["fc"])(x)
    mix_output = Dense(1, activation=output_activation, name="mix_output")(x)
    model = Model(inputs=[sequence_input, biological_input], outputs=[mix_output])
    loss = "mse" if loss_mode == "mse" else combined_loss
    model.compile(loss=loss, optimizer=Nadam(lr=lr))
    return model


def prepare_targets(df, mode, y_raw_col="percentage"):
    y_raw = df[y_raw_col].values.astype(np.float64)
    if mode == "raw_frac":
        y = y_raw / 100.0
    elif mode == "log1p":
        y = np.log1p(y_raw) / np.log1p(100.0)
    elif mode == "smoothed_frac":
        num = df["edited_count"].values + 0.5
        den = df["total_count"].values + 1.0
        y = num / den
    elif mode == "rank_norm":
        r = stats.rankdata(y_raw, method="average")
        y = (r - r.min()) / (r.max() - r.min() + 1e-9)
    elif mode == "rank_smoothed":
        num = df["edited_count"].values + 0.5
        den = df["total_count"].values + 1.0
        sm = 100.0 * num / den
        r = stats.rankdata(sm, method="average")
        y = (r - r.min()) / (r.max() - r.min() + 1e-9)
    elif mode == "gauss_rank":
        r = stats.rankdata(y_raw, method="average")
        u = (r - 0.5) / len(r)
        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        y = stats.norm.ppf(u)
    else:
        raise ValueError(mode)
    return y_raw, y


def resolve_output_activation(target_mode, output_activation_mode="auto"):
    if output_activation_mode == "auto":
        if target_mode in ("raw_frac", "log1p", "smoothed_frac"):
            return "sigmoid"
        return "linear"
    return output_activation_mode


def apply_read_depth_filter(df, min_count=100, count_inclusive=False):
    if min_count <= 0:
        return df.reset_index(drop=True)
    if count_inclusive:
        df = df[df["total_count"] >= min_count].copy()
    else:
        df = df[df["total_count"] > min_count].copy()
    return df.reset_index(drop=True)


def _normalize_count_columns(df):
    """Unify Lco/Tsp2 CSV schemas (Tsp2 KOD table uses unique_edited_reads)."""
    if "edited_count" not in df.columns and "unique_edited_reads" in df.columns:
        df = df.copy()
        df["edited_count"] = df["unique_edited_reads"]
    return df


def load_xy_arrays(
    csv_path,
    protein_key,
    min_count=100,
    tabular="minimal",
    feature_order=1,
    num_proc=4,
    skip_cache="",
    save_cache="",
    label_col="percentage",
):
    df = pd.read_csv(csv_path)
    df = _normalize_count_columns(df)
    df = apply_read_depth_filter(df, min_count=min_count)
    df["21mer"] = df["gRNA"].astype(str)

    if skip_cache:
        with open(skip_cache, "rb") as f:
            cache = pickle.load(f)
        X_seq = cache["X_seq"]
        X_bio = cache["X_bio_wide"] if tabular == "wide" else cache["X_bio"]
        if len(X_seq) != len(df):
            raise ValueError(
                "Cache rows %d != CSV rows %d for %s" % (len(X_seq), len(df), csv_path)
            )
        print("Loaded cached features:", skip_cache)
        return df, X_seq, X_bio

    feature_options = {
        "protein": protein_key,
        "testing_non_binary_target_name": "ranks",
        "include_pi_nuc_feat": True,
        "gc_features": True,
        "nuc_features": True,
        "include_Tm": True,
        "include_structure_features": True,
        "order": feature_order,
        "num_proc": num_proc,
        "normalize_features": None,
    }
    print(
        "Featurizing %s (%d rows) protein=%s ..."
        % (os.path.basename(csv_path), len(df), protein_key)
    )
    feat = featurize_data(df[["21mer"]], feature_options, quiet=False)
    X_bio = build_tabular_matrix(feat, wide=(tabular == "wide"))
    X_seq = make_data(df["21mer"])
    if save_cache:
        payload = {
            "X_seq": X_seq,
            "X_bio": build_bio_matrix(feat).astype(np.float32),
            "X_bio_wide": build_tabular_matrix(feat, wide=True),
            "tabular": tabular,
            "protein": protein_key,
            "csv": os.path.abspath(csv_path),
        }
        with open(save_cache, "wb") as f:
            pickle.dump(payload, f, protocol=2)
        print("Wrote feature cache", save_cache)
        X_bio = payload["X_bio_wide"] if tabular == "wide" else payload["X_bio"]
    return df, X_seq, X_bio


def train_rnn(
    Xs_tr,
    Xb_tr,
    y_tr,
    seq_len,
    n_bio,
    out_hd5,
    seed=42,
    epochs=45,
    batch_size=96,
    lr=0.001,
    patience=7,
    val_split=0.1,
    em_dim=56,
    rnn_units=80,
    fc_units=400,
    fc_layers=3,
    output_activation="sigmoid",
    loss_mode="combined",
    init_weights=None,
    loss_alpha=0.3,
    loss_beta=12.0,
    verbose=2,
):
    """Fit one BiLSTM+bio model; restores best val weights; saves full model to out_hd5."""
    global alpha, beta
    alpha, beta = float(loss_alpha), float(loss_beta)
    np.random.seed(seed)
    model = make_lco_model(
        seq_len,
        n_bio,
        em_dim=em_dim,
        rnn_units=rnn_units,
        fc_units=fc_units,
        fc_layers=fc_layers,
        drops=DEFAULT_DROPS,
        lr=lr,
        output_activation=output_activation,
        loss_mode=loss_mode,
    )
    if init_weights and os.path.isfile(init_weights):
        try:
            model.load_weights(init_weights, by_name=False)
            print("Loaded weights:", init_weights)
        except ValueError as exc:
            print("Strict load failed (%s); loading by_name." % exc)
            model.load_weights(init_weights, by_name=True)
            print("Partial init from:", init_weights)
    early = EarlyStopping(monitor="val_loss", patience=patience, verbose=0)
    get_best = GetBest(out_hd5, monitor="val_loss", verbose=0, mode="min")
    model.fit(
        [Xs_tr, Xb_tr],
        y_tr,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=val_split,
        shuffle=False,
        callbacks=[get_best, early],
    )
    os.makedirs(os.path.dirname(out_hd5) or ".", exist_ok=True)
    model.save(out_hd5)
    print("Wrote", out_hd5)
    return model


def resolve_csv(*candidates):
    """Return first existing path among candidates."""
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError("CSV not found. Tried: %s" % (candidates,))


def save_train_meta(path, meta):
    with open(path, "wb") as f:
        pickle.dump(meta, f, protocol=2)
    print("Wrote", path)
