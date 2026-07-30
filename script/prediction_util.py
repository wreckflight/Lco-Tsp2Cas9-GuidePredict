#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sequence encoding + prediction helpers for Lco / Tsp2 main models.

Importing this module does not load Keras weights (safe for training scripts
that only need make_data).

Typical usage::

    from prediction_util import predict_sgrnas
    df = predict_sgrnas(["GTAACCGCGGCTCTCGGTGA"], model_type="lco")
"""
from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
from keras.preprocessing import text, sequence

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Published main checkpoints under model/
TSP2_MAIN_MODEL = os.path.join(ROOT, "model/Tsp2_main.hd5")
TSP2_MAIN_META = os.path.join(ROOT, "model/Tsp2_main_train_meta.pkl")
LCO_MAIN_MODEL = os.path.join(ROOT, "model/Lco_main.hd5")
LCO_MAIN_META = os.path.join(ROOT, "model/Lco_main_train_meta.pkl")

MODEL_PATHS = {
    "tsp2": TSP2_MAIN_MODEL,
    "lco": LCO_MAIN_MODEL,
}

# model_type → ViennaRNA / scaffold protein key in feature_util
PROTEIN_KEYS = {
    "tsp2": "Tsp2",
    "lco": "Lco",
}


def make_data(X):
    """Char-level tokenization for 20-nt spacers (START + A/T/C/G).

    Input strings are 20-nt sgRNA spacers. A START token is prepended, so the
    returned array has shape (N, 21). The column name ``21mer`` elsewhere is a
    DeepHF legacy label for the same 20-nt sequences.
    """
    vectorizer = text.Tokenizer(lower=False, split=" ", num_words=None, char_level=True)
    vectorizer.fit_on_texts(X)
    alphabet = "ATCG"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    word_index = {k: (v + 1) for k, v in char_dict.items()}
    word_index["PAD"] = 0
    word_index["START"] = 1
    vectorizer.word_index = word_index.copy()
    X = vectorizer.texts_to_sequences(X)
    X = [[word_index["START"]] + [w for w in x] for x in X]
    X = sequence.pad_sequences(X)
    return X


def normalize_sgrna(seq):
    """Uppercase DNA spacer; accept A/T/C/G/U (U→T)."""
    s = str(seq).strip().upper().replace("U", "T")
    if len(s) != 20:
        raise ValueError("sgRNA must be 20 nt, got length %d: %r" % (len(s), seq))
    bad = sorted(set(c for c in s if c not in "ATCG"))
    if bad:
        raise ValueError("sgRNA has non-ACGT characters %s: %r" % (bad, seq))
    return s


def _as_sequence_list(sequences):
    if isinstance(sequences, (str, bytes)):
        sequences = [sequences]
    elif isinstance(sequences, pd.Series):
        sequences = sequences.tolist()
    else:
        sequences = list(sequences)
    if not sequences:
        raise ValueError("No sgRNA sequences provided")
    return [normalize_sgrna(s) for s in sequences]


def resolve_model_path(model_type="lco"):
    """Return .hd5 path for 'lco' / 'tsp2'."""
    key = str(model_type).lower()
    if key not in MODEL_PATHS:
        raise ValueError("Unknown model_type %r; use one of %s" % (model_type, list(MODEL_PATHS)))
    path = MODEL_PATHS[key]
    if not os.path.isfile(path):
        raise FileNotFoundError("Model file not found: %s" % path)
    return path


def resolve_meta_path(model_type="lco"):
    key = str(model_type).lower()
    if key == "tsp2":
        path = TSP2_MAIN_META
    elif key == "lco":
        path = LCO_MAIN_META
    else:
        raise ValueError("Unknown model_type %r" % model_type)
    if not os.path.isfile(path):
        raise FileNotFoundError("Meta file not found: %s" % path)
    return path


def load_main_model(model_type="lco", custom_objects=None):
    """Load a published main .hd5 (full Keras model)."""
    from keras.models import load_model

    path = resolve_model_path(model_type)
    kwargs = {}
    if custom_objects is None:
        # Lazy import avoids circular import with model_common at module load.
        from model_common import combined_loss

        custom_objects = {"combined_loss": combined_loss}
    kwargs["custom_objects"] = custom_objects
    return load_model(path, **kwargs)


def prepare_inputs(sequences, model_type="lco", num_proc=1, quiet=True):
    """Featurize 20-nt sgRNA spacers → (X_seq, X_bio) for the BiLSTM+bio model.

    Parameters
    ----------
    sequences : str | list[str] | pandas.Series
        20-nt spacer sequence(s).
    model_type : {'lco', 'tsp2'}
        Selects the tracr scaffold used for structure features.
    """
    from feature_util import featurize_data
    from model_common import build_tabular_matrix

    seqs = _as_sequence_list(sequences)
    key = str(model_type).lower()
    if key not in PROTEIN_KEYS:
        raise ValueError("Unknown model_type %r; use one of %s" % (model_type, list(PROTEIN_KEYS)))

    feature_options = {
        "protein": PROTEIN_KEYS[key],
        "testing_non_binary_target_name": "ranks",
        "include_pi_nuc_feat": True,
        "gc_features": True,
        "nuc_features": True,
        "include_Tm": True,
        "include_structure_features": True,
        "order": 1,
        "num_proc": int(num_proc),
        "normalize_features": None,
    }
    data = pd.DataFrame({"21mer": seqs})
    feat = featurize_data(data, feature_options, quiet=quiet)
    X_bio = build_tabular_matrix(feat, wide=False).astype(np.float32)
    X_seq = make_data(seqs)
    return X_seq, X_bio, seqs


def predict_efficiency(model, X_seq, X_bio, clip=True):
    """Run model.predict; optionally clip to [0, 1] (sigmoid heads)."""
    pred = model.predict([X_seq, X_bio], batch_size=96, verbose=0).ravel()
    if clip:
        pred = np.clip(pred, 0.0, 1.0)
    return pred


def attach_predictions(df, pred, seq_col="21mer"):
    """Add gRNA_Seq / Efficiency columns and sort by Efficiency descending."""
    out = df.copy()
    if seq_col in out.columns:
        out["gRNA_Seq"] = out[seq_col].astype(str)
    out["Efficiency"] = pred
    out = out.reset_index(drop=True)
    return out.sort_values(by="Efficiency", ascending=False)


def predict_sgrnas(
    sequences,
    model_type="lco",
    model=None,
    num_proc=1,
    clip=True,
    quiet=True,
):
    """Predict editing efficiency from 20-nt sgRNA spacer sequence(s).

    Returns a DataFrame with columns ``sgRNA`` and ``Efficiency``, sorted by
    score (descending).
    """
    X_seq, X_bio, seqs = prepare_inputs(
        sequences, model_type=model_type, num_proc=num_proc, quiet=quiet
    )
    if model is None:
        model = load_main_model(model_type)
    pred = predict_efficiency(model, X_seq, X_bio, clip=clip)
    out = pd.DataFrame({"sgRNA": seqs, "Efficiency": pred})
    return out.sort_values(by="Efficiency", ascending=False).reset_index(drop=True)
