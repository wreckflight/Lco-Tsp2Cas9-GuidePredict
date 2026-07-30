#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Predict LcoCas9 / Tsp2Cas9 sgRNA activity from 20-nt spacer sequences.

Examples
--------
Predict one or more spacers on the command line::

    python script/predict.py --seq GTAACCGCGGCTCTCGGTGA
    python script/predict.py --seq GTAACCGCGGCTCTCGGTGA --seq GGAATCTGGTGAAGTATCAC

Predict from a CSV/TSV/TXT file (column ``sgRNA`` or ``gRNA``, or one sequence
per line)::

    python script/predict.py --input examples/example_sgrnas.csv --output prediction.csv
"""
from __future__ import print_function

import argparse
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import prediction_util as pu


def _read_sequences(path, seq_col=""):
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if seq_col:
            if seq_col not in df.columns:
                raise ValueError("Column %r not found in %s" % (seq_col, path))
            return df[seq_col].astype(str).tolist()
        for cand in ("sgRNA", "gRNA", "21mer", "sequence", "Sequence"):
            if cand in df.columns:
                return df[cand].astype(str).tolist()
        raise ValueError(
            "No sequence column found in %s; expected one of "
            "sgRNA/gRNA/21mer/sequence (or pass --seq-col)" % path
        )
    # Plain text: one spacer per line
    seqs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seqs.append(line.split()[0])
    return seqs


def main():
    p = argparse.ArgumentParser(
        description="Predict sgRNA activity with Pre-LcoCas9 / Tsp2 main models"
    )
    p.add_argument(
        "--seq",
        action="append",
        default=[],
        help="20-nt spacer (repeat flag for multiple sequences)",
    )
    p.add_argument(
        "--input",
        default="",
        help="CSV/TSV with sgRNA/gRNA column, or TXT with one spacer per line",
    )
    p.add_argument("--seq-col", default="", help="Sequence column name for --input CSV/TSV")
    p.add_argument(
        "--model",
        default="lco",
        choices=["lco", "tsp2"],
        help="Published model to use (default: lco)",
    )
    p.add_argument(
        "--output",
        default="",
        help="Write predictions CSV (default: print to stdout)",
    )
    p.add_argument("--num-proc", type=int, default=1, help="Feature extraction workers")
    p.add_argument("--no-clip", action="store_true", help="Do not clip scores to [0, 1]")
    args = p.parse_args()

    seqs = list(args.seq)
    if args.input:
        seqs.extend(_read_sequences(args.input, seq_col=args.seq_col))
    if not seqs:
        p.error("Provide at least one --seq or --input file")

    print("Predicting %d spacer(s) with model=%s ..." % (len(seqs), args.model), flush=True)
    out = pu.predict_sgrnas(
        seqs,
        model_type=args.model,
        num_proc=args.num_proc,
        clip=not args.no_clip,
        quiet=True,
    )

    if args.output:
        out_path = os.path.abspath(args.output)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out.to_csv(out_path, index=False)
        print("Wrote", out_path, flush=True)
    else:
        print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
