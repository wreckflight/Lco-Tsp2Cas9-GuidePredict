# Pre-LcoCas9

**Pre-LcoCas9** is a deep learning framework for predicting sgRNA activity of **Listeria costaricensis Cas9 (LcoCas9)**. The model was developed using a high-throughput dataset of approximately **20,000 sgRNA–target pairs** and further improved by transfer learning from the closely related **Tsp2Cas9**.

This repository contains the source code, trained models, and scripts for reproducing the prediction results described in our publication.

<p align="center">
  <img src="figures/overview.png" width="900" alt="Overview of the Pre-LcoCas9 framework">
</p>
<p align="center"><em>Overview of the Pre-LcoCas9 framework.</em></p>

## Online Prediction

Try the online predictor here:

**https://xxxx.edu.cn/Pre-LcoCas9**

*(Replace with the public web-server URL when available.)*

---

### System requirements

The code were tested on Linux and Mac OS systems.

> #### Note:
> - Keras should be run with tensorflow as its backend.
> - ViennaRNA, a C code library for prediction of RNA secondary structure, needs to be downloaded before installation.

The required software/packages are:

* python=3.6.5
* numpy=1.14.0
* scipy=1.0.0
* h5py=2.7.1
* tensorflow=1.8.0
* keras=2.1.6
* scikit-learn=0.19.1
* biopython=1.71
* viennarna=2.4.5
* matplotlib
* DotMap
* GPyOpt
* pandas

It is worth noting that when the computing environment (e.g., the version of tensorflow or biopython) changes, the prediction results might change slightly, but the main conclusion won't be affected.

### Installation Guide

```bash
git clone https://github.com/wreckflight/Lco-Tsp2Cas9-GuidePredict.git
cd Lco-Tsp2Cas9-GuidePredict

conda create -n crispr python=3.6.5 ipykernel matplotlib pandas numpy=1.14.0 scipy=1.0.0 h5py=2.7.1 tensorflow=1.8.0 keras=2.1.6 scikit-learn=0.19.1 biopython=1.71 viennarna=2.4.5
conda activate crispr
pip install GPyOpt
pip install DotMap
ipython kernel install --user --name crispr --display-name "Python3(crispr)"
```

Installation time depends on your own network environment.

Alternatively, after creating a compatible Python 3.6 environment:

```bash
pip install -r requirements.txt
```

### Training data for reproduction

Feature matrices are provided under `data/`. Training scripts also need the corresponding KOD-corrected activity CSVs for labels and row alignment. Place them under `data/`, or pass `--csv`:

| Dataset | Expected CSV name |
|---------|-------------------|
| Tsp2Cas9 | `Tsp-12d_merged_fasta_gRNA_results_legacy_dp_KOD_corrected.csv` |
| LcoCas9  | `Lco-12d_merged_fasta_gRNA_results_legacy_dp_KOD_corrected.csv` |

Default search order: `data/<csv>` → sibling folders `Tsp_data/` / `Lco_data/` when present.

---

### Prediction

Provide **20-nt sgRNA spacer** sequences. The pipeline automatically extracts biological / structure features and runs the published model.

**Command line**

```bash
# Single spacer (LcoCas9 main model by default)
python script/predict.py --seq GTAACCGCGGCTCTCGGTGA

# Multiple spacers
python script/predict.py \
  --seq GTAACCGCGGCTCTCGGTGA \
  --seq GGAATCTGGTGAAGTATCAC \
  --model lco

# From a CSV (column sgRNA or gRNA) / TXT (one spacer per line)
python script/predict.py \
  --input examples/example_sgrnas.csv \
  --output prediction.csv \
  --model lco
```

**Python API**

```python
import sys
sys.path.insert(0, "script")

from prediction_util import predict_sgrnas

df = predict_sgrnas(
    ["GTAACCGCGGCTCTCGGTGA", "GGAATCTGGTGAAGTATCAC"],
    model_type="lco",   # or "tsp2"
)
print(df)
#                sgRNA  Efficiency
# GTAACCGCGGCTCTCGGTGA    0.283...
```

Higher scores indicate higher predicted editing activity (clipped to \[0, 1\] for main sigmoid models).

| Key | Checkpoint | Description |
|-----|------------|-------------|
| `lco` | `model/Lco_main.hd5` | LcoCas9 transfer (recommended) |
| `tsp2` | `model/Tsp2_main.hd5` | Tsp2Cas9 main model |
| — | `model/Lco_scratch.hd5` | LcoCas9 scratch baseline |

---

### Reproducing the Results

This repository contains scripts necessary to reproduce the major modeling analyses reported in the manuscript, including feature loading, model training, transfer learning, and hold-out evaluation.

Run all commands from the repository root with the `crispr` environment activated.

**Train Tsp2Cas9 (main)**

```bash
python script/train_tsp2.py
```

```bash
python script/train_tsp2.py \
  --csv /path/to/Tsp-12d_merged_fasta_gRNA_results_legacy_dp_KOD_corrected.csv \
  --cache data/Tsp2_feat_data.pkl \
  --out-model model/Tsp2_main.hd5 \
  --seed 42 \
  --epochs 45 \
  --batch-size 96 \
  --lr 0.001 \
  --alpha 0.3 \
  --beta 12.0
```

**Train LcoCas9 by transfer learning (main)**

Requires `model/Tsp2_main.hd5` (or pass `--init-weights`).

```bash
python script/train_lco_transfer.py
```

```bash
python script/train_lco_transfer.py \
  --csv /path/to/Lco-12d_merged_fasta_gRNA_results_legacy_dp_KOD_corrected.csv \
  --cache data/Lco_feat_data.pkl \
  --init-weights model/Tsp2_main.hd5 \
  --out-model model/Lco_main.hd5 \
  --seed 42 \
  --epochs 45 \
  --finetune-lr 0.00035 \
  --alpha 0.3 \
  --beta 12.0
```

**Train LcoCas9 scratch baseline (no transfer)**

```bash
python script/train_lco_scratch.py
```

```bash
python script/train_lco_scratch.py \
  --csv /path/to/Lco-12d_merged_fasta_gRNA_results_legacy_dp_KOD_corrected.csv \
  --cache data/Lco_feat_data.pkl \
  --out-model model/Lco_scratch.hd5 \
  --seed 42
```

Each run writes a Keras checkpoint (`*.hd5`) and a companion metadata file (`*_train_meta.pkl`) including the reported hold-out Spearman correlation.

> **Tip.** To avoid overwriting published weights during tests, use `--out-model /tmp/smoke.hd5` and `--epochs 1` for `train_tsp2.py` / `train_lco_transfer.py`.

The optimized hyperparameters are only fit for the aforementioned software/package environment.

---

### Files description

* [script/feature_util.py](script/feature_util.py) contains the code for extracting position-related features and biological features (adapted from DeepHF).

* [script/prediction_util.py](script/prediction_util.py) contains sequence encoding helpers and the end-to-end `predict_sgrnas()` API (20-nt spacer → Efficiency).

* [script/predict.py](script/predict.py) is the command-line prediction entry point for the website / local use.

* [examples/example_sgrnas.csv](examples/example_sgrnas.csv) contains example 20-nt spacers for a quick prediction demo.

* [script/model_common.py](script/model_common.py) provides the shared BiLSTM+bio architecture, combined loss, training callbacks, and data-loading utilities.

* [script/train_tsp2.py](script/train_tsp2.py) trains the Tsp2Cas9 main model in your own computing environment.

* [script/train_lco_transfer.py](script/train_lco_transfer.py) fine-tunes the LcoCas9 main model from Tsp2Cas9 pretrained weights (transfer learning).

* [script/train_lco_scratch.py](script/train_lco_scratch.py) trains the LcoCas9 BiLSTM+bio scratch baseline without transfer.

* [data/Tsp2_feat_data.pkl](data/Tsp2_feat_data.pkl), features for Tsp2Cas9. It can be used to train the model (together with the KOD-corrected activity CSV).

* [data/Lco_feat_data.pkl](data/Lco_feat_data.pkl), features for LcoCas9. It can be used to train the model (together with the KOD-corrected activity CSV).

* [model/Tsp2_main.hd5](model/Tsp2_main.hd5), the final model file of Tsp2Cas9.

* [model/Lco_main.hd5](model/Lco_main.hd5), the final model file of LcoCas9 (Tsp2 → Lco transfer) used as the main predictor.

* [model/Lco_scratch.hd5](model/Lco_scratch.hd5), the final model file of the LcoCas9 scratch baseline (BiLSTM+bio, no transfer).

Each `.hd5` model has a companion `*_train_meta.pkl` recording training settings and hold-out Spearman correlation.

---

### Model Architecture

Pre-LcoCas9 consists of:

- Character-level sequence embedding
- Bidirectional LSTM (BiLSTM) encoder
- Hand-crafted biological / thermodynamic features
- Fully connected layers
- Output head (sigmoid for main models; linear for the scratch baseline recipe)

Transfer learning initializes the network with weights pretrained on the Tsp2Cas9 dataset and subsequently fine-tunes on LcoCas9 data (default fine-tune learning rate `3.5e-4`).

---

### Performance

Hold-out evaluation (15% test split, seed = 42; Spearman correlation vs. measured activity):

| Model | Description | Spearman |
|-------|-------------|---------:|
| **Lco_main** | Tsp2 → Lco transfer (main) | **0.756** |
| Lco_scratch | BiLSTM+bio without transfer | 0.744 |
| Tsp2_main | Tsp2Cas9 main | 0.908 |

Transfer learning improves LcoCas9 prediction relative to training from scratch on LcoCas9 data alone.
