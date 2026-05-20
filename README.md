# PARCE Material Structure Prediction System

[中文说明](README_zh.md)

PARCE is a machine-learning workflow for predicting material structure types. It takes phonon frequencies or Raman-active frequencies as input, combines structural-parameter clustering with the RCNet deep-learning model, and predicts the structure type ID (cluster ID) of a material to provide structural information.

## Table of Contents

- [Project Overview](#project-overview)
- [Runtime Environment](#runtime-environment)
- [Datasets](#datasets)
- [Directory Structure](#directory-structure)
- [Workflow](#workflow)
- [Main Parameters](#main-parameters)
- [Inputs and Outputs](#inputs-and-outputs)
- [Examples](#examples)
- [Notes](#notes)

## Project Overview

The project mainly consists of five stages:

1. **Structural-parameter clustering**
   - Perform K-means clustering using structural parameters such as space group, Pearson symbol, Wyckoff sequence, `c/a` ratio, and `β` angle.
   - Select a better clustering scheme using the Silhouette Score and SSE elbow analysis.

2. **Frequency data processing**
   - Standardize phonon or Raman frequencies within the same structural cluster.
   - Truncate or pad frequency sequences.
   - Compute the average frequency within each cluster to obtain representative cluster frequencies.

3. **Affinity Propagation clustering**
   - Run Affinity Propagation clustering based on representative cluster frequencies.
   - Build training data linking frequency features to structure types.

4. **RCNet model training**
   - Train an RCNet neural network to learn the mapping from frequency features to structure type IDs.
   - Supports batch training, automatic GPU detection, and parallel training.

5. **PARCE integrated prediction**
   - Use trained models to predict new material frequency data.
   - Output predicted structure type IDs.

## Runtime Environment

A Conda environment is recommended.

| Item | Recommended Configuration |
| --- | --- |
| Conda environment name | `PARCE` |
| Python version | `3.8.20` |
| Main execution mode | Jupyter Notebook + Python scripts |

### Create and Activate the Environment

```bash
conda create -n PARCE python=3.8.20
conda activate PARCE
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib
pip install scikit-learn umap-learn kneed
pip install tqdm joblib jupyter tensorboard
pip install torch torchvision torchaudio
```

> If CUDA-enabled PyTorch is required, install the PyTorch package that matches the CUDA version on your server.

## Datasets

This repository currently includes two example datasets:

| File | Description |
| --- | --- |
| `dataset/phono_freq_dataset.csv` | Phonon frequency dataset |
| `dataset/raman_dataset.csv` | Raman-active frequency dataset |

Common fields are listed below:

| Field | Description |
| --- | --- |
| `id` / `material_id` | Material ID |
| `frequency` | Frequency sequence |
| `space_group_number` | Space group number |
| `pearson_symbol` | Pearson symbol |
| `wyckoff_sequence` | Wyckoff sequence |
| `c_a_ratio` | `c/a` axis ratio |
| `beta_angle` | `β` angle |
| `cif` | CIF structural information |

## Directory Structure

```text
.
├── README.md                  # English documentation
├── README_zh.md               # Chinese documentation
├── cluster.ipynb              # Main workflow for structural-parameter clustering and frequency clustering
├── cluster_umap_1.py          # Clustering script
├── parce.ipynb                # PARCE training-data processing and RCNet folder generation
├── parce_batch.py             # RCNet batch training script
├── predict.ipynb              # Prediction workflow
├── dataset/
│   ├── phono_freq_dataset.csv # Example phonon frequency data
│   └── raman_dataset.csv      # Example Raman frequency data
├── examples/
│   ├── phonon_freq/           # Phonon-frequency prediction example
│   └── raman/                 # Raman-frequency prediction example
└── template/
    ├── getdata.py             # RCNet data loading module
    ├── network.py             # RCNet network architecture
    ├── train.py               # RCNet training script template
    └── utils.py               # RCNet utility functions
```

The following directories may be generated during execution:

| Directory | Description |
| --- | --- |
| `all_cluster_results*/` | K-means clustering results for different feature combinations and cluster counts |
| `Kmeans_model/` | Saved K-means models |
| `clustering_results/` | Results of the selected clustering scheme |
| `standardized_frequencies/` | Standardized frequency data |
| `frequency_cuts/` | Truncated frequency data |
| `avg/` | Intra-cluster average frequency results |
| `training_data/` | RCNet training data |
| `data_for_conversion/` | Temporary directory for data conversion |
| `rcnet_training_umap/` | Batch-generated RCNet training directories |
| `model/` / `model_results/` | Trained model files and results |
| `batch_logs/` | Batch training logs |
| `test/` | Prediction test data and results |
| `id/` | ID mapping files |

## Workflow

### 1. Activate the Environment

```bash
conda activate PARCE
```

### 2. Prepare Data

Place data files in the `dataset/` directory and make sure they contain at least the following fields:

```text
frequency
space_group_number
pearson_symbol
wyckoff_sequence
c_a_ratio
beta_angle
```

If you use `cluster_umap_1.py` to run structural clustering in the background, first check and update the path configuration in the script:

```python
DATASET_PATH = "dataset/your_dataset.csv"
BASE_OUTPUT_DIR = "all_cluster_results"
```

The notebooks may also contain data path settings. Check and update them before running.

### 3. Structural-Parameter Clustering and Frequency Clustering

Open and run the following notebook in order:

```bash
jupyter notebook cluster.ipynb
```

This step performs:

- Structural-parameter feature encoding;
- K-means clustering;
- Clustering performance evaluation; `cluster_umap_1.py` can be used as an alternative;
- Frequency standardization and truncation;
- Affinity Propagation clustering;
- Initial training data generation.

### 4. Generate RCNet Training Directories

Open and run the following notebook in order:

```bash
jupyter notebook parce.ipynb
```

This step batch-generates RCNet training folders based on the `template/` directory.

### 5. Batch Train RCNet Models

First inspect the training tasks to be executed:

```bash
python parce_batch.py --dry-run
```

After confirming the tasks, start training:

```bash
python parce_batch.py --yes
```

Common parameter examples:

```bash
# Specify the training directory
python parce_batch.py --rcnet-training-dir ./rcnet_training_umap --yes

# Specify feature combinations and frequency lengths
python parce_batch.py --main-folders 124 125 --num-folders 6 12 18 --yes

# Force parallel training
python parce_batch.py --parallel --max-workers 4 --yes

# Use CPU only
python parce_batch.py --cpu-only --yes
```

### 6. Model Prediction

Open and run the following notebook in order:

```bash
jupyter notebook predict.ipynb
```

Prediction results are usually written to `test/` or to the result directory configured in the notebook.

## Main Parameters

### Structural Feature Encoding

| Code | Feature |
| --- | --- |
| `1` | `space_group_number` |
| `2` | `pearson_symbol` |
| `3` | `wyckoff_sequence` |
| `4` | `c_a_ratio` |
| `5` | `beta_angle` |

Examples:

- `124` means using space group, Pearson symbol, and `c/a` ratio;
- `125` means using space group, Pearson symbol, and `β` angle;
- `12345` means using all structural features.

### Clustering Parameters

| Parameter | Default / Example |
| --- | --- |
| K-means cluster-count range | `30-90`, step size `5` |
| Evaluation metrics | Silhouette Score, SSE elbow |
| Frequency truncation length | Subject to notebook or script configuration |
| Batch training directory | `./rcnet_training_umap` |

## Inputs and Outputs

### Inputs

- Material frequency sequences: phonon frequencies or Raman-active frequencies;
- Material structural parameters: space group, Pearson symbol, Wyckoff sequence, `c/a` ratio, and `β` angle.

### Outputs

- Structure type ID (cluster ID);
- K-means clustering model;
- RCNet training data;
- RCNet model files;
- Prediction result CSV files.

## Examples

| Example Directory | Description |
| --- | --- |
| `examples/phonon_freq/` | Structural classification based on phonon frequencies |
| `examples/raman/` | Structural classification based on Raman-active frequencies |

## Notes

1. Keep `README.md` as the standard English entry point. The Chinese version is `README_zh.md`; do not use `READ.ME`.
2. Before running notebooks, confirm that data paths and output paths point to the current project directory.
3. Some scripts or notebooks may contain historical absolute paths. Update them manually after moving the project.
4. Batch training can generate many models, logs, and intermediate files. Make sure there is enough disk space before running.
5. If the server has no GPU, `parce_batch.py` will automatically fall back to CPU mode. You can also force CPU mode with `--cpu-only`.
