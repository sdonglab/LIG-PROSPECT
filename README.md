# LIG-PROSPECT

**LIG**and-based **PRO**tein-agnostic **SPEC**tral **T**prediction predicts four excitation wavelength/oscillator-strength pairs from structural features.

| Key | Representation | Inference input |
| --- | --- | --- |
| `dd` | Distance descriptor | Text file with 4 ordered distances |
| `add` | Angle-distance descriptor | Text file with 4 distances and 3 angles (radians) |
| `pca_cc` | PCA Cartesian coordinates | XYZ file with the training atom order |
| `umap_ic` | UMAP internal coordinates | XYZ file with the training atom order |

Start with [`notebooks/01_predict_with_saved_model.ipynb`](notebooks/01_predict_with_saved_model.ipynb), which runs on the bundled example data.

## Install

LIG-PROSPECT requires Python 3.9 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the notebook interface, install the optional dependency and open Jupyter from this project directory.

```bash
python -m pip install -e ".[notebook]"
jupyter lab
```

## Predict spectra for your own structures

Pretrained models are in `saved_models/`. Put **only feature files** for the structures you want to predict in a new directory; excitation CSVs are not needed for inference. The descriptor type and feature ordering must match the selected model. The small `example_data/` directory is included only to verify installation and demonstrate the expected layouts.

The default configuration uses the DD single-run model:

```bash
ligprospect-predict --config configs/pred_config.yaml
```

Edit these fields in [`configs/pred_config.yaml`](configs/pred_config.yaml) for your data:

```yaml
prediction:
  workflow: single_run_random_split  # or hold_out_mutant
  descriptor: dd                     # dd, add, pca_cc, or umap_ic
data:
  features_dir: path/to/my_features
  file_glob: "*.txt"                 # use "*.xyz" for pca_cc/umap_ic
  dataset_name: my_experiment
```

Or specify all paths explicitly:

```bash
ligprospect-predict \
  --model saved_models/single_run_random_split/DD/ligprospect_dd_single_iter008_best_model.joblib \
  --metadata saved_models/single_run_random_split/DD/ligprospect_dd_single_iter008_best_metadata.yaml \
  --descriptor dd \
  --features-dir path/to/my_distance_files \
  --file-glob "*.txt" \
  --output results/my_predictions.csv
```

The output has one row per structure and eight values: `pred_1, osc_1, ..., pred_4, osc_4`.

Read [`docs/input-data.md`](docs/input-data.md) before preparing a new dataset. A different atom order, descriptor definition, or distance/angle unit is not compatible with a saved model.

## Train and evaluate a model

Training labels are CSV files containing excitation wavelengths and matching oscillator strengths. Structural feature files and label CSVs must share a filename stem. The private `input-database/` directory is intentionally excluded from this repository; update the paths in `configs/config_900nm.yaml` or `configs/config_1100nm.yaml` to point to your own training data before training:

```bash
ligprospect-train-single-split --config configs/config_900nm.yaml
ligprospect-evaluate-bootstrap --config configs/config_900nm.yaml
ligprospect-evaluate-heldout-mutant --config configs/config_900nm.yaml
ligprospect-compare-models --config configs/config_900nm.yaml
```

Use a new `run.out_root` for each analysis. Begin with `ligprospect-train-single-split` and confirm the reported sample count and feature shape before a larger bootstrap run.

### Which command should I use?

| Command | Purpose | Needs experimental labels? |
| --- | --- | --- |
| `ligprospect-predict` | Apply a saved model to new structures. | No |
| `ligprospect-train-single-split` | Train, evaluate, and save one model using one reproducible split. | Yes |
| `ligprospect-evaluate-bootstrap` | Repeat train/test splits to estimate performance stability. | Yes |
| `ligprospect-evaluate-heldout-mutant` | Train on other mutants and evaluate on one held-out mutant. | Yes |
| `ligprospect-compare-models` | Compare regression-model families on a fixed split. | Yes |


Each training configuration is grouped by purpose: `run` selects the output directory and descriptor, `data` holds input locations and wavelength filtering, `split` controls automatic or fixed split sizes, and each workflow has its own short block (`bootstrap`, `single`, `mutant_prediction`, or `baselines`).

## Project layout

```text
configs/       YAML settings for training and inference
docs/          Data-format reference
example_data/  Small public inputs and labels for installation checks and the tutorial
notebooks/     Step-by-step Jupyter tutorial
saved_models/  Bundled trained models and metadata
src/           Package source code
outputs/       Generated results (not source data)
```

`outputs/`, `new_data/`, and `input-database/` are intentionally ignored by Git. Saved models and `example_data/` are included so users can run inference immediately.

## Citation

Bhumika Jayee, Sunny Lee, Sijia S. Dong. *Sequence-Transferrable Machine Learning Prediction of Flavin-Dependent Photoenzyme Spectral Properties.* ChemRxiv (2026). https://doi.org/10.26434/chemrxiv.15002532/v2

## Contact

Sijia Dong — s.dong (AT) northeastern.edu
Bhumika Jayee — bhumikajayee03 (AT) gmail.com

© 2025 Northeastern University. Any commercial use requires written permission from the copyright holder.
