# Preparing input data

## Inference: predicting spectra for new structures

An inference directory contains one feature file per structure. Filenames become sample identifiers and must be unique. No experimental spectrum or label CSV is needed for inference.

| Model descriptor | Files to provide | Per-file requirements |
| --- | --- | --- |
| `dd` | Text (`.txt`) | Exactly four whitespace-separated distance values in training order |
| `add` | Text (`.txt`) | Four distances followed by three angles in radians, in training order |
| `pca_cc` | XYZ (`.xyz`) | Standard XYZ; same atom count and atom order as training structures |
| `umap_ic` | XYZ (`.xyz`) | Standard XYZ; same atom count and atom order as training structures |

Descriptor definitions are scientific choices, not just file formats. Reuse the same atom selections, ordering, units, and angle convention used during training. For coordinate-based models, changing atom order changes the feature vector and invalidates predictions.

### Minimal DD file

`my_features/example_001.txt`

```text
5.25800347 4.75548604 4.92800626 10.21508074
```

## Training data

Training needs one feature file and a matching excitation CSV per sample. Stems must match exactly:

```text
features/structure_001.txt
excitations/structure_001.csv
```

Each label CSV must contain the four requested excitation wavelengths and their matching oscillator strengths. Keep a consistent column arrangement across all files; the loader uses a `wavelength` column when present, otherwise treats the first column as wavelength.

Before a large training run, create a dedicated output directory and run `ligprospect-train-single-split`. Confirm its sample count and feature shape first.

## Generalization note

LIG-PROSPECT can pad shorter vectors when a saved model expects a larger vector. That supports the original variable-length representation; it does not make an incompatible descriptor scientifically valid. Treat a changed atom count, feature-length error, or descriptor definition as a data-preparation issue.
