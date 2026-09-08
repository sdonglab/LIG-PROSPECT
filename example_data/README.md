# Public example data

This small dataset is included solely to verify installation and illustrate the
required file layouts. It contains 15 matched samples: five WT, five A36E, and
five K283G conformations. Every sample has a CSV label file, DD feature file,
ADD feature file, and XYZ structure. Their maximum excitation wavelengths span
approximately 428 to 811 nm.

It is not the full training dataset and should not be used to estimate model
performance or retrain a production model.

- `desc-4/` contains four-distance (`dd`) feature files.
- `desc-7/` contains seven-value angle-distance (`add`) feature files.
- `xyz/` contains XYZ structures for `pca_cc` and `umap_ic` features.
- `csv/` contains the matching excitation labels.

Run the bundled inference check with:

```bash
ligprospect-predict --config configs/predict_config.yaml
```

The command reads `desc-4/` and does not require the CSV labels. The labels are
provided only to demonstrate the training-data filename convention.
