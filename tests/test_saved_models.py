"""Smoke tests for every public saved-model representation."""

from __future__ import annotations

from pathlib import Path
import unittest

import joblib

from lig_prospect.predict_cli import infer_expected_dim, load_features_only, predict_with_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_CASES = {
    "dd": ("DD", "desc-4", "*.txt"),
    "add": ("ADD", "desc-7", "*.txt"),
    "pca_cc": ("PCA-CC", "xyz", "*.xyz"),
    "umap_ic": ("UMAP-IC", "xyz", "*.xyz"),
}


class SavedModelTests(unittest.TestCase):
    def test_each_bundled_model_predicts_public_examples(self) -> None:
        for descriptor, (model_dir, feature_dir, pattern) in MODEL_CASES.items():
            with self.subTest(descriptor=descriptor):
                model_path = (
                    PROJECT_ROOT
                    / "saved_models"
                    / "single_run_random_split"
                    / model_dir
                    / f"ligprospect_{descriptor}_single_iter008_best_model.joblib"
                )
                bundle = joblib.load(model_path)
                features, _ = load_features_only(
                    descriptor=descriptor,
                    features_dir=PROJECT_ROOT / "example_data" / feature_dir,
                    file_glob=pattern,
                    expected_dim=infer_expected_dim(bundle),
                )
                predictions = predict_with_bundle(bundle, features)
                self.assertEqual(predictions.shape, (15, 8))
