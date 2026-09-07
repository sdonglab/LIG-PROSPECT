"""Release checks for the small public example dataset."""

from __future__ import annotations

import csv
from pathlib import Path
import unittest

from lig_prospect.io import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = PROJECT_ROOT / "example_data"


def stems(directory: Path) -> set[str]:
    return {path.stem for path in directory.iterdir() if path.is_file()}


class ExampleDataTests(unittest.TestCase):
    def test_all_representations_have_the_same_15_samples(self) -> None:
        csv_stems = stems(EXAMPLE_ROOT / "csv")
        self.assertEqual(len(csv_stems), 15)
        self.assertEqual(csv_stems, stems(EXAMPLE_ROOT / "desc-4"))
        self.assertEqual(csv_stems, stems(EXAMPLE_ROOT / "desc-7"))
        self.assertEqual(csv_stems, stems(EXAMPLE_ROOT / "xyz"))

    def test_wavelengths_do_not_exceed_public_example_limit(self) -> None:
        for csv_path in (EXAMPLE_ROOT / "csv").glob("*.csv"):
            with csv_path.open(encoding="utf-8", newline="") as handle:
                wavelengths = [float(row["wavelength"]) for row in csv.DictReader(handle)]
            self.assertLessEqual(max(wavelengths), 1500.0, csv_path.name)

    def test_all_descriptor_inputs_load_with_the_public_labels(self) -> None:
        descriptor_inputs = {
            "dd": ("desc-4", "*.txt"),
            "add": ("desc-7", "*.txt"),
            "pca_cc": ("xyz", "*.xyz"),
            "umap_ic": ("xyz", "*.xyz"),
        }
        for descriptor, (directory, pattern) in descriptor_inputs.items():
            with self.subTest(descriptor=descriptor):
                dataset = load_dataset(
                    descriptor=descriptor,
                    features_dir=EXAMPLE_ROOT / directory,
                    excitations_dir=EXAMPLE_ROOT / "csv",
                    file_glob=pattern,
                    wavelength_max_nm=1500.0,
                )
                self.assertEqual(dataset.X.shape[0], 15)
                self.assertEqual(dataset.y.shape, (15, 8))
