"""Regression checks for compressed fixtures and repeatable provenance output."""
from contextlib import redirect_stdout
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReproductionIntegrityTests(unittest.TestCase):
    def test_alignment_fixture_has_restoration_entry(self):
        entries = json.loads((ROOT / "COMPRESSED_FILES.json").read_text())
        target = "alignment/baseline/results/104_annual_scenario_cube_v1_2.csv"
        entry, = [item for item in entries if item["target"] == target]
        checksum = hashlib.sha256()
        with gzip.open(ROOT / entry["stored"], "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        self.assertEqual(checksum.hexdigest(), entry["uncompressed_sha256"])

    def test_verification_restores_fixture_and_repeats(self):
        verify = module("test_package_verification", ROOT / "verify_package.py")
        with TemporaryDirectory() as folder:
            root = Path(folder)
            content = b"planning_region,year\nMISO,2035\n"
            stored = root / "fixture.csv.gz"
            stored.write_bytes(gzip.compress(content))
            (root / "PACKAGE_MANIFEST.json").write_text(json.dumps({"files": [
                {"path": stored.name, "sha256": digest(stored)}]}))
            (root / "COMPRESSED_FILES.json").write_text(json.dumps([{
                "stored": stored.name, "target": "restored/fixture.csv",
                "uncompressed_sha256": hashlib.sha256(content).hexdigest()}]))
            verify.ROOT = root
            self.assertTrue(verify.verify()["passed"])
            self.assertTrue(verify.verify()["passed"])
            self.assertEqual((root / "restored/fixture.csv").read_bytes(), content)
            stored.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Package bytes differ"):
                verify.verify()

    def test_mapping_build_preserves_frozen_sources_on_repeated_runs(self):
        mapping = module("test_fixed_energy_mapping", ROOT / "numerical/scripts/build_fixed_energy_mapping.py")
        source = ROOT / "numerical/sources/lbnl_2025_update"
        with TemporaryDirectory() as folder:
            root = Path(folder)
            copied = root / "sources"
            copied.mkdir()
            (root / "results").mkdir()
            for path in source.glob("cec*"):
                shutil.copy2(path, copied / path.name)
            for name in ["iso_coincidence_peak_window_3h.csv",
                         "peak_mapping_factors_fixed_energy_20261004.csv",
                         "fixed_energy_normalized_hourly_shapes.csv",
                         "fixed_energy_mapping_protocol.json"]:
                shutil.copy2(source / name, copied / name)
            before = {p.name: digest(p) for p in copied.iterdir()}
            mapping.ROOT, mapping.SOURCE = root, copied
            with redirect_stdout(io.StringIO()):
                mapping.build()
                first = digest(root / "results/cec_source_hashes.csv")
                mapping.build()
            self.assertEqual(before, {p.name: digest(p) for p in copied.iterdir()})
            self.assertEqual(first, digest(root / "results/cec_source_hashes.csv"))
            self.assertNotIn("cec_provenance_hashes.csv",
                             (root / "results/cec_source_hashes.csv").read_text())


if __name__ == "__main__":
    unittest.main()
