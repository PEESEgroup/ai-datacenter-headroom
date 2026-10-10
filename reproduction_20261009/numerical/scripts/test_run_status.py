"""A failed or not-yet-run attempt must not look like the preceding success."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import run_all
import run_frozen_outputs
from run_status import RunStatus, mark_not_run


class RunStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "verification.json"
        self.manifest = self.root / "manifest.txt"
        self.manifest.write_text("frozen inputs\n")

    def seed(self):
        self.path.write_text(json.dumps({"passed": True, "status": "passed", "attempt_id": "prior-success"}))

    def read(self):
        return json.loads(self.path.read_text())

    def test_start_invalidates_old_pass(self):
        self.seed()
        with RunStatus(self.path, {"inputs": self.manifest}) as attempt:
            self.assertFalse(self.read()["passed"])
            self.assertEqual(self.read()["status"], "running")
            self.assertEqual(self.read()["last_success"]["attempt_id"], "prior-success")
            attempt.succeed({"tables_compared": 40})
        self.assertTrue(self.read()["passed"])

    def test_failure_records_attempt_and_preserves_history(self):
        self.seed()
        with self.assertRaisesRegex(ValueError, "changed table"):
            with RunStatus(self.path, {"inputs": self.manifest}):
                raise ValueError("changed table")
        record = self.read()
        self.assertFalse(record["passed"])
        self.assertEqual(record["status"], "failed")
        self.assertNotEqual(record["attempt_id"], "prior-success")
        self.assertEqual(record["last_success"]["attempt_id"], "prior-success")
        self.assertIn("finished_at_UTC", record)
        self.assertEqual(record["manifest_sha256"]["inputs"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())

    def test_repeated_failure_retains_only_last_success(self):
        self.seed()
        for _ in range(2):
            with self.assertRaises(ValueError):
                with RunStatus(self.path, {}):
                    raise ValueError("failure")
        self.assertEqual(self.read()["last_success"]["attempt_id"], "prior-success")
        self.assertNotIn("last_success", self.read()["last_success"])

    def test_not_run_never_inherits_a_current_pass(self):
        self.seed()
        mark_not_run(self.path, "parent-run")
        self.assertFalse(self.read()["passed"])
        self.assertEqual(self.read()["status"], "not_run")
        self.assertEqual(self.read()["parent_attempt_id"], "parent-run")

    def test_success_requires_explicit_completion(self):
        with self.assertRaisesRegex(RuntimeError, "without declaring success"):
            with RunStatus(self.path, {}):
                pass
        self.assertEqual(self.read()["status"], "failed")

    def test_missing_manifest_is_not_a_claimed_verified_hash(self):
        with self.assertRaises(ValueError):
            with RunStatus(self.path, {"missing": self.root / "absent.txt"}):
                raise ValueError("missing input")
        self.assertIsNone(self.read()["manifest_sha256"]["missing"])

    def test_comparator_failure_updates_stale_pass(self):
        output = self.root / "frozen_output_verification.json"
        output.write_text(json.dumps({"passed": True, "attempt_id": "old-comparison"}))
        with patch.object(run_frozen_outputs, "ROOT", self.root), \
             patch.object(run_frozen_outputs, "compare_outputs", side_effect=ValueError("text changed")):
            with self.assertRaisesRegex(ValueError, "text changed"):
                run_frozen_outputs.main()
        record = json.loads(output.read_text())
        self.assertFalse(record["passed"])
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["last_success"]["attempt_id"], "old-comparison")

    def test_full_hash_failure_invalidates_both_records(self):
        for name in ("reproduction_verification.json", "frozen_output_verification.json"):
            (self.root / name).write_text(json.dumps({"passed": True, "attempt_id": "old-pass"}))
        (self.root / "input.txt").write_text("changed")
        (self.root / "SHA256SUMS.txt").write_text("0" * 64 + "  input.txt\n")
        with patch.object(run_all, "ROOT", self.root), patch.object(run_all.subprocess, "run") as child:
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_all.main()
            child.assert_not_called()
        parent = json.loads((self.root / "reproduction_verification.json").read_text())
        comparison = json.loads((self.root / "frozen_output_verification.json").read_text())
        self.assertEqual(parent["status"], "failed")
        self.assertFalse(parent["passed"])
        self.assertEqual(comparison["status"], "not_run")
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["parent_attempt_id"], parent["attempt_id"])


if __name__ == "__main__":
    unittest.main()
