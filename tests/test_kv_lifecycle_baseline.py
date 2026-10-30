import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline_gate", ROOT / "scripts/check_kv_lifecycle_baseline.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class BaselineGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = json.loads((ROOT / "docs/contracts/kv-lifecycle-baseline-v1.json").read_text(encoding="utf-8"))

    def read(self, manifest):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return gate.read_manifest(path)

    def test_frozen_manifest_and_all_profiles(self):
        manifest = self.read(self.manifest)
        for profile in manifest["profiles"]:
            gate.validate_profile(manifest, profile["id"])

    def test_no_empty_or_duplicate_test_success(self):
        for tests in ([], ["memory_contract_test", "memory_contract_test"]):
            value = copy.deepcopy(self.manifest)
            value["host_tests"] = tests
            with self.assertRaises(gate.BaselineError):
                self.read(value)

    def test_contract_requires_registered_coverage(self):
        value = copy.deepcopy(self.manifest)
        value["contracts"][0]["tests"] = ["not_a_test"]
        with self.assertRaises(gate.BaselineError):
            self.read(value)

    def test_unpinned_model_is_not_accepted(self):
        value = copy.deepcopy(self.manifest)
        value["model"]["sha256"] = None
        with self.assertRaises(gate.BaselineError):
            self.read(value)

    def test_out_of_scope_profiles_cannot_be_promoted(self):
        for key, value in (("max_concurrency", 2), ("historical_tier", "ssd"),
                           ("admission", "strict_request"), ("spec", "mtp8"),
                           ("kv_dtype", "f16"), ("modality", "image")):
            manifest = copy.deepcopy(self.manifest)
            manifest["profiles"][0][key] = value
            with self.subTest(key=key), self.assertRaises(gate.BaselineError):
                gate.validate_profile(manifest, manifest["profiles"][0]["id"])

    def test_native_shadow_is_not_invented(self):
        value = copy.deepcopy(self.manifest)
        value["profiles"][0]["policy_modes"].append("shadow")
        with self.assertRaises(gate.BaselineError):
            gate.validate_profile(value, value["profiles"][0]["id"])

    def test_required_auxiliary_budget_is_explicit_positive_and_bounded(self):
        for auxiliary in (None, 0, -1, True, 1.5, 1 << 44):
            value = copy.deepcopy(self.manifest)
            value["auxiliary_budget_required"] = True
            if auxiliary is not None: value["profiles"][0]["gc_aux_mib"] = auxiliary
            with self.subTest(auxiliary=auxiliary), self.assertRaises(gate.BaselineError):
                gate.validate_profile(value, value["profiles"][0]["id"])
        value["profiles"][0]["gc_aux_mib"] = 1024
        gate.validate_profile(value, value["profiles"][0]["id"])

    def test_junit_must_execute_exactly_required_tests(self):
        path = self.root / "ctest.xml"
        for body in ("", '<testcase name="other"/>',
                     '<testcase name="one"/><testcase name="one"/>'):
            path.write_text('<testsuite>' + body + '</testsuite>', encoding="utf-8")
            with self.subTest(body=body), self.assertRaises(gate.BaselineError):
                gate.verify_ctest_report(path, ["one"])

    def test_skipped_failure_error_and_notrun_do_not_pass(self):
        path = self.root / "ctest.xml"
        for body in ('<testcase name="one"><skipped/></testcase>',
                     '<testcase name="one"><failure/></testcase>',
                     '<testcase name="one"><error/></testcase>',
                     '<testcase name="one" status="notrun"/>'):
            path.write_text('<testsuite>' + body + '</testsuite>', encoding="utf-8")
            with self.subTest(body=body), self.assertRaises(gate.BaselineError):
                gate.verify_ctest_report(path, ["one"])

    def test_completed_junit_and_checkout_line_endings(self):
        path = self.root / "ctest.xml"
        path.write_text('<testsuite><testcase name="one" status="run" time="0.2"/></testsuite>', encoding="utf-8")
        self.assertEqual(gate.verify_ctest_report(path, ["one"]), [{"name": "one", "seconds": .2}])
        self.assertEqual(gate.digest(b"source\r\n"), gate.digest(b"source\n"))


if __name__ == "__main__":
    unittest.main()
