#!/usr/bin/env python3
"""Verify a frozen lifecycle baseline and run its model-free contract tests.

This is a development qualification gate, not a native server feature gate or
GPU/Agent qualification. It never installs dependencies or prepares a backend.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


class BaselineError(RuntimeError):
    pass


def digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise BaselineError(f"Command failed ({result.returncode}): {command}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def read_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1:
        raise BaselineError("Unsupported manifest schema")
    if manifest.get("primary", {}).get("backend") != "ninfer":
        raise BaselineError("This baseline requires the NInfer primary backend")
    model = manifest.get("model", {})
    model_hash = model.get("sha256")
    if not isinstance(model_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", model_hash) or model.get("bytes", 0) <= 0:
        raise BaselineError("An exact model digest and size are required")
    if not manifest.get("source_files") or not manifest.get("host_tests"):
        raise BaselineError("Source pins and executable host tests must not be empty")
    tests = manifest["host_tests"]
    if len(tests) != len(set(tests)):
        raise BaselineError("Duplicate host tests")
    ids = [contract["id"] for contract in manifest["contracts"]]
    if len(ids) != len(set(ids)):
        raise BaselineError("Duplicate contract IDs")
    for contract in manifest["contracts"]:
        if contract["level"] == "host_test":
            if not contract.get("tests") or not set(contract["tests"]) <= set(tests):
                raise BaselineError(f"Unmapped executable contract: {contract['id']}")
    return manifest


def validate_profile(manifest, name):
    profiles = {profile["id"]: profile for profile in manifest["profiles"]}
    if name not in profiles:
        raise BaselineError(f"Unsupported profile {name!r}; choose one of {sorted(profiles)}")
    profile = profiles[name]
    exact = {"backend": "ninfer", "gpu_count": 1, "max_concurrency": 1,
             "modality": "text", "historical_tier": "ram", "page_tokens": 64,
             "kv_dtype": "rk4v4", "admission": "bounded_prefill", "query_scorer": "meanq-cosine"}
    for key, expected in exact.items():
        if profile.get(key) != expected:
            raise BaselineError(f"Profile {name}: {key} must be {expected!r}")
    if profile.get("spec") not in ("none", "mtp4"):
        raise BaselineError("Only ordinary decode and MTP4 are in scope")
    if profile.get("policy_modes") != ["off", "lru", "redundancy"]:
        raise BaselineError("Native NInfer exposes off/LRU/redundancy, not shadow")
    for key in ("active_tokens", "reserve_tokens", "host_mib", "context_tokens", "output_tokens"):
        if not isinstance(profile.get(key), int) or profile[key] <= 0:
            raise BaselineError(f"Profile {name}: {key} must be positive")
    # NInfer selected history and generation reserve are separate budgets;
    # the tested 8K/16K Agent profile legitimately reserves more than it selects.
    if profile["reserve_tokens"] + profile["active_tokens"] > profile["context_tokens"] or profile["output_tokens"] > profile["context_tokens"]:
        raise BaselineError("Inconsistent profile budgets")
    return profile


def verify_source(root, manifest):
    baseline = manifest["primary"]["source_commit"]
    run(["git", "merge-base", "--is-ancestor", baseline, "HEAD"], root)
    for relative, expected in manifest["source_files"].items():
        path = root / relative
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise BaselineError(f"Frozen source differs or is missing: {relative}")
    versions = json.loads((root / "backends/versions.json").read_text(encoding="utf-8-sig"))["ninfer"]
    native = manifest["primary"]["native"]
    for key in ("base_commit", "patched_tree"):
        if versions[key] != native[key]:
            raise BaselineError(f"Native {key} differs")
    if len(versions["patches"]) != native["patch_count"]:
        raise BaselineError("Native patch count differs")
    for patch in versions["patches"]:
        if digest((root / patch["path"]).read_bytes()) != patch["sha256"]:
            raise BaselineError(f"Native patch hash differs: {patch['path']}")
    return {"commit": run(["git", "rev-parse", "HEAD"], root).strip(),
            "pinned_source_files": len(manifest["source_files"]),
            "native_patches": len(versions["patches"]),
            "native_tree_replayed": False}


def verify_ctest_report(path, expected):
    tree = ET.parse(path).getroot()
    cases = list(tree.iter("testcase"))
    names = [case.get("name") for case in cases]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise BaselineError("CTest did not execute exactly the required contracts")
    for case in cases:
        if case.get("status") not in (None, "run") or any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
            raise BaselineError(f"Nonpassing or skipped contract: {case.get('name')}")
    return [{"name": case.get("name"), "seconds": float(case.get("time", "0"))} for case in cases]


def run_contracts(root, build, manifest, cmake, ctest, output):
    cache_path = build / "CMakeCache.txt"
    if not cache_path.exists():
        raise BaselineError("Configure a dedicated Debug host-only build first")
    cache = {}
    for line in cache_path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and ":" in line.split("=", 1)[0]:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    if Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != root.resolve():
        raise BaselineError("Build directory belongs to a different source checkout")
    if cache.get("CMAKE_BUILD_TYPE") != "Debug":
        raise BaselineError("Use Debug so assertion-based contracts are active")
    for flag in ("KVMEM_BUILD_LLAMA", "KVMEM_BUILD_SERVER_TESTS", "KVMEM_CUDA_TESTS"):
        if cache.get(flag) != "OFF":
            raise BaselineError(f"Host baseline requires {flag}=OFF")
    tests = manifest["host_tests"]
    registered = json.loads(run([ctest, "--test-dir", str(build), "--show-only=json-v1"], root))
    if not set(tests) <= {item["name"] for item in registered["tests"]}:
        raise BaselineError("Required CTest targets are not registered")
    build_log = run([cmake, "--build", str(build), "--parallel", "4", "--target", *tests], root)
    (output / "build.log").write_text(build_log, encoding="utf-8")
    regex = "^(" + "|".join(re.escape(name) for name in tests) + ")$"
    junit = output / "host-contracts.xml"
    log = run([ctest, "--test-dir", str(build), "--no-tests=error", "--output-on-failure",
               "--timeout", "120", "--output-junit", str(junit), "-R", regex], root)
    (output / "ctest.log").write_text(log, encoding="utf-8")
    results = verify_ctest_report(junit, tests)
    binaries = {}
    for name in tests:
        binary = build / "bin" / (name + (".exe" if sys.platform == "win32" else ""))
        binaries[name] = hashlib.sha256(binary.read_bytes()).hexdigest()
    return {"results": results, "binary_sha256": binaries,
            "compiler": cache.get("CMAKE_CXX_COMPILER"), "build_type": "Debug"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--profile", default="native-token-micro")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--model", type=Path, help="Optional local model; hash full bytes when supplied")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ctest", default="ctest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "gpu_or_agent_tests_run": False,
              "production_qualified": False, "local_model_verified": False}
    try:
        path = args.manifest or root / "docs/contracts/kv-lifecycle-baseline-v1.json"
        manifest = read_manifest(path)
        report["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["profile"] = validate_profile(manifest, args.profile)
        # Validate every declared profile, including those not selected today.
        for profile in manifest["profiles"]:
            validate_profile(manifest, profile["id"])
        report["source"] = verify_source(root, manifest)
        if args.model:
            before = args.model.stat()
            h = hashlib.sha256()
            with args.model.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            after = args.model.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise BaselineError("Model changed while hashing")
            if before.st_size != manifest["model"]["bytes"] or h.hexdigest() != manifest["model"]["sha256"]:
                raise BaselineError("Model bytes differ from frozen baseline")
            report["local_model_verified"] = True
        if args.build_dir:
            report["host_contracts"] = run_contracts(root, args.build_dir.resolve(), manifest,
                                                      args.cmake, args.ctest, args.output)
            # Detect changes made while the build/tests were running.
            verify_source(root, manifest)
            report["status"] = "host_contracts_passed"
        else:
            report["status"] = "source_verified_only"
        code = 0
    except (BaselineError, OSError, ValueError, TypeError, KeyError, ET.ParseError) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        code = 1
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "production_qualified", "gpu_or_agent_tests_run")}, ensure_ascii=False))
    if code:
        print(report["error"], file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
