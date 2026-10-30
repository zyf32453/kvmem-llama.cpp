#!/usr/bin/env python3
"""Run fixed Terminal-Bench tasks against matched NInfer GC configurations."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

TASKS = ("cobol-modernization", "constraints-scheduling")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(dataset, destination):
    destination.mkdir(exist_ok=False)
    suffix = b"""
# Isolated verifier dependency repair. Task inputs, instruction and tests are unchanged.
COPY --from=python:3.13-slim-bookworm /usr/local /usr/local
RUN apt-get update && apt-get install -y curl \\
 && python3.13 -m pip install --no-cache-dir uv==0.9.5 \\
 && mkdir -p /root/.local/bin \\
 && printf 'export PATH=/usr/local/bin:$PATH\\n' > /root/.local/bin/env \\
 && UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --version
"""
    manifest = {}
    for name in TASKS:
        source = dataset / "tasks" / name
        target = destination / name
        shutil.copytree(source, target)
        dockerfile = target / "environment/Dockerfile"
        dockerfile.write_bytes(dockerfile.read_bytes() + suffix)
        checked = {}
        for original in sorted(source.rglob("*")):
            if original.is_file() and original.relative_to(source).as_posix() != "environment/Dockerfile":
                relative = original.relative_to(source)
                expected = hashlib.sha256(original.read_bytes()).hexdigest()
                assert hashlib.sha256((target / relative).read_bytes()).hexdigest() == expected
                checked[relative.as_posix()] = expected
        manifest[name] = dict(unchanged=checked, environment_dockerfile_sha256=
                             hashlib.sha256(dockerfile.read_bytes()).hexdigest())
    save(destination / "source-audit.json", manifest)


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def server(args, policy, name):
    active, host = (98304, 4760) if args.profile == "teacher" else (8192, 1190)
    cmd = ["docker", "run", "-d", "--name", name, "--gpus", "all",
           "-p", f"127.0.0.1:{args.port}:{args.port}", "-v", f"{args.root}:/work",
           "-v", f"{args.model_dir}:/models:ro",
           "-v", "/usr/local/cuda-13.2:/usr/local/cuda-13.2:ro", "-e", "KVMEM_GC_AUDIT=1",
           "kvmem-ninfer-gc-builder:20261029", "/work/build-native/apps/ninfer-serve",
           "/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer",
           "--model-id", "qwen38-ninfer-gc-mtp", "--host", "0.0.0.0", "--port", str(args.port),
           "--kvmem", "--kvmem-budget", str(active), "--kvmem-gen-reserve", "16384",
           "--kvmem-host-mib", str(host), "--kvmem-sink-tokens", "1024", "--kvmem-recent-tokens", "64",
           "--max-context", "262144", "--kvmem-query-scorer", "meanq-cosine", "--kvmem-score-agg", "max",
           "--no-kvmem-lrfu", "--kvmem-graph-alpha", "0", "--kvmem-query-max-tokens", "8192",
           "--kvmem-query-replay-max-tokens", "8192", "--kv-dtype", "rk4v4", "--max-concurrency", "1",
           "--prefill-chunk", "512", "--default-max-tokens", "16384", "--default-thinking-budget", "4096",
           "--spec", "mtp", "--draft-tokens", "4", "--lm-head-draft", "--device-profile", "off"]
    if policy != "off":
        cmd += [f"--kvmem-gc-{policy}"]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--task-root", type=Path, required=True)
    p.add_argument("--harbor", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--profile", choices=("teacher", "pressure"), default="teacher")
    p.add_argument("--mode", choices=("prepare", "oracle", "agent"), required=True)
    p.add_argument("--policies", nargs="+", choices=("off", "lru", "redundancy"),
                   default=["off", "lru", "redundancy"])
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--port", type=int, default=18098)
    args = p.parse_args()
    if args.mode == "prepare":
        if not args.dataset:
            p.error("prepare requires --dataset")
        prepare(args.dataset, args.task_root)
        return
    if not args.out or not args.harbor or (args.mode == "agent" and not args.model_dir):
        p.error("run requires --out and --harbor; agent also requires --model-dir")
    args.out.mkdir(exist_ok=False)
    save(args.out / "plan.json", dict(mode=args.mode, profile=args.profile, tasks=args.tasks,
         policies=args.policies, sampling_temperature=.6, seed=1000, output_tokens=16384,
         thinking_tokens=4096, gen_reserve=16384, active_tokens=98304 if args.profile == "teacher" else 8192,
         historical_tokens=262144 if args.profile == "teacher" else 65536,
         sink_tokens=1024, recent_tokens=64, block_tokens=64, mtp_drafts=4,
         agent_timeout_seconds=7200, verifier_timeout="official unchanged", harness_summarization=False,
         fresh_model_and_host_per_trial=True,
         task_sources=json.loads((args.task_root / "source-audit.json").read_text())))
    results = []
    for task in args.tasks:
        policies = ["oracle"] if args.mode == "oracle" else args.policies
        for policy in policies:
            case = args.out / f"{task}-{policy}"
            case.mkdir()
            name = f"ninfer-qual-{policy}-{task}-{int(time.time())}"
            result = dict(task=task, policy=policy, profile=args.profile, started=time.time())
            save(args.out / "run-status.json", dict(status="running", current=result, completed=results))
            agent = dict(name="oracle") if policy == "oracle" else dict(
                name="terminus-2", model_name="openai/qwen38-ninfer-gc-mtp", override_timeout_sec=7200,
                kwargs=dict(api_base=f"http://127.0.0.1:{args.port}/v1", enable_summarize=False,
                    proactive_summarization_threshold=0, interleaved_thinking=True, reasoning_effort="low",
                    store_all_messages=True, model_info=dict(max_input_tokens=245760, max_output_tokens=16384,
                        input_cost_per_token=0, output_cost_per_token=0),
                    llm_call_kwargs=dict(max_tokens=16384, temperature=.6, seed=1000)))
            config = dict(job_name="job", jobs_dir=str(case / "jobs"), n_concurrent_trials=1,
                          n_attempts=1, environment=dict(type="docker", force_build=True),
                          agents=[agent], tasks=[dict(path=str(args.task_root / task))])
            save(case / "config.json", config)
            launched = False
            try:
                if policy != "oracle":
                    cmd = server(args, policy, name)
                    save(case / "server-command.json", cmd)
                    command(cmd)
                    launched = True
                    deadline = time.monotonic() + 90
                    while True:
                        try:
                            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2).close()
                            break
                        except (OSError, TimeoutError):
                            if command(["docker", "inspect", "--format", "{{.State.Running}}", name]) != "true":
                                raise RuntimeError("model startup failed")
                            if time.monotonic() > deadline:
                                raise TimeoutError("model readiness")
                            time.sleep(.5)
                env = dict(os.environ, OPENAI_API_KEY="kvmem-local", NO_PROXY="localhost,127.0.0.1",
                           DOCKER_BUILDKIT="0", COMPOSE_DOCKER_CLI_BUILD="0")
                with (case / "harbor.log").open("w") as log:
                    done = subprocess.run([str(args.harbor), "run", "-c", str(case / "config.json")],
                                          env=env, stdout=log, stderr=subprocess.STDOUT)
                result["harbor_exit_code"] = done.returncode
                trials = [path for path in (case / "jobs/job").glob("*/result.json")]
                result["trials"] = []
                for path in trials:
                    trial = json.loads(path.read_text())
                    rewards = (trial.get("verifier_result") or {}).get("rewards")
                    result["trials"].append(dict(result_path=str(path), rewards=rewards,
                        exception=trial.get("exception_info"), usage=(trial.get("agent_result") or {}).get("model_usage")))
            except Exception as error:
                result["controller_exception"] = repr(error)
            finally:
                if launched:
                    logs = subprocess.run(["docker", "logs", name], text=True, capture_output=True)
                    text = logs.stdout + logs.stderr
                    (case / "server.log").write_text(text)
                    result["gc_summaries"] = [dict(re.findall(r"(\w+)=([^ ]+)", row)) for row in
                                              re.findall(r"KVMEM_NATIVE_GC_SUMMARY ([^\n]+)", text)]
                    result["proposal_reasons"] = {}
                    for reason in re.findall(r"KVMEM_GC_AUDIT event=proposed[^\n]*? reason=(\w+)", text):
                        result["proposal_reasons"][reason] = result["proposal_reasons"].get(reason, 0) + 1
                    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
                result["finished"] = time.time()
                save(case / "result.json", result)
                results.append(result)
                save(args.out / "results.json", results)
                print(json.dumps(result), flush=True)
    save(args.out / "run-status.json", dict(status="finished", completed=results))


if __name__ == "__main__":
    main()
