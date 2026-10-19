"""Real-model HTTP lifecycle regression for finite historical KV admission.

Starts only a private loopback server; logs, response evidence and the exact
command are saved under --output. SSE delivery still reserves the whole finite
request up front. It does not test incremental/streaming RAM reservations.
"""

import argparse
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["off", "shadow", "delete"], default="delete")
    parser.add_argument("--native-template", action="store_true", help="smoke-test the model's normal chat template")
    parser.add_argument("--pinned-gb", type=float, default=0, help="reserve independent page-locked historical copies")
    args = parser.parse_args()
    if not math.isfinite(args.pinned_gb) or not 0 <= args.pinned_gb <= 1048576:
        parser.error("--pinned-gb must be finite and between 0 and 1048576")
    server, model = args.server.resolve(strict=True), args.model.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    template = out / "plain.jinja"
    template.write_text("{{ messages[0]['content'] }}", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_") and k != "LLAMA_API_KEY"}
    checks, replies = [], []

    def check(label, ok):
        checks.append({"name": label, "pass": bool(ok)})
        print(("PASS " if ok else "FAIL ") + label, flush=True)
        if not ok:
            raise AssertionError(label)

    # Invalid combinations must fail before trying to read model weights.
    for extra in (["--kvmem-method", "retrieval"], ["--parallel", "2"],
                  ["--kvmem-cpu-gb", "0.1"], ["--flash-attn", "off"]):
        result = subprocess.run([str(server), "-m", "/nonexistent-historical-model",
            "--kvmem-method", "recency", "--kvmem-historical-capture", "32", *extra],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        check("early configuration rejection " + " ".join(extra),
              result.returncode != 0 and "historical server requires" in result.stdout)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-ngl", "99", "-c", "2048", "-b", "64", "-ub", "16", "-n", "4", "-t", "4",
        "--no-ui", "--kv-dtype", "f16", "--kvmem-method", "recency",
        "--kvmem-block-tokens", "32", "--kvmem-budget", "128", "--kvmem-gen-reserve", "32",
        "--kvmem-historical-capture", "32", "--kvmem-historical-gc-mode", args.mode,
        "--reasoning-effort", "none", "--kvmem-trace", "-lv", "4"]
    if not args.native_template:
        command += ["--chat-template-file", str(template)]
    if args.pinned_gb:
        command += ["--kvmem-historical-pinned-gb", str(args.pinned_gb)]
    if args.mode != "off":
        command += ["--kvmem-historical-gc-low", "20", "--kvmem-historical-gc-high", "26"]
    (out / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    log_path = out / "server.log"

    def call(path, body=None):
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def body(text, **kwargs):
        return {"messages": [{"role": "user", "content": text}], "max_tokens": 4,
                "temperature": 0, "seed": 42, "grammar": 'root ::= "Z"', **kwargs}

    def turn(label, text, **kwargs):
        status, raw = call("/v1/chat/completions", body(text, **kwargs))
        result = json.loads(raw)
        replies.append({"label": label, "status": status, "response": result})
        check(label, status == 200 and "choices" in result and "error" not in result)
        return result

    def idle():
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status, raw = call("/slots")
            if status == 200 and not json.loads(raw)[0]["is_processing"]:
                return
            time.sleep(0.02)
        raise AssertionError("cancelled HTTP request did not release its lane")

    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}; see {log_path}")
                try:
                    if call("/health")[0] == 200:
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("model startup timeout")
            status, raw = call("/props")
            config = json.loads(raw)["kvmem"]["historical"]
            check("published historical configuration", status == 200 and config["capacity"] == 32 and
                  config["mode"] == args.mode and config["admission"] == "finite_request" and
                  config["pinned_bytes"] == int(args.pinned_gb * 1024 ** 3))
            if args.native_template:
                first = turn("native chat template first turn", "Write Z.")
                continuation = body("Write Z again.")
                continuation["messages"] = [{"role": "user", "content": "Write Z."},
                    first["choices"][0]["message"], {"role": "user", "content": "Write Z again."}]
                status, raw = call("/v1/chat/completions", continuation)
                result = json.loads(raw)
                replies.append({"label": "native continuation", "status": status, "response": result})
                check("native chat template second turn", status == 200 and "choices" in result)
                turn("native template pressure/reset", " p" * 384 + "\nWrite Z.", cache_reset=True)
                return
            prompt = "Reply with Z.\n" + " a" * 64 + "\n"
            first = turn("first finite request", prompt)
            answer = first["choices"][0]["message"]["content"]
            check("constrained output", answer == "Z")
            extension = prompt + answer + "\nMore text.\n"
            second = turn("exact prefix extension", extension)
            check("exact prefix was reused", second["usage"]["prompt_cache_hit_tokens"] > 0)
            # Oversized demand fits n_ctx but cannot fit the finite RAM pool.
            for streaming in (False, True):
                status, raw = call("/v1/chat/completions", body(" x" * 1100, stream=streaming))
                check(f"reservation refusal before SSE={streaming}", status == 503 and
                      "reservation unavailable" in json.loads(raw)["error"])
            third = turn("old cache after refused resets", extension + "Z\nNext.\n")
            check("refusal preserved previous cache", third["usage"]["prompt_cache_hit_tokens"] > 0)
            reset = turn("explicit reset", prompt, cache_reset=True)
            check("reset reports no cache hit", reset["usage"]["prompt_cache_hit_tokens"] == 0)
            status, raw = call("/v1/chat/completions", body("Different conversation.\n", stream=True))
            check("SSE completes", status == 200 and "data: [DONE]" in raw and '"error"' not in raw)
            replies.append({"label": "SSE", "status": status, "response": raw})
            for streaming in (False, True):
                status, raw = call("/v1/responses", {"input": "Respond with a single Z.\n",
                    "max_output_tokens": 4, "temperature": 0, "stream": streaming})
                if streaming:
                    ok = "response.completed" in raw and '"error"' not in raw
                else:
                    result = json.loads(raw)
                    ok = result.get("status") == "completed" and "error" not in result
                check(f"Responses transport SSE={streaming}", status == 200 and ok)
                replies.append({"label": "Responses", "status": status, "response": raw})

            for phase in ("prefill", "generation"):
                text = (" c" * 384 + "\n") if phase == "prefill" else "Repeat Z indefinitely.\n"
                request = body(text, stream=True, max_tokens=8 if phase == "prefill" else 24,
                               grammar='root ::= "Z"+')
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
                connection.request("POST", "/v1/chat/completions", json.dumps(request),
                                   {"Content-Type": "application/json"})
                response = connection.getresponse()
                check("cancellation admitted during " + phase, response.status == 200)
                while True:
                    line = response.readline()
                    if not line:
                        raise AssertionError("SSE ended before cancellation point")
                    if line.startswith(b"data: "):
                        event = json.loads(line[6:])
                        delta = event.get("choices", [{}])[0].get("delta", {})
                        if phase == "prefill" or delta.get("content"):
                            break
                if phase == "prefill":
                    time.sleep(0.015)
                # shutdown, not just close(): the response still owns a socket file.
                connection.sock.shutdown(socket.SHUT_RDWR)
                response.close()
                connection.close()
                idle()
                after = turn("new request after " + phase + " cancellation", "After cancellation.\n")
                check("cancelled cache invalidated for " + phase,
                      after["usage"]["prompt_cache_hit_tokens"] == 0)

            for cycle in range(4):
                turn(f"pressure request {cycle}", f"Cycle {cycle}.\n" + " p" * 384 + "\n", cache_reset=True)
            idle()
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if args.pinned_gb:
                pinned_allocations = re.findall(r"KVMEM_HISTORICAL_CAPTURE begin .*?pinned_bytes=(\d+)", text)
                check("real pinned request reservations", bool(pinned_allocations) and
                      all(0 < int(value) <= int(args.pinned_gb * 1024 ** 3) for value in pinned_allocations))
            cancelled = re.findall(r"KVMEM_HISTORICAL_CAPTURE cancel rows=(\d+) input_remaining=(\d+)", text)
            check("unused reservations queued on both cancellations", len(cancelled) >= 2)
            check("partial input and generation cancellation reached",
                  any(int(rows) > 0 and int(left) > 0 for rows, left in cancelled) and
                  any(int(left) == 0 for _, left in cancelled))
            check("no checkpoint rollback on historical path", "KVMEM_CHECKPOINT_ROLLBACK" not in text and
                  "historical cancellation failed" not in text)
            if args.mode != "off":
                stats = re.findall(r"KVMEM_HISTORICAL_GC finish .*?committed=(\d+) shadow=(\d+) reclaimed=(\d+).*?policy_failures=(\d+) worker_failures=(\d+)", text)
                check("GC completion statistics", bool(stats) and all(int(p) == int(w) == 0 for _, _, _, p, w in stats))
                if args.mode == "delete":
                    check("actual background release", any(int(c) > 0 and int(r) > 0 for c, _, r, _, _ in stats))
                else:
                    check("shadow proposals without deletion", any(int(s) > 0 for _, s, _, _, _ in stats) and
                          all(int(c) == int(r) == 0 for c, _, r, _, _ in stats))
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        (out / "results.json").write_text(json.dumps({"checks": checks, "replies": replies}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
