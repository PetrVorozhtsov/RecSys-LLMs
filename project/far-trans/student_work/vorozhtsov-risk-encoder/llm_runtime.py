"""Actual local open-weight Qwen inference via a loopback-only llama.cpp server."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from download_runtime import digest


PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "profile": {
            "type": "array", "minItems": 7, "maxItems": 7,
            "items": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
    },
    "required": ["profile"], "additionalProperties": False,
}

RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {
    "name": "risk_profile", "strict": True, "schema": PROFILE_SCHEMA,
}}


class LocalGenerator:
    """Greedy, non-thinking chat inference; all reported tokens come from the server.

    The schema constrains JSON structure and numeric ranges, not profile semantics.
    The caller checks normalization and implements any statistical fallback itself.
    """

    def __init__(self, model_path=None, server_url="http://127.0.0.1:8089", auto_start=False):
        self.base_dir = Path(__file__).resolve().parent.parent
        manifest_path = self.base_dir / "runtime" / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.model_path = Path(model_path or self.manifest["model_path"])
        if digest(self.model_path) != self.manifest["model_sha256"]:
            raise ValueError("Model file does not match the pinned SHA-256")
        self.server_url = server_url.rstrip("/")
        if not self.server_url.startswith("http://127.0.0.1:"):
            raise ValueError("This experiment only sends its inputs to a loopback-local server")
        self.metadata = dict(self.manifest)
        driver_store = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "DriverStore" / "FileRepository"
        nvidia_icds = list(driver_store.glob("nv*/nv-vk64.json"))
        self.server_environment = os.environ.copy()
        if nvidia_icds:
            selected_icd = str(max(nvidia_icds, key=lambda path: path.stat().st_mtime))
            self.server_environment["VK_DRIVER_FILES"] = selected_icd
            self.server_environment["VK_ICD_FILENAMES"] = selected_icd
            self.metadata["vulkan_icd_path"] = selected_icd
            self.metadata["vulkan_driver_selection"] = "NVIDIA ICD only, per-process environment"
        directory = Path(self.manifest["runtime_directory"])
        matches = list(directory.rglob("llama-server.exe"))
        if len(matches) != 1:
            raise FileNotFoundError("Run download_runtime.py first; exactly one llama-server.exe is required")
        self.executable = matches[0]
        port = self.server_url.rsplit(":", 1)[1]
        self.server_command = [str(self.executable), "--model", str(self.model_path), "--host", "127.0.0.1",
                               "--port", port, "--ctx-size", "4096", "--parallel", "1",
                               "--gpu-layers", "99", "--threads", "8", "--threads-batch", "8",
                               "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}',
                               "--no-webui"]
        self.metadata.update({
            "temperature": 0.0, "seed": 42, "top_k": 1, "top_p": 1.0,
            "repeat_penalty": 1.0, "thinking": False,
            "grammar": {"type": "json_schema", "schema": PROFILE_SCHEMA},
            "context_limit": 4096, "parallel_slots": 1, "gpu_layers": 99,
            "threads": 8, "threads_batch": 8, "server_url": self.server_url,
            "platform": platform.platform(), "python_version": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "gpu": "NVIDIA GeForce GTX 1650, 4096 MiB, Vulkan",
            "inference_scope": "one sequential request per user; no response cache",
            "server_command": self.server_command,
        })
        if auto_start:
            self.ensure_server()
        elif self.healthy():
            self.verify_server()

    def _get(self, endpoint):
        with urllib.request.urlopen(self.server_url + endpoint, timeout=10) as reply:
            return json.load(reply)

    def healthy(self):
        try:
            return self._get("/health").get("status") == "ok"
        except (OSError, urllib.error.URLError):
            return False

    def verify_server(self):
        """Reject an unrelated loopback service or a differently configured model."""
        properties = self._get("/props")
        active_model = Path(properties.get("model_path", "")).resolve()
        settings = properties.get("default_generation_settings", {})
        if active_model != self.model_path.resolve():
            raise ValueError("The local server is serving a different model file")
        if settings.get("n_ctx") != 4096 or properties.get("total_slots") != 1:
            raise ValueError("The local server context/parallel-slot settings differ from this experiment")
        if properties.get("build_info") != "b12722-e6443bd7e":
            raise ValueError("The local server build differs from the pinned experiment runtime")

    def ensure_server(self, timeout=180):
        if self.healthy():
            self.verify_server()
            return
        command = self.server_command
        log_dir = self.base_dir / "runtime"
        with (log_dir / "server.stdout.log").open("ab") as out, (log_dir / "server.stderr.log").open("ab") as err:
            process = subprocess.Popen(command, cwd=self.executable.parent, stdout=out, stderr=err,
                                       env=self.server_environment,
                                       creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        (log_dir / "server.pid").write_text(str(process.pid), encoding="ascii")
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            if self.healthy():
                self.verify_server()
                return
            if process.poll() is not None:
                raise RuntimeError(f"llama.cpp server exited with {process.returncode}; see runtime/server.stderr.log")
            time.sleep(1)
        process.terminate()
        process.wait(timeout=15)
        raise TimeoutError("llama.cpp server did not become ready; see runtime/server.stderr.log")

    def generate(self, prompt, max_new_tokens=256):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("A nonempty prompt is required")
        request_data = {
            "model": self.manifest["model_file"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens, "temperature": 0.0, "seed": 42,
            "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": RESPONSE_FORMAT,
            "cache_prompt": False, "stream": False,
        }
        request = urllib.request.Request(self.server_url + "/v1/chat/completions",
                                         data=json.dumps(request_data).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as reply:
            response = json.load(reply)
        elapsed = time.perf_counter() - started
        usage = response["usage"]
        return {"text": response["choices"][0]["message"]["content"],
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"], "seconds": elapsed,
                "finish_reason": response["choices"][0]["finish_reason"],
                "server_timings": response.get("timings"),
                "model_response_id": response.get("id"),
                "generation_settings": {k: request_data[k] for k in
                    ("max_tokens", "temperature", "seed", "top_k", "top_p", "repeat_penalty",
                     "chat_template_kwargs", "response_format", "cache_prompt")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "smoke-test"])
    args = parser.parse_args()
    generator = LocalGenerator(auto_start=True)
    print(json.dumps(generator.metadata, indent=2), flush=True)
    if args.command == "smoke-test":
        result = generator.generate('Return JSON with the key "profile" containing exactly seven arrays. '
                                    'Every array must contain three numbers: 0.2, 0.3, 0.5. '
                                    'Return only the JSON object.', 256)
        parsed = json.loads(result["text"])
        assert list(parsed) == ["profile"] and len(parsed["profile"]) == 7
        assert all(len(row) == 3 and all(isinstance(value, (int, float)) and 0 <= value <= 1
                                      for value in row) for row in parsed["profile"])
        destination = generator.base_dir / "runtime" / "schema_smoke_test.json"
        destination.write_text(json.dumps({"metadata": generator.metadata, "result": result}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
