"""Fetch the exact public model/runtime used by the HW8 experiment (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


MODEL_REPO = "Qwen/Qwen3-1.7B-GGUF"
MODEL_REVISION = "90862c4b9d2787eaed51d12237eafdfe7c5f6077"
MODEL_FILE = "Qwen3-1.7B-Q8_0.gguf"
MODEL_SHA256 = "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a"
RUNTIME_TAG = "b12722"
RUNTIME_COMMIT = "e6443bd7eb970d940b0669cd9ea0d82d91d620a1"
RUNTIME_FILE = "llama-b12722-bin-win-vulkan-x64.zip"
RUNTIME_SHA256 = "1241fdbe50964b74bad10b873688fb95aad3504fd7b159f80a59f59fbddb31f8"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(url, target, expected):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and digest(target) == expected:
        print(f"Already verified: {target.name}", flush=True)
        return
    partial = target.with_suffix(target.suffix + ".partial")
    started = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": "FAR-Trans-HW8-reproduction"})
    with urllib.request.urlopen(request, timeout=180) as source, partial.open("wb") as destination:
        total = int(source.headers.get("Content-Length", "0"))
        copied = 0
        last_message = 0
        while block := source.read(4 * 1024 * 1024):
            destination.write(block)
            copied += len(block)
            if copied - last_message >= 128 * 1024 * 1024:
                print(f"{target.name}: {copied / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
                last_message = copied
    actual = digest(partial)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {target.name}: {actual}")
    partial.replace(target)
    print(f"Verified {target.name}, {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    model = args.base_dir / "models" / MODEL_FILE
    archive = args.base_dir / "runtime" / RUNTIME_FILE
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(download, f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}", model, MODEL_SHA256),
            pool.submit(download, f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_TAG}/{RUNTIME_FILE}", archive, RUNTIME_SHA256),
        ]
        for future in futures:
            future.result()
    runtime = args.base_dir / "runtime" / RUNTIME_TAG
    runtime.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        root = runtime.resolve()
        for member in source.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError("Unsafe path in runtime archive")
        source.extractall(runtime)
    metadata = {
        "model_repository": MODEL_REPO, "model_revision": MODEL_REVISION,
        "model_file": MODEL_FILE, "model_sha256": MODEL_SHA256,
        "quantization": "Q8_0", "runtime_repository": "ggml-org/llama.cpp",
        "runtime_tag": RUNTIME_TAG, "runtime_commit": RUNTIME_COMMIT,
        "runtime_archive": RUNTIME_FILE, "runtime_archive_sha256": RUNTIME_SHA256,
        "backend": "Vulkan", "model_path": str(model.resolve()),
        "runtime_directory": str(runtime.resolve()),
    }
    (args.base_dir / "runtime" / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
