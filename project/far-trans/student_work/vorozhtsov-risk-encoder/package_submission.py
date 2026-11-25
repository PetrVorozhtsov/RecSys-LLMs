"""Package report, reproducible code/results and a documented coursework AI export.

The export retains visible messages, generated code, calls and their outputs.
Private reasoning, system/developer instructions and internal runtime metadata
are excluded. HW2 explicitly permits a documented derivative session export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SESSION_FILES = {
    "root": "rollout-2026-11-25T15-19-22-01a212a5-7456-7781-bd22-c5a5b3689c78.jsonl",
    "requirements_report": "rollout-2026-11-25T15-29-09-01a212ae-699b-76f2-af54-14bb98c176f0.jsonl",
    "risk_implementation": "rollout-2026-11-25T15-31-39-01a212b0-b075-7570-a9af-4046b6aea2e8.jsonl",
    "inference_runtime": "rollout-2026-11-25T15-31-54-01a212b0-eb81-7e41-bb6a-49bdc4534041.jsonl",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def visible_text(payload):
    return "\n".join(item.get("text", "") for item in payload.get("content", [])
                     if isinstance(item, dict))


def export_session(source, destination, label):
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    first_meta = next(row["payload"] for row in rows if row.get("type") == "session_meta")
    created = first_meta.get("timestamp", rows[0].get("timestamp", ""))
    allowed_types = {"function_call", "function_call_output", "custom_tool_call",
                     "custom_tool_call_output", "agent_message"}
    selected = []
    web_calls = set()
    started = False
    for index, row in enumerate(rows, 1):
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            continue
        # Forked history can be re-timestamped at child creation. Its first
        # parent-triggered task marks actual child work more reliably than time.
        if (label != "root" and row.get("type") == "event_msg"
                and payload.get("type") == "task_started"
                and payload.get("turn_attribution", {}).get("initiating_agent_path") is not None):
            started = True
        if row.get("type") != "response_item":
            continue
        item_type = payload.get("type")
        role = payload.get("role")
        if label == "root" and item_type == "message" and role == "user":
            if "выполни последнее задание" in visible_text(payload).lower():
                started = True
        if not started or (label != "root" and row.get("timestamp", "") < created):
            continue
        if item_type in ("function_call", "custom_tool_call"):
            descriptor = str(payload.get("name", "")) + str(payload.get("arguments", payload.get("input", "")))
            if any(marker in descriptor for marker in ("tools.web__run(", "search_service_web_run", "web.run")):
                web_calls.add(payload.get("call_id"))
        keep = item_type in allowed_types
        if item_type == "message":
            keep = role == "user" or (role == "assistant" and payload.get("phase") in ("commentary", "final_answer", "final"))
        if keep:
            if item_type in ("function_call_output", "custom_tool_call_output") and payload.get("call_id") in web_calls:
                original = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
                payload = {"type": item_type, "call_id": payload.get("call_id"),
                           "output": {"fetched_source_body_omitted": True,
                                      "original_payload_sha256": digest(original),
                                      "note": "The source request/URLs are retained; primary-source findings and citations are in the report. Computational results are retained in full."}}
            selected.append({"source_session": label, "source_line": index,
                             "timestamp": row.get("timestamp"), "type": row["type"],
                             "payload": payload})
    if not selected:
        raise ValueError("No relevant visible session records found in " + str(source))
    metadata = {"format": "documented-coursework-derivative-v1", "source_session": label,
                "source_file": source.name, "source_snapshot_sha256": digest(raw),
                "session_id": first_meta.get("session_id", first_meta.get("id")),
                "cli_version": first_meta.get("cli_version"),
                "scope": "Visible coursework messages, code, calls and computational outputs. Fetched web source bodies are replaced by hashes; internal instructions/reasoning/runtime metadata omitted."}
    exported = ("\n".join(json.dumps(value, ensure_ascii=False) for value in [metadata] + selected) + "\n").encode("utf-8")
    destination.write_bytes(exported)
    return {"file": destination.name, "source": source.name, "records": len(selected),
            "source_snapshot_sha256": digest(raw), "export_sha256": digest(exported)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", type=Path, default=Path.home() / ".codex" / "sessions" / "2026" / "11" / "25")
    parser.add_argument("--skip-export", action="store_true", help="Repackage existing documented exports on another computer")
    args = parser.parse_args()
    submission = ROOT / "submission"
    report = submission / "vorozhtsov_petr_session2_report.pdf"
    if not report.exists() or not (ROOT / "outputs/main/verification.json").exists():
        raise FileNotFoundError("Complete evaluation, independent audit and PDF first")
    export_dir = submission / "ai_sessions"
    export_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_export:
        entries = []
        for label, filename in SESSION_FILES.items():
            entries.append(export_session(args.sessions_dir / filename, export_dir / (label + ".jsonl"), label))
        manifest = {"format": "documented-coursework-derivative-v1", "sessions": entries,
                    "includes": "Visible coursework messages, generated code, tool calls and computational results, including delegated implementation.",
                    "omits": "Internal reasoning/instructions, runtime bookkeeping, inherited earlier unrelated records and fetched web source bodies (requests and payload hashes retained).",
                    "root_start": "User request to inspect the instructor repository and complete HW8.",
                    "note": "Snapshot taken after the completed experiment and PDF checks; later packaging/final messages may be absent."}
        (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (export_dir / "README.txt").write_text(
            "Documented AI-session derivative for FAR-Trans HW2\n\n"
            "The assignment permits a documented derivative preserving generated code, calls and results.\n"
            "Each JSONL record retains the source response-item payload plus its original session label,\n"
            "line number and timestamp. Tool call arguments, generated patches/code and tool results\n"
            "are preserved. Fetched web source bodies are replaced by their hashes; source requests/URLs\n"
            "remain recorded, and consulted-source findings/citations are in the PDF. Computational\n"
            "outputs are retained in full. Visible coursework messages are retained. Internal reasoning,\n"
            "system/developer instructions and runtime bookkeeping are omitted. Child exports exclude\n"
            "inherited earlier context; the root begins at the execution request. The manifest records\n"
            "the SHA-256 of each source snapshot and each derivative. Code and results also appear\n"
            "as normal files in this package for convenient verification.\n",
            encoding="utf-8")
    target = submission / "vorozhtsov_petr_hw8_submission.zip"
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in (".git", "__pycache__", ".venv") for part in relative.parts):
            continue
        if path.suffix.lower() in (".zip", ".gguf", ".dll", ".exe", ".pyc", ".png"):
            continue
        if relative.as_posix() == "outputs/main/baseline/contexts.jsonl":
            continue
        files.append((path, relative))
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, relative in files:
            archive.write(path, "vorozhtsov_hw8/" + relative.as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Submission archive failed CRC check")
    print(json.dumps({"archive": str(target), "bytes": target.stat().st_size,
                      "files": len(files), "sha256": digest(target.read_bytes())}, indent=2))


if __name__ == "__main__":
    main()
