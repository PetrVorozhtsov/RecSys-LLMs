"""The documented coursework export must retain evidence and omit private context."""
import json
import tempfile
import unittest
from pathlib import Path

from package_submission import export_session


class CourseworkExportTests(unittest.TestCase):
    def make_export(self, rows, label="root"):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.jsonl", Path(directory) / "export.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            manifest = export_session(source, target, label)
            return target.read_text(encoding="utf-8"), manifest

    def metadata(self):
        return {"timestamp": "2026-11-25T12:00:00Z", "type": "session_meta",
                "payload": {"id": "example", "timestamp": "2026-11-25T12:00:00Z",
                            "base_instructions": "PRIVATE_INSTRUCTIONS"}}

    def message(self, role, message, phase=None, timestamp="2026-11-25T12:01:00Z"):
        return {"timestamp": timestamp, "type": "response_item",
                "payload": {"type": "message", "role": role, "phase": phase,
                            "content": [{"type": "output_text", "text": message}]}}

    def test_keeps_generated_code_and_results_without_private_records(self):
        call = {"timestamp": "2026-11-25T12:02:00Z", "type": "response_item",
                "payload": {"type": "function_call", "name": "apply_patch", "arguments": "GENERATED_CODE"}}
        output = {"timestamp": "2026-11-25T12:02:01Z", "type": "response_item",
                  "payload": {"type": "function_call_output", "output": "TEST_RESULT"}}
        rows = [self.metadata(), self.message("user", "выполни последнее задание hw8"),
                self.message("developer", "DEVELOPER_PRIVATE"),
                self.message("assistant", "PRIVATE_REASONING", "analysis"),
                {"type": "response_item", "payload": {"type": "reasoning", "summary": "INTERNAL_SUMMARY"}},
                self.message("assistant", "VISIBLE_UPDATE", "commentary"), call, output]
        content, manifest = self.make_export(rows)
        for secret in ("PRIVATE_INSTRUCTIONS", "DEVELOPER_PRIVATE", "PRIVATE_REASONING", "INTERNAL_SUMMARY"):
            self.assertNotIn(secret, content)
        for evidence in ("GENERATED_CODE", "TEST_RESULT", "VISIBLE_UPDATE"):
            self.assertIn(evidence, content)
        payloads = [json.loads(line).get("payload") for line in content.splitlines()]
        self.assertIn(call["payload"], payloads)
        self.assertIn(output["payload"], payloads)
        self.assertEqual(manifest["records"], 4)

    def test_root_omits_unrelated_previous_work(self):
        content, _ = self.make_export([self.metadata(), self.message("user", "OLD_UNRELATED"),
                                      self.message("user", "выполни последнее задание hw8")])
        self.assertNotIn("OLD_UNRELATED", content)

    def test_child_omits_inherited_earlier_context(self):
        content, _ = self.make_export([
            self.metadata(), self.message("user", "INHERITED_OLD", timestamp="2026-11-25T11:00:00Z"),
            self.message("assistant", "NEW_CHILD_RESULT", "final_answer")], label="child")
        self.assertNotIn("INHERITED_OLD", content)
        self.assertIn("NEW_CHILD_RESULT", content)


if __name__ == "__main__":
    unittest.main()
