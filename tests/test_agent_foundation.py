import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from backend.agent_tools import ToolValidationError, validate_tool_call
from backend.audit_store import AuditStore
from backend.rag_store import RagStore
from backend import main


class AuditStoreTests(unittest.TestCase):
    def test_hash_chain_is_valid_and_rows_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AuditStore(os.path.join(directory, "audit.db"))
            store.append("test.one", "success", actor_id="employee@example.com")
            store.append("test.two", "denied", actor_id="employee@example.com")
            self.assertTrue(store.verify_chain()["valid"])
            connection = sqlite3.connect(store.path)
            try:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute("UPDATE audit_events SET outcome='changed'")
            finally:
                connection.close()


class RoleAwareRagTests(unittest.TestCase):
    def test_document_acl_and_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge = os.path.join(directory, "knowledge")
            os.makedirs(knowledge)
            with open(os.path.join(knowledge, "staff__handbook.md"), "w", encoding="utf-8") as handle:
                handle.write("นโยบายวันลาพนักงาน company vacation policy")
            with open(os.path.join(knowledge, "admin__secret.md"), "w", encoding="utf-8") as handle:
                handle.write("รหัสระบบสำรอง disaster recovery secret")
            store = RagStore(os.path.join(directory, "knowledge.db"))
            store.index_directory(knowledge)

            staff = store.search("vacation policy", "staff")
            self.assertEqual(staff[0]["source"], "staff__handbook.md")
            self.assertTrue(staff[0]["citation"].startswith("[KB:"))
            self.assertEqual(store.search("disaster recovery secret", "staff"), [])
            self.assertEqual(store.search("disaster recovery secret", "admin")[0]["source"], "admin__secret.md")


class StructuredToolTests(unittest.TestCase):
    def test_role_and_argument_validation(self):
        call = validate_tool_call(
            {"name": "list_processes", "arguments": {"limit": 999}}, "manager"
        )
        self.assertEqual(call["arguments"]["limit"], 20)
        with self.assertRaises(ToolValidationError):
            validate_tool_call(
                {"name": "list_processes", "arguments": {"limit": 5}}, "staff"
            )
        with self.assertRaises(ToolValidationError):
            validate_tool_call(
                {"name": "system_status", "arguments": {"command": "whoami"}}, "admin"
            )

    def test_restricted_worker_has_no_generic_command_tool(self):
        worker = os.path.join(os.path.dirname(__file__), "..", "backend", "sandbox_worker.py")
        completed = subprocess.run(
            [sys.executable, worker],
            input=json.dumps({"tool": "shell", "arguments": {"command": "whoami"}}),
            capture_output=True,
            text=True,
            timeout=10,
        )
        response = json.loads(completed.stdout)
        self.assertFalse(response["ok"])

    def test_agent_runs_registered_read_only_tool_in_worker(self):
        result = main.agent.process(
            '[TOOL: {"name":"system_status","arguments":{}}]',
            session_id="tool_test", role="staff", actor_id="tester@example.com",
        )
        self.assertIn("cpu_percent", result)

    def test_malformed_tool_tag_is_a_safe_denial(self):
        result = main.agent.process(
            '[TOOL: {"name": bad-json}]',
            session_id="tool_test", role="staff", actor_id="tester@example.com",
        )
        self.assertIn("รูปแบบไม่ถูกต้อง", result)


if __name__ == "__main__":
    unittest.main()
