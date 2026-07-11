import os
import unittest

from fastapi.testclient import TestClient

from backend import main


class SecurityFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def issue_session(self):
        response = self.client.post("/api/session")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        return payload["session_id"], payload["token"]

    @staticmethod
    def auth(token):
        return {"Authorization": f"Bearer {token}"}

    def test_health_is_public_but_does_not_leak_local_paths(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("directory", response.json()["knowledge"])

    def test_session_data_requires_a_server_issued_token(self):
        response = self.client.get("/api/chat/history", params={"session_id": "made_up"})
        self.assertEqual(response.status_code, 401)

    def test_token_can_only_access_its_bound_session(self):
        first_id, first_token = self.issue_session()
        second_id, _ = self.issue_session()

        own = self.client.get(
            "/api/chat/history",
            params={"session_id": first_id},
            headers=self.auth(first_token),
        )
        self.assertEqual(own.status_code, 200)

        other = self.client.get(
            "/api/chat/history",
            params={"session_id": second_id},
            headers=self.auth(first_token),
        )
        self.assertEqual(other.status_code, 403)

    def test_staff_session_cannot_read_admin_logs(self):
        _, token = self.issue_session()
        response = self.client.get("/api/system-log", headers=self.auth(token))
        self.assertEqual(response.status_code, 403)

    def test_admin_login_unlocks_admin_endpoint_for_same_token_only(self):
        session_id, token = self.issue_session()
        old_username = main.settings.admin_username
        old_password = main.settings.admin_password
        try:
            main.settings.admin_username = "test-admin"
            main.settings.admin_password = "test-password"
            login = self.client.post(
                "/api/auth/login",
                headers=self.auth(token),
                json={
                    "username": "test-admin",
                    "password": "test-password",
                    "session_id": session_id,
                },
            )
            self.assertEqual(login.status_code, 200)
            self.assertEqual(login.json()["status"], "success")

            logs = self.client.get("/api/system-log", headers=self.auth(token))
            self.assertEqual(logs.status_code, 200)
        finally:
            main.store.set_admin(session_id, False)
            main.settings.admin_username = old_username
            main.settings.admin_password = old_password
            metadata = os.path.join(main.LOGS_DIR, f"session_metadata_{main.safe_session_key(session_id)}.json")
            if os.path.isfile(metadata):
                os.remove(metadata)

    def test_command_execution_is_safe_by_default(self):
        self.assertFalse(main.Settings().enable_command_execution)
        self.assertFalse(main.Settings().enable_shell_commands)

    def test_system_prompt_renders_json_examples_without_format_errors(self):
        prompt = main.build_system_prompt(
            "clear", "", False, "", "staff",
            "[KB:company.md#chunk-1]\nข้อมูลบริษัท",
        )
        self.assertIn('{"title": "..."', prompt)
        self.assertIn("[KB:company.md#chunk-1]", prompt)

    def test_raw_shell_is_held_for_explicit_approval(self):
        session_id = "test_shell_approval"
        old_shell = main.settings.enable_shell_commands
        old_confirmation = main.settings.require_shell_confirmation
        try:
            main.settings.enable_shell_commands = True
            main.settings.require_shell_confirmation = True
            result = main.agent._shell("Get-Date", session_id)
            self.assertIn("รอการอนุมัติ", result)
            pending = main.store.get_pending_action(session_id)
            self.assertEqual(pending["command"], "Get-Date")
            self.assertTrue(str(pending["id"]).startswith("act_"))
        finally:
            main.store.clear_pending_action(session_id)
            main.settings.enable_shell_commands = old_shell
            main.settings.require_shell_confirmation = old_confirmation


if __name__ == "__main__":
    unittest.main()
