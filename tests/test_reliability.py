import copy
import unittest

from backend import main


class ProviderCircuitBreakerTests(unittest.TestCase):
    def setUp(self):
        with main._PROVIDER_RUNTIME_LOCK:
            self.original = copy.deepcopy(main._PROVIDER_RUNTIME)
        self.original_quota_cooldown = main.settings.provider_quota_cooldown_seconds
        main.settings.provider_quota_cooldown_seconds = 30

    def tearDown(self):
        with main._PROVIDER_RUNTIME_LOCK:
            main._PROVIDER_RUNTIME.clear()
            main._PROVIDER_RUNTIME.update(self.original)
        main.settings.provider_quota_cooldown_seconds = self.original_quota_cooldown

    def test_quota_failure_opens_circuit_and_success_closes_it(self):
        main._mark_provider_failure("gemini", RuntimeError("429 quota exceeded"), quota=True)
        self.assertFalse(main._provider_is_available("gemini"))
        status = main.get_provider_runtime_status()["gemini"]
        self.assertGreater(status["cooldown_seconds"], 0)
        self.assertEqual(status["failures"], 1)

        main._mark_provider_success("gemini")
        self.assertTrue(main._provider_is_available("gemini"))
        self.assertEqual(main.get_provider_runtime_status()["gemini"]["failures"], 0)

    def test_retryable_provider_errors_are_classified(self):
        self.assertTrue(main._is_retryable_provider_error(RuntimeError("503 service unavailable")))
        self.assertTrue(main._is_retryable_provider_error(TimeoutError("timed out")))
        self.assertFalse(main._is_retryable_provider_error(ValueError("invalid model schema")))


if __name__ == "__main__":
    unittest.main()
