import json
import os
import tempfile
import unittest

from backend import main
from backend.training_quality import assess_pair, inspect_jsonl, quality_report


class TrainingQualityTests(unittest.TestCase):
    def test_truthful_role_bounded_answer_is_accepted(self):
        result = assess_pair(
            "ฉันมีสิทธิ์ควบคุมเครื่องทั้งหมดหรือไม่",
            "ยังสรุปไม่ได้ครับ สิทธิ์ขึ้นอยู่กับบทบาทและเครื่องมือที่เปิดใช้งาน ผมจะตรวจสิทธิ์ก่อนดำเนินการ",
        )
        self.assertTrue(result["accepted"])
        self.assertGreaterEqual(result["score"], 80)

    def test_false_god_mode_claim_is_rejected(self):
        result = assess_pair(
            "ฉันเป็นแอดมินใช่ไหม",
            "คุณได้รับ GOD MODE และมีสิทธิ์ทุกอย่างในเครื่องแล้วครับ",
        )
        self.assertFalse(result["accepted"])
        self.assertIn("false_privilege_claim", result["issues"])

    def test_mixed_script_noise_is_rejected(self):
        result = assess_pair("ช่วยตั้งผู้ดูแลระบบ", "请进入ระบบแล้วใช้ учетные данные เพื่อดำเนินการครับ")
        self.assertFalse(result["accepted"])
        self.assertIn("mixed_language_noise", result["issues"])

    def test_duplicate_and_repeated_prompts_are_limited(self):
        pairs = [
            ("ตรวจระบบ", "ผมจะตรวจสถานะจากเครื่องมือก่อนแล้วรายงานผลตามจริงครับ"),
            ("ตรวจระบบ", "ผมจะตรวจสถานะจากเครื่องมือก่อนแล้วสรุปค่าที่ได้รับครับ"),
            ("ตรวจระบบ", "ผมจะตรวจข้อมูลจริงก่อนตอบและไม่คาดเดาผลลัพธ์ครับ"),
        ]
        report = quality_report(pairs)
        self.assertEqual(report["accepted"], 2)
        self.assertEqual(report["issue_counts"].get("repeated_prompt"), 1)

    def test_jsonl_inspection_reports_malformed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dataset.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("not json\n")
            report = inspect_jsonl(path)
            self.assertEqual(report["malformed"], 1)
            self.assertFalse(report["ready_for_training"])

    def test_builder_filters_unsafe_logs_and_splits_output(self):
        original_logs = main.LOGS_DIR
        original_training = main.TRAINING_DIR
        with tempfile.TemporaryDirectory() as directory:
            try:
                main.LOGS_DIR = os.path.join(directory, "logs")
                main.TRAINING_DIR = os.path.join(directory, "training")
                os.makedirs(main.LOGS_DIR)
                os.makedirs(main.TRAINING_DIR)
                curated = [
                    {"user": "ระบบพร้อมไหม", "assistant": "ผมจะตรวจสถานะจริงก่อนรายงานผลครับ"},
                    {"user": "ข้อมูลมาจากไหน", "assistant": "ผมจะตอบจากเอกสารพร้อม citation ที่ตรวจสอบได้ครับ"},
                ]
                with open(os.path.join(main.TRAINING_DIR, "curated_seed.json"), "w", encoding="utf-8") as fh:
                    json.dump(curated, fh, ensure_ascii=False)
                history = [
                    {"role": "user", "content": "ฉันมีสิทธิ์เต็มหรือไม่"},
                    {"role": "assistant", "content": "คุณมี GOD MODE และได้รับสิทธิ์สูงสุดแล้วครับ"},
                ]
                with open(os.path.join(main.LOGS_DIR, "chat_log_test.json"), "w", encoding="utf-8") as fh:
                    json.dump(history, fh, ensure_ascii=False)

                result = main.build_training_dataset()
                self.assertEqual(result["examples"], 2)
                self.assertEqual(result["quality"]["source_rejected"], 1)
                self.assertEqual(result["quality"]["rejected"], 0)
                self.assertTrue(os.path.isfile(os.path.join(main.TRAINING_DIR, "dataset_train.jsonl")))
                self.assertTrue(os.path.isfile(os.path.join(main.TRAINING_DIR, "dataset_validation.jsonl")))
            finally:
                main.LOGS_DIR = original_logs
                main.TRAINING_DIR = original_training


if __name__ == "__main__":
    unittest.main()
