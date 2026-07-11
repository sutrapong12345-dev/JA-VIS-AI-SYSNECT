"""Quality gates for J.A.R.V.I.S. supervised fine-tuning data.

The production chat logs may contain retries, provider errors, unsafe advice,
prompt-injection attempts, or confident claims that do not match the actual
role/tool boundary.  Those turns must not be copied blindly into a fine-tune.
This module is intentionally dependency-free so validation can run on the
backend machine before a GPU training environment is prepared.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple


_USER_START = "<|start_header_id|>user<|end_header_id|>\n\n"
_ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>\n\n"
_EOT = "<|eot_id|>"

_HARD_REJECT_ASSISTANT = {
    "false_privilege_claim": (
        "god mode", "admin major", "เข้าถึงและดำเนินการต่างๆ ได้อย่างเต็มที่",
        "มีสิทธิ์ทุกอย่าง", "ได้รับสิทธิ์สูงสุดแล้ว",
    ),
    "credential_or_bypass_advice": (
        "ลองใช้ข้อมูลผู้ดูแลระบบเดิม", "admin_password", "admin_username",
        "ชื่อผู้ใช้: [admin", "password: [admin", "ssh key", "ssh 密钥",
    ),
    "provider_or_system_error": (
        "เกิดข้อผิดพลาดในการเชื่อมต่อ", "ยังไม่ได้ตั้งค่า api", "admin mode unlocked",
        "เกินโควต้าชั่วคราว", "ระบบสั่งการถูกปิดใช้งาน",
    ),
}

_SOFT_ISSUES = {
    "weak_uncertainty": ("น่าจะ", "อาจจะ", "คงจะ"),
    "over_apology": ("ขออภัยอีกครั้ง", "ต้องขออภัย"),
    "persona_inconsistent": ("ฉันสามารถ", "ฉันไม่สามารถ", "ค่ะคุณบอส", "คะคุณบอส"),
}

_SECRET_RE = re.compile(
    r"(?i)(?:api[_ -]?key|secret|password|รหัสผ่าน)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/.]{12,}"
)
_MIXED_SCRIPT_RE = re.compile(r"[\u0400-\u04ff\u4e00-\u9fff]")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    return _SPACE_RE.sub(" ", (value or "").strip()).lower()


def parse_llama3_text(text: str) -> Optional[Tuple[str, str]]:
    """Extract a single user/assistant pair from the repository's Llama-3 format."""
    if _USER_START not in text or _ASSISTANT_START not in text:
        return None
    tail = text.split(_USER_START, 1)[1]
    if _ASSISTANT_START not in tail:
        return None
    user, assistant_tail = tail.split(_ASSISTANT_START, 1)
    user = user.split(_EOT, 1)[0].strip()
    assistant = assistant_tail.split(_EOT, 1)[0].strip()
    return user, assistant


def assess_pair(user: str, assistant: str) -> Dict[str, object]:
    """Score one pair and return deterministic, explainable quality findings."""
    issues: List[str] = []
    hard_reject = False
    user_norm = normalize_text(user)
    assistant_norm = normalize_text(assistant)

    if len(user_norm) < 2 or len(assistant_norm) < 10:
        issues.append("too_short")
        hard_reject = True
    if len(user) > 4000 or len(assistant) > 8000:
        issues.append("too_long")
        hard_reject = True
    if _SECRET_RE.search(user) or _SECRET_RE.search(assistant):
        issues.append("possible_secret")
        hard_reject = True
    if _MIXED_SCRIPT_RE.search(assistant):
        issues.append("mixed_language_noise")
        hard_reject = True

    for issue, markers in _HARD_REJECT_ASSISTANT.items():
        if any(marker in assistant_norm for marker in markers):
            issues.append(issue)
            hard_reject = True

    for issue, markers in _SOFT_ISSUES.items():
        if any(marker in assistant_norm for marker in markers):
            issues.append(issue)

    # A refusal that only says authentication is required is not sufficient
    # training material when the user asks for harmful access; it should also
    # state the safe organizational path. Mark it for review, not hard reject.
    harmful_request = any(marker in user_norm for marker in (
        "เจาะระบบ", "ขโมย", "bypass", "ignore all previous", "เปิดเผย system prompt",
        "สิทธิ์ admin", "รหัสผ่าน", "ช่องโหว่ทั้งหมด",
    ))
    safe_boundary = any(marker in assistant_norm for marker in (
        "ไม่สามารถ", "ไม่ได้รับอนุญาต", "ยืนยันสิทธิ์", "ผู้ดูแลระบบ",
        "นโยบาย", "ขอบเขต", "audit",
    ))
    if harmful_request and not safe_boundary:
        issues.append("unsafe_request_not_bounded")
        hard_reject = True

    score = 100
    penalties = {
        "too_short": 70, "too_long": 40, "possible_secret": 100,
        "mixed_language_noise": 55, "false_privilege_claim": 100,
        "credential_or_bypass_advice": 100, "provider_or_system_error": 75,
        "unsafe_request_not_bounded": 100, "weak_uncertainty": 8,
        "over_apology": 8, "persona_inconsistent": 12,
    }
    for issue in set(issues):
        score -= penalties.get(issue, 10)
    score = max(0, score)

    return {
        "accepted": not hard_reject and score >= 70,
        "score": score,
        "issues": sorted(set(issues)),
        "fingerprint": hashlib.sha256((user_norm + "\0" + assistant_norm).encode("utf-8")).hexdigest()[:16],
    }


def quality_report(pairs: Iterable[Tuple[str, str]]) -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    issue_counts: Counter = Counter()
    seen_pairs = set()
    seen_user_counts: Counter = Counter()

    for user, assistant in pairs:
        result = assess_pair(user, assistant)
        pair_key = normalize_text(user) + "\0" + normalize_text(assistant)
        user_key = normalize_text(user)
        if pair_key in seen_pairs:
            result["accepted"] = False
            result["score"] = min(int(result["score"]), 40)
            result["issues"] = sorted(set(result["issues"]) | {"duplicate_pair"})
        seen_pairs.add(pair_key)
        seen_user_counts[user_key] += 1
        # Repeated retries for the same prompt bias the model. Keep at most two.
        if seen_user_counts[user_key] > 2:
            result["accepted"] = False
            result["score"] = min(int(result["score"]), 45)
            result["issues"] = sorted(set(result["issues"]) | {"repeated_prompt"})
        issue_counts.update(result["issues"])
        results.append(result)

    total = len(results)
    accepted = sum(1 for item in results if item["accepted"])
    rejected = total - accepted
    average_score = round(sum(int(item["score"]) for item in results) / total, 1) if total else 0.0
    pass_rate = round((accepted / total) * 100, 1) if total else 0.0
    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "average_score": average_score,
        "pass_rate": pass_rate,
        "ready_for_training": accepted >= 100 and average_score >= 80 and pass_rate >= 80,
        "issue_counts": dict(issue_counts.most_common()),
        "results": results,
    }


def inspect_jsonl(path: str) -> Dict[str, object]:
    pairs: List[Tuple[str, str]] = []
    malformed = 0
    if not os.path.isfile(path):
        report = quality_report([])
        report.update({"path": path, "exists": False, "malformed": 0})
        return report
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                pair = parse_llama3_text(str(payload.get("text", "")))
            except (ValueError, TypeError):
                pair = None
            if pair is None:
                malformed += 1
            else:
                pairs.append(pair)
    report = quality_report(pairs)
    report.update({"path": path, "exists": True, "malformed": malformed})
    if malformed:
        report["ready_for_training"] = False
        report["issue_counts"]["malformed"] = malformed
    return report

