"""
test_utils_buddhism.py — Utilities for Buddhism-only Religious-AI test suite
"""
import os
import json
import time
import requests
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
API_BASE     = os.environ.get("API_BASE", "http://localhost:8000")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_MODEL  = "llama-3.3-70b-versatile"

RELIGION = "Buddhism"

PASS_THRESHOLD   = 6
TIMEOUT_SECONDS  = 60

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────
@dataclass
class TestCase:
    question:     str
    language:     str = "en"
    description:  str = ""
    expect_flag:  bool = False
    expect_no_ctx: bool = False


@dataclass
class TestResult:
    case:         TestCase
    passed:       bool
    score:        int
    answer:       str = ""
    sources:      list = field(default_factory=list)
    flagged:      bool = False
    low_conf:     bool = False
    judge_reason: str = ""
    error:        str = ""
    duration_ms:  int = 0


# ──────────────────────────────────────────────────────────────
# API helpers (Buddhism only)
# ──────────────────────────────────────────────────────────────
def ask(question: str, language: str = "en") -> dict:
    resp = requests.post(
        f"{API_BASE}/ask",
        json={"question": question, "religion": RELIGION, "language": language},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def wait_for_buddhism(timeout: int = 120) -> bool:
    print(f"  [wait] Waiting for Buddhism to load", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{API_BASE}/status/{RELIGION}", timeout=10)
            if r.json().get("status") == "ready":
                print(" ✓")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(3)
    print(" ✗ TIMEOUT")
    return False


# ──────────────────────────────────────────────────────────────
# LLM Judge (unchanged)
# ──────────────────────────────────────────────────────────────
_JUDGE_SYSTEM = """You are a strict quality evaluator for a scripture-based religious chatbot.
Score the answer on a scale from 0 to 10.

Return ONLY JSON:
{"score": <0-10>, "reason": "<one sentence>"}"""

def judge(question: str, answer: str, language: str) -> tuple[int, str]:
    if not GROQ_API_KEY:
        return 7, "No API key — skipped"

    prompt = f"Language: {language}\nQuestion: {question}\nAnswer: {answer}"
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": JUDGE_MODEL,
                "messages": [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 120,
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = raw.strip("`").replace("json", "", 1).strip()
        data = json.loads(raw)
        return int(data["score"]), str(data.get("reason", ""))
    except Exception as exc:
        return 5, f"Judge error: {exc}"


# ──────────────────────────────────────────────────────────────
# Runner (Buddhism only)
# ──────────────────────────────────────────────────────────────
def run_suite(cases: list[TestCase]) -> list[TestResult]:
    results: list[TestResult] = []

    print(f"\n{BOLD}{CYAN}{'═'*50}{RESET}")
    print(f"{BOLD}{CYAN}  Testing Buddhism ({len(cases)} cases){RESET}")
    print(f"{BOLD}{CYAN}{'═'*50}{RESET}\n")

    for i, case in enumerate(cases, 1):
        print(f"[{i}] {case.description or case.question[:50]}")

        t0 = time.time()
        try:
            data = ask(case.question, case.language)
        except Exception as exc:
            print(f"{RED}ERROR: {exc}{RESET}")
            continue

        elapsed = int((time.time() - t0) * 1000)
        answer  = data.get("answer", "")

        score, reason = judge(case.question, answer, case.language)
        passed = score >= PASS_THRESHOLD

        print(f"{GREEN if passed else RED}{'PASS' if passed else 'FAIL'}{RESET} "
              f"score={score}/10 ({elapsed}ms)")
        print(f"{YELLOW}{reason}{RESET}\n")

        results.append(TestResult(
            case=case, passed=passed, score=score,
            answer=answer, judge_reason=reason, duration_ms=elapsed
        ))

    return results


def print_summary(results: list[TestResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total  = len(results)

    print(f"\n{BOLD}{'─'*50}{RESET}")
    print(f"Buddhism Results: {passed}/{total} passed")