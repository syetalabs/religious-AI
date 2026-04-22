"""
test_utils.py — Shared utilities for Religious-AI test suites
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

PASS_THRESHOLD  = 6
TIMEOUT_SECONDS = 60

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
# API helpers
# ──────────────────────────────────────────────────────────────
def ask(question: str, religion: str, language: str = "en") -> dict:
    resp = requests.post(
        f"{API_BASE}/ask",
        json={"question": question, "religion": religion, "language": language},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def wait_for_religion(religion: str, timeout: int = 120) -> bool:
    print(f"  [wait] Waiting for {religion} to load", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{API_BASE}/status/{religion}", timeout=10)
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
# LLM Judge
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
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
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
# Runner
# ──────────────────────────────────────────────────────────────
def run_suite(religion: str, cases: list[TestCase]) -> list[TestResult]:
    results: list[TestResult] = []

    print(f"\n{BOLD}{CYAN}{'═'*50}{RESET}")
    print(f"{BOLD}{CYAN}  Testing {religion} ({len(cases)} cases){RESET}")
    print(f"{BOLD}{CYAN}{'═'*50}{RESET}\n")

    for i, case in enumerate(cases, 1):
        print(f"[{i}] {case.description or case.question[:50]}")

        t0 = time.time()
        try:
            data = ask(case.question, religion, case.language)
        except Exception as exc:
            print(f"{RED}ERROR: {exc}{RESET}\n")
            results.append(TestResult(
                case=case, passed=False, score=0, error=str(exc)
            ))
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
            answer=answer, judge_reason=reason, duration_ms=elapsed,
        ))

    return results


# ──────────────────────────────────────────────────────────────
# Run multiple religion suites
# ──────────────────────────────────────────────────────────────
def run_all_suites(suites: dict[str, list[TestCase]]) -> dict[str, list[TestResult]]:
    all_results = {}

    for religion, cases in suites.items():
        if not wait_for_religion(religion):
            print(f"{RED}Skipping {religion} (not ready){RESET}")
            continue

        results = run_suite(religion, cases)
        print_summary(religion, results)
        all_results[religion] = results

    return all_results


def print_summary(religion: str, results: list[TestResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    pct    = 100 * passed // total if total else 0
    colour = GREEN if pct >= 75 else (YELLOW if pct >= 50 else RED)

    print(f"\n{BOLD}{'─'*50}{RESET}")
    print(f"{religion} Results: {colour}{passed}/{total} passed ({pct}%){RESET}")
    print(f"{BOLD}{'─'*50}{RESET}")