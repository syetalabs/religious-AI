"""
test_utils.py  —  Shared utilities for Religious-AI test suite
LLM-as-judge scoring via Groq (reuses GROQ_API_KEY from .env)
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

PASS_THRESHOLD   = 6   # score ≥ 6/10 to pass
TIMEOUT_SECONDS  = 60  # per /ask call

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
    expect_flag:  bool = False          # True → expect flagged=True
    expect_no_ctx: bool = False         # True → expect low_confidence=True


@dataclass
class TestResult:
    case:         TestCase
    passed:       bool
    score:        int                   # 0-10
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
    """Poll /status/{religion} until ready or timeout."""
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
Score the answer on a scale from 0 to 10 using these criteria:
- Correctness (0-4): Is the answer factually accurate and relevant to the question?
- Grounding (0-3): Is the answer grounded in scripture/religious teaching (not hallucinated)?
- Language quality (0-3): Is the answer clear, natural, and free of garbled or untranslated text?

Deduct points for:
- Hallucinated scripture references or verse numbers not in the provided context
- Answering a different question than was asked
- Garbled, untranslated, or incoherent text
- Refusal without good reason

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{"score": <0-10>, "reason": "<one sentence>"}"""

def judge(question: str, answer: str, language: str) -> tuple[int, str]:
    """Returns (score 0-10, one-line reason)."""
    if not GROQ_API_KEY:
        return 7, "GROQ_API_KEY not set — skipping judge"

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
        # Strip any markdown code fences just in case
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

    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Testing: {religion}  ({len(cases)} cases){RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")

    for i, case in enumerate(cases, 1):
        label = case.description or case.question[:55]
        print(f"  [{i:02d}] {label}")

        t0 = time.time()
        try:
            data = ask(case.question, religion, case.language)
        except Exception as exc:
            r = TestResult(case=case, passed=False, score=0, error=str(exc))
            results.append(r)
            print(f"        {RED}ERROR: {exc}{RESET}")
            continue

        elapsed = int((time.time() - t0) * 1000)
        answer  = data.get("answer", "")
        flagged = data.get("flagged", False)
        low_conf = data.get("confidence_warning", False)

        # Special-case: expect flagged
        if case.expect_flag:
            passed = flagged
            score  = 10 if flagged else 0
            reason = "Correctly flagged" if flagged else "Should have been flagged"
        # Special-case: expect low confidence / no-context
        elif case.expect_no_ctx:
            passed = low_conf or "not have enough" in answer.lower()
            score  = 10 if passed else 3
            reason = "Correctly refused" if passed else "Should have returned no-context"
        else:
            score, reason = judge(case.question, answer, case.language)
            passed = score >= PASS_THRESHOLD

        result = TestResult(
            case=case, passed=passed, score=score,
            answer=answer, sources=data.get("sources", []),
            flagged=flagged, low_conf=low_conf,
            judge_reason=reason, duration_ms=elapsed,
        )
        results.append(result)

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"        {status}  score={score}/10  ({elapsed}ms)")
        print(f"        {YELLOW}{reason}{RESET}")
        if not passed:
            print(f"        Answer: {answer[:120]}...")
        print()

    return results


def print_summary(religion: str, results: list[TestResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    pct    = 100 * passed // total if total else 0
    colour = GREEN if pct >= 75 else (YELLOW if pct >= 50 else RED)

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{religion} Results: {colour}{passed}/{total} passed ({pct}%){RESET}")

    fails = [r for r in results if not r.passed]
    if fails:
        print(f"\n  {RED}Failed cases:{RESET}")
        for r in fails:
            print(f"    • [{r.case.language}] {r.case.description or r.case.question[:60]}")
            print(f"      Score {r.score}/10 — {r.judge_reason}")
    print()
