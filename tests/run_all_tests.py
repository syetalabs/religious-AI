"""
run_all_tests.py — Run Buddhism, Christianity, and Hinduism Religious-AI test suites
Usage:
    python run_all_tests.py

Requires: GROQ_API_KEY in .env and a running backend at API_BASE (default localhost:8000)
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from test_utils import (
    BOLD, GREEN, RED, YELLOW, CYAN, RESET,
    wait_for_religion, run_suite, print_summary,
)

from test_buddhism import CASES as BU_CASES
from test_christianity import CASES as CH_CASES
from test_hinduism import CASES as HI_CASES   

SUITES = [
    ("Buddhism", BU_CASES),
    ("Christianity", CH_CASES),
    ("Hinduism", HI_CASES),   
]


def main():
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Religious-AI — All Test Suites{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")

    start_time = time.time()
    all_results = []

    for religion, cases in SUITES:
        if not wait_for_religion(religion):
            print(f"{RED}  {religion} failed to load — skipping.{RESET}\n")
            continue

        results = run_suite(religion, cases)
        print_summary(religion, results)
        all_results.extend(results)

    # ── Grand total ───────────────────────────────────────────
    passed  = sum(1 for r in all_results if r.passed)
    total   = len(all_results)
    elapsed = int(time.time() - start_time)
    pct     = 100 * passed // total if total else 0
    colour  = GREEN if pct >= 75 else (YELLOW if pct >= 50 else RED)

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  TOTAL: {colour}{passed}/{total} passed ({pct}%){RESET}  [{elapsed}s]")
    print(f"{BOLD}{'═'*60}{RESET}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()