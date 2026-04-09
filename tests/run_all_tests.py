"""
run_buddhism_tests.py — Run only the Buddhism Religious-AI test suite
Usage:
    python run_buddhism_tests.py

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

# Import only Buddhism cases
from test_buddhism import CASES as BU_CASES


def main():
    religion = "Buddhism"

    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Religious-AI  Buddhism Test Suite{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")

    start_time = time.time()

    # Load Buddhism
    if not wait_for_religion(religion):
        print(f"{RED}  {religion} failed to load — exiting.{RESET}\n")
        sys.exit(1)

    # Run tests
    results = run_suite(religion, BU_CASES)
    print_summary(religion, results)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    elapsed = int(time.time() - start_time)
    pct = 100 * passed // total if total else 0
    colour = GREEN if pct >= 75 else (YELLOW if pct >= 50 else RED)

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  TOTAL: {colour}{passed}/{total} passed ({pct}%){RESET}  [{elapsed}s]")
    print(f"{BOLD}{'═'*60}{RESET}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()