"""
run_all_tests.py  —  Run all four Religious-AI test suites in sequence
Usage:
    python run_all_tests.py                    # all four religions
    python run_all_tests.py Buddhism           # single religion
    python run_all_tests.py Buddhism Islam     # specific set

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

# ──────────────────────────────────────────────────────────────
# Import per-religion case lists
# ──────────────────────────────────────────────────────────────
from test_buddhism    import CASES as BU_CASES
from test_christianity import CASES as CH_CASES
from test_hinduism    import CASES as HI_CASES
from test_islam       import CASES as IS_CASES

ALL_SUITES = {
    "Buddhism":    BU_CASES,
    "Christianity": CH_CASES,
    "Hinduism":    HI_CASES,
    "Islam":       IS_CASES,
}


def main():
    # Determine which religions to test
    if len(sys.argv) > 1:
        religions = [r for r in sys.argv[1:] if r in ALL_SUITES]
        unknown   = [r for r in sys.argv[1:] if r not in ALL_SUITES]
        if unknown:
            print(f"{YELLOW}Unknown religion(s): {unknown}. Valid: {list(ALL_SUITES)}{RESET}")
            sys.exit(1)
    else:
        religions = list(ALL_SUITES)

    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Religious-AI  Full Test Suite{RESET}")
    print(f"{BOLD}{CYAN}  Religions: {', '.join(religions)}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")

    grand_passed = 0
    grand_total  = 0
    all_results  = {}
    start_time   = time.time()

    for religion in religions:
        # Load the religion (server handles eviction of the previous one)
        if not wait_for_religion(religion):
            print(f"{RED}  {religion} failed to load — skipping.{RESET}\n")
            continue

        cases   = ALL_SUITES[religion]
        results = run_suite(religion, cases)
        print_summary(religion, results)

        passed = sum(1 for r in results if r.passed)
        grand_passed += passed
        grand_total  += len(results)
        all_results[religion] = (passed, len(results))

    # ── Grand summary ──────────────────────────────────────────
    elapsed = int(time.time() - start_time)
    pct     = 100 * grand_passed // grand_total if grand_total else 0
    colour  = GREEN if pct >= 75 else (YELLOW if pct >= 50 else RED)

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  GRAND TOTAL: {colour}{grand_passed}/{grand_total} passed ({pct}%){RESET}  [{elapsed}s]")
    print(f"{BOLD}{'═'*60}{RESET}")

    for religion, (p, t) in all_results.items():
        bar_colour = GREEN if p == t else (YELLOW if p >= t * 0.75 else RED)
        print(f"  {religion:<14} {bar_colour}{p}/{t}{RESET}")

    print()
    sys.exit(0 if grand_passed == grand_total else 1)


if __name__ == "__main__":
    main()
