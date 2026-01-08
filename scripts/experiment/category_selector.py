#!/usr/bin/env python3
"""
Interactive Category Selector for I2I Experiments
Provides a CLI menu for selecting prompt categories to run.
"""

import sys
from typing import Optional

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

CATEGORIES = {
    "A": "Neutral Baseline (검증용)",
    "B": "Occupational Stereotype (직업)",
    "C": "Cultural/Religious Expression (문화/종교)",
    "D": "Vulnerability Attributes (취약성)",
    "E": "Harmful/Safety-Triggering (안전)",
}


def print_header():
    """Print menu header."""
    print()
    print(f"{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}{BOLD}   I2I Refusal Bias Study - Category Selector{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    print()


def print_categories(selected: set):
    """Print category list with selection status."""
    print(f"{CYAN}프롬프트 카테고리 선택:{RESET}")
    print()

    # ALL option
    all_selected = len(selected) == 5
    mark = f"{GREEN}[X]{RESET}" if all_selected else "[ ]"
    print(f"  {mark} {BOLD}0. ALL{RESET} - 전체 카테고리 (50 prompts)")
    print()

    # Individual categories
    for key, desc in CATEGORIES.items():
        is_selected = key in selected
        mark = f"{GREEN}[X]{RESET}" if is_selected else "[ ]"
        prompts_info = "10 prompts"
        print(f"  {mark} {key}. {desc} ({prompts_info})")

    print()


def print_selection_summary(selected: set):
    """Print current selection summary."""
    if not selected:
        print(f"{YELLOW}선택된 카테고리: 없음{RESET}")
        return

    sorted_cats = sorted(selected)
    total_prompts = len(selected) * 10
    total_requests = total_prompts * 84  # 84 images

    print(f"{GREEN}선택됨: {', '.join(sorted_cats)} ({total_prompts} prompts x 84 images = {total_requests:,} requests){RESET}")


def interactive_select() -> Optional[str]:
    """
    Interactive category selection menu.

    Returns:
        Comma-separated category string (e.g., "A,B,C") or None if cancelled
    """
    selected = set()

    while True:
        print_header()
        print_categories(selected)
        print_selection_summary(selected)
        print()
        print(f"{CYAN}명령어:{RESET}")
        print("  숫자/문자 입력: 카테고리 토글 (0=ALL, A-E=개별)")
        print("  Enter: 선택 확정")
        print("  q: 취소")
        print()

        try:
            choice = input(f"{BOLD}선택> {RESET}").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if choice == "":
            # Confirm selection
            if not selected:
                print(f"{RED}최소 1개 이상의 카테고리를 선택하세요.{RESET}")
                input("Press Enter to continue...")
                continue
            break

        if choice == "Q":
            return None

        if choice == "0":
            # Toggle ALL
            if len(selected) == 5:
                selected.clear()
            else:
                selected = set(CATEGORIES.keys())
        elif choice in CATEGORIES:
            # Toggle individual category
            if choice in selected:
                selected.remove(choice)
            else:
                selected.add(choice)
        else:
            print(f"{RED}잘못된 입력: {choice}{RESET}")
            input("Press Enter to continue...")

    return ",".join(sorted(selected))


def quick_select() -> str:
    """
    Quick one-line category selection.

    Returns:
        Comma-separated category string
    """
    print()
    print(f"{CYAN}카테고리 빠른 선택{RESET}")
    print("  0 또는 ALL = 전체")
    print("  개별 카테고리: A, B, C, D, E (콤마로 구분)")
    print("  예: A,B 또는 A,B,C,D,E")
    print()

    while True:
        try:
            choice = input(f"{BOLD}카테고리 입력 (기본값: ALL)> {RESET}").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if choice == "" or choice == "0" or choice == "ALL":
            return "A,B,C,D,E"

        # Parse comma-separated
        cats = [c.strip() for c in choice.split(",")]
        valid = all(c in CATEGORIES for c in cats)

        if valid and cats:
            return ",".join(sorted(set(cats)))
        else:
            invalid = [c for c in cats if c not in CATEGORIES]
            print(f"{RED}잘못된 카테고리: {invalid}{RESET}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive category selector")
    parser.add_argument("--quick", action="store_true",
                       help="Quick one-line selection mode")
    parser.add_argument("--default", type=str, default=None,
                       help="Default selection (skip menu if provided)")
    args = parser.parse_args()

    if args.default:
        # Validate and return default
        cats = [c.strip().upper() for c in args.default.split(",")]
        if args.default.upper() in ("ALL", "0"):
            print("A,B,C,D,E")
        elif all(c in CATEGORIES for c in cats):
            print(",".join(sorted(set(cats))))
        else:
            sys.exit(1)
        return

    if args.quick:
        result = quick_select()
    else:
        result = interactive_select()

    if result:
        print(result)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
