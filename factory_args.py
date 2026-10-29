"""Tell the user which command-line flags the current factory path will ignore."""

from __future__ import annotations

import argparse
import sys


def explicit_flags(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> list[str]:
    """Option strings the user actually typed, not defaults."""
    tokens = sys.argv[1:] if argv is None else argv
    found: list[str] = []
    for token in tokens:
        name = token.split("=", 1)[0]
        for action in parser._actions:
            if name in action.option_strings:
                found.append(name)
                break
    return found


def warn_unused_arguments(
    parser: argparse.ArgumentParser,
    applicable: set[str],
    scope: str,
    argv: list[str] | None = None,
) -> list[str]:
    """Print flags that were passed but this factory path does not execute.

    Returns the ignored option strings. Defaults are not warned about.
    """
    tokens = sys.argv[1:] if argv is None else argv
    ignored: list[str] = []
    for token in tokens:
        name = token.split("=", 1)[0]
        for action in parser._actions:
            if name in action.option_strings and action.dest not in applicable:
                ignored.append(name)
                break
    if ignored:
        joined = ", ".join(ignored)
        print(f"[!] 这些参数对 {scope} 不会执行，已忽略: {joined}", flush=True)
    return ignored
