"""CLI to print feature names for a chosen feature set."""

from __future__ import annotations

import argparse

from .features import FEATURE_BUILDERS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print all feature names for a chosen feature set."
    )
    parser.add_argument(
        "feature_set",
        choices=sorted(FEATURE_BUILDERS),
        help="Feature set to inspect.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepared = FEATURE_BUILDERS[args.feature_set]()

    print(f"Feature set: {prepared.name}")
    print(f"Feature count: {prepared.X.shape[1]}")
    print(f"Row count: {prepared.X.shape[0]}")
    print()

    for column in prepared.X.columns:
        print(column)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
