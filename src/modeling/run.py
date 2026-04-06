"""CLI for standardized model evaluation."""

import argparse

from .registry import MODEL_CLASSES
from ..evaluation import evaluate_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UFC models.")
    parser.add_argument("model", choices=["all", *sorted(MODEL_CLASSES)], help="Model to evaluate.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_names = sorted(MODEL_CLASSES) if args.model == "all" else [args.model]
    for model_name in model_names:
        evaluate_model(MODEL_CLASSES[model_name]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
