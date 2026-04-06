import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import diff_only_feature_set


def main() -> None:
    dataset = diff_only_feature_set()
    X, y, metadata = dataset.X, dataset.y, dataset.metadata


if __name__ == "__main__":
    main()
