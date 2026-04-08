"""Feature-set builders for UFC modeling."""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .config import UFC_CLEAN_CSV

FeatureBuilder = Callable[[pd.DataFrame | None], "PreparedDataset"]

DROP_KEYWORDS = ("odds", "prob")
BASE_DROP_COLUMNS = {
    "RedFighter",
    "BlueFighter",
    "Date",
    "EventDate",
    "Location",
    "Country",
    "Winner",
    "WinnerRed",
}
METADATA_COLUMNS = [
    "EventDate",
    "Date",
    "RedFighter",
    "BlueFighter",
    "RedDecimalOdds",
    "BlueDecimalOdds",
    "RedImpliedProb",
    "BlueImpliedProb",
]
DIFF_CONTEXT_COLUMNS = [
    "TitleBout",
    "Gender",
    "NumberOfRounds",
    "EmptyArena",
]
WEIGHTCLASS_COLUMN = "WeightClass"


@dataclass(frozen=True)
class PreparedDataset:
    """Standardized data payload shared by every model evaluation."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


def load_interim_ufc_data(path=UFC_CLEAN_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def load_ground_zero_frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = load_interim_ufc_data() if df is None else df.copy()
    frame = frame[frame["Winner"].isin(["Red", "Blue"])].copy()
    frame = frame.drop(columns=["EmptyArena"], errors="ignore")
    frame["WinnerRed"] = (frame["Winner"] == "Red").astype(int)
    frame["EventDate"] = pd.to_datetime(frame["Date"])
    return frame.sort_values("EventDate").reset_index(drop=True)


def _build_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[METADATA_COLUMNS].reset_index(drop=True)


def _finalize_dataset(
    name: str,
    frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    drop_missing: bool,
) -> PreparedDataset:
    model_frame = pd.concat([frame[["WinnerRed"]], feature_frame], axis=1)
    if drop_missing:
        model_frame = model_frame.dropna()

    aligned_frame = frame.loc[model_frame.index].reset_index(drop=True)
    X = model_frame.drop(columns=["WinnerRed"]).reset_index(drop=True)
    y = model_frame["WinnerRed"].reset_index(drop=True)
    metadata = _build_metadata(aligned_frame)
    return PreparedDataset(name=name, X=X, y=y, metadata=metadata)


def _drop_columns_by_keyword(
    frame: pd.DataFrame, keywords: tuple[str, ...]
) -> list[str]:
    return [
        column
        for column in frame.columns
        if any(keyword in column.lower() for keyword in keywords)
    ]


def _paired_corner_columns(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for column in sorted(frame.columns):
        if not column.startswith("Red"):
            continue
        counterpart = f"Blue{column[3:]}"
        if counterpart not in frame.columns:
            continue
        if column in {"RedFighter", "RedDecimalOdds", "RedImpliedProb"}:
            continue
        if any(keyword in column.lower() for keyword in DROP_KEYWORDS):
            continue
        pairs.append((column[3:], column, counterpart))
    return pairs


def _build_raw_feature_frame(
    frame: pd.DataFrame, *, include_weightclass: bool
) -> pd.DataFrame:
    drop_columns = set(_drop_columns_by_keyword(frame, DROP_KEYWORDS))
    drop_columns.update(BASE_DROP_COLUMNS)
    if not include_weightclass:
        drop_columns.add(WEIGHTCLASS_COLUMN)
    return frame.drop(columns=sorted(drop_columns), errors="ignore")


def _build_ground_zero_complete_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    feature_frame = _build_raw_feature_frame(frame, include_weightclass=False)
    categorical_columns = feature_frame.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    return pd.get_dummies(feature_frame, columns=categorical_columns, drop_first=True)


def _build_context_feature_frame(
    frame: pd.DataFrame, *, include_weightclass: bool
) -> pd.DataFrame:
    drop_columns = set(_drop_columns_by_keyword(frame, DROP_KEYWORDS))
    drop_columns.update(BASE_DROP_COLUMNS)
    if not include_weightclass:
        drop_columns.add(WEIGHTCLASS_COLUMN)
    drop_columns.update(
        column
        for _, red_column, blue_column in _paired_corner_columns(frame)
        for column in (red_column, blue_column)
    )

    feature_frame = frame.drop(columns=sorted(drop_columns), errors="ignore")
    categorical_columns = feature_frame.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    return pd.get_dummies(feature_frame, columns=categorical_columns, drop_first=True)


def _build_diff_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    diff_features: dict[str, pd.Series] = {}

    for stem, red_column, blue_column in _paired_corner_columns(frame):
        red_series = frame[red_column]
        blue_series = frame[blue_column]

        if is_numeric_dtype(red_series) and is_numeric_dtype(blue_series):
            diff_features[f"{stem}Diff"] = red_series - blue_series
            continue

        red_labels = red_series.astype("string").str.strip()
        blue_labels = blue_series.astype("string").str.strip()
        categories = sorted(
            set(red_labels.dropna().unique()) | set(blue_labels.dropna().unique())
        )
        if len(categories) <= 1:
            continue

        red_dummies = pd.get_dummies(red_labels, dtype=int).reindex(
            columns=categories, fill_value=0
        )
        blue_dummies = pd.get_dummies(blue_labels, dtype=int).reindex(
            columns=categories, fill_value=0
        )

        for category in categories[1:]:
            diff_features[f"Red{stem}_{category}"] = red_dummies[category]
            diff_features[f"Blue{stem}_{category}"] = blue_dummies[category]

    return pd.DataFrame(diff_features, index=frame.index)


def _build_diff_context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    context_columns = [
        column for column in DIFF_CONTEXT_COLUMNS if column in frame.columns
    ]
    return frame[context_columns].copy()


def _build_weightclass_main_effect_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if WEIGHTCLASS_COLUMN not in frame.columns:
        return pd.DataFrame(index=frame.index)

    weightclass = frame[WEIGHTCLASS_COLUMN].astype("string").str.strip()
    return pd.get_dummies(weightclass, prefix="WeightClass", drop_first=True, dtype=int)


def _build_weightclass_interaction_frame(
    diff_feature_frame: pd.DataFrame, weightclass_frame: pd.DataFrame
) -> pd.DataFrame:
    interaction_features: dict[str, pd.Series] = {}
    diff_columns = sorted(
        column for column in diff_feature_frame.columns if column.endswith("Diff")
    )

    for diff_column in diff_columns:
        for weightclass_column in weightclass_frame.columns:
            interaction_name = f"{diff_column}__x__{weightclass_column}"
            interaction_features[interaction_name] = (
                diff_feature_frame[diff_column] * weightclass_frame[weightclass_column]
            )

    return pd.DataFrame(interaction_features, index=diff_feature_frame.index)


def ground_zero_feature_set(df: pd.DataFrame | None = None) -> PreparedDataset:
    frame = load_ground_zero_frame(df)
    feature_frame = _build_raw_feature_frame(frame, include_weightclass=False)
    return _finalize_dataset("ground_zero", frame, feature_frame, drop_missing=False)


def ground_zero_complete_feature_set(df: pd.DataFrame | None = None) -> PreparedDataset:
    frame = load_ground_zero_frame(df)
    feature_frame = _build_ground_zero_complete_feature_frame(frame)
    return _finalize_dataset(
        "ground_zero_complete", frame, feature_frame, drop_missing=True
    )


def diff_feature_set(df: pd.DataFrame | None = None) -> PreparedDataset:
    frame = load_ground_zero_frame(df)
    context_feature_frame = _build_diff_context_frame(frame)
    diff_feature_frame = _build_diff_feature_frame(frame)
    feature_frame = pd.concat([context_feature_frame, diff_feature_frame], axis=1)
    return _finalize_dataset("diff", frame, feature_frame, drop_missing=False)


def diff_complete_feature_set(df: pd.DataFrame | None = None) -> PreparedDataset:
    frame = load_ground_zero_frame(df)
    context_feature_frame = _build_context_feature_frame(
        frame, include_weightclass=False
    )
    diff_feature_frame = _build_diff_feature_frame(frame)
    feature_frame = pd.concat([context_feature_frame, diff_feature_frame], axis=1)
    return _finalize_dataset(
        "diff_complete", frame, feature_frame, drop_missing=True
    )


def _build_weightclass_main_effect_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    context_feature_frame = _build_context_feature_frame(
        frame, include_weightclass=False
    )
    diff_feature_frame = _build_diff_feature_frame(frame)
    weightclass_feature_frame = _build_weightclass_main_effect_frame(frame)
    return pd.concat(
        [context_feature_frame, diff_feature_frame, weightclass_feature_frame], axis=1
    )


def _build_weightclass_interaction_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    base_feature_frame = _build_weightclass_main_effect_feature_frame(frame)
    diff_feature_frame = _build_diff_feature_frame(frame)
    weightclass_feature_frame = _build_weightclass_main_effect_frame(frame)
    interaction_feature_frame = _build_weightclass_interaction_frame(
        diff_feature_frame,
        weightclass_feature_frame,
    )
    return pd.concat([base_feature_frame, interaction_feature_frame], axis=1)


def weightclass_main_effect_feature_set(
    df: pd.DataFrame | None = None,
) -> PreparedDataset:
    frame = load_ground_zero_frame(df)
    feature_frame = _build_weightclass_main_effect_feature_frame(frame)
    return _finalize_dataset(
        "weightclass_main_effect", frame, feature_frame, drop_missing=False
    )


def weightclass_main_effect_complete_feature_set(
    df: pd.DataFrame | None = None,
) -> PreparedDataset:
    frame = load_ground_zero_frame(df)
    feature_frame = _build_weightclass_main_effect_feature_frame(frame)
    return _finalize_dataset(
        "weightclass_main_effect_complete",
        frame,
        feature_frame,
        drop_missing=True,
    )


def weightclass_interaction_feature_set(
    df: pd.DataFrame | None = None,
) -> PreparedDataset:
    frame = load_ground_zero_frame(df)
    feature_frame = _build_weightclass_interaction_feature_frame(frame)
    return _finalize_dataset(
        "weightclass_interaction", frame, feature_frame, drop_missing=False
    )


def weightclass_interaction_complete_feature_set(
    df: pd.DataFrame | None = None,
) -> PreparedDataset:
    frame = load_ground_zero_frame(df)
    feature_frame = _build_weightclass_interaction_feature_frame(frame)
    return _finalize_dataset(
        "weightclass_interaction_complete",
        frame,
        feature_frame,
        drop_missing=True,
    )


FEATURE_BUILDERS: dict[str, FeatureBuilder] = {
    "ground_zero": ground_zero_feature_set,
    "ground_zero_complete": ground_zero_complete_feature_set,
    "diff": diff_feature_set,
    "diff_complete": diff_complete_feature_set,
    "weightclass_main_effect": weightclass_main_effect_feature_set,
    "weightclass_main_effect_complete": weightclass_main_effect_complete_feature_set,
    "weightclass_interaction": weightclass_interaction_feature_set,
    "weightclass_interaction_complete": weightclass_interaction_complete_feature_set,
}
