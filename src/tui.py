"""Stateless Textual TUI for launching and inspecting model evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual import events
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Static

from .config import METRICS_DIR, PREDICTIONS_DIR
from .modeling.registry import MODEL_CLASSES

SORT_FIELDS = [
    ("log_loss", "log loss"),
    ("accuracy", "accuracy"),
    ("brier", "brier"),
    ("roc_auc", "roc auc"),
    ("updated_at", "updated"),
    ("model_name", "model"),
    ("feature_set", "feature set"),
]
DETAIL_BREAKPOINT = 140
DETAIL_PANEL_MIN_WIDTH = 50
DETAIL_PANEL_MAX_WIDTH = 100
MIN_MODEL_LABEL_WIDTH = 26
MIN_FEATURE_LABEL_WIDTH = 16
PREFERRED_MODEL_LABEL_WIDTH = 34
PREFERRED_FEATURE_LABEL_WIDTH = 24
METRIC_COLUMN_WIDTH = 8
TABLE_FRAME_WIDTH = 10


def _clip_label(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return f"{value[: width - 3]}..."


def _feature_set_name(model_name: str) -> str:
    feature_builder_name = MODEL_CLASSES[model_name]().feature_builder.__name__
    return feature_builder_name.removesuffix("_feature_set")


def _display_feature_set(
    model_name: str, metrics: dict[str, object] | None
) -> str:
    if metrics is not None:
        feature_set = metrics.get("feature_set")
        if isinstance(feature_set, str) and feature_set:
            return feature_set
    return _feature_set_name(model_name)


def _read_metrics(path: Path) -> tuple[dict[str, object] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:  # pragma: no cover - defensive UI path
        return None, str(exc)


def _last_updated(*paths: Path) -> datetime | None:
    timestamps = [datetime.fromtimestamp(path.stat().st_mtime) for path in paths if path.exists()]
    return max(timestamps) if timestamps else None


def _format_timestamp(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value is not None else "-"


def _format_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "-"


def _metric_delta_text(model_value: float, bookmaker_value: float, *, lower_is_better: bool) -> Text:
    delta = model_value - bookmaker_value
    improved = delta < 0 if lower_is_better else delta > 0
    style = "green" if improved else "red"
    return Text(f"{delta:+.4f}", style=style)


def _prediction_preview(path: Path, limit: int = 8) -> pd.DataFrame | None:
    if not path.exists():
        return None

    columns = [
        "Date",
        "RedFighter",
        "BlueFighter",
        "WinnerRed",
        "ModelProbRed",
        "BookmakerProbRed",
        "ModelPredRed",
        "BookmakerPredRed",
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame.tail(limit).iloc[::-1].copy()
    frame["Winner"] = frame["WinnerRed"].map({1: "Red", 0: "Blue"})
    frame["ModelOK"] = (frame["WinnerRed"] == frame["ModelPredRed"]).map({True: "yes", False: "no"})
    frame["BookOK"] = (frame["WinnerRed"] == frame["BookmakerPredRed"]).map({True: "yes", False: "no"})
    return frame


def load_model_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in sorted(MODEL_CLASSES):
        metrics_path = METRICS_DIR / f"{model_name}.json"
        predictions_path = PREDICTIONS_DIR / f"{model_name}.csv"
        metrics, metrics_error = _read_metrics(metrics_path)
        updated_at = _last_updated(metrics_path, predictions_path)
        rows.append(
            {
                "model_name": model_name,
                "feature_set": _display_feature_set(model_name, metrics),
                "status": (
                    "metrics error"
                    if metrics_error is not None
                    else ("evaluated" if metrics is not None else "not evaluated")
                ),
                "updated_at": updated_at,
                "updated_display": _format_timestamp(updated_at),
                "metrics_error": metrics_error,
                "metrics_path": metrics_path if metrics_path.exists() else None,
                "predictions_path": predictions_path if predictions_path.exists() else None,
                "train_rows": metrics.get("train_rows") if metrics else None,
                "test_rows": metrics.get("test_rows") if metrics else None,
                "accuracy": metrics.get("accuracy") if metrics else None,
                "bookmaker_accuracy": metrics.get("bookmaker_accuracy") if metrics else None,
                "log_loss": metrics.get("log_loss") if metrics else None,
                "bookmaker_log_loss": metrics.get("bookmaker_log_loss") if metrics else None,
                "brier": metrics.get("brier") if metrics else None,
                "bookmaker_brier": metrics.get("bookmaker_brier") if metrics else None,
                "roc_auc": metrics.get("roc_auc") if metrics else None,
                "bookmaker_roc_auc": metrics.get("bookmaker_roc_auc") if metrics else None,
            }
        )
    return rows


def _run_evaluation_subprocess(model_name: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", "src.modeling.run", model_name],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _tail_lines(text: str, limit: int = 16) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(lines[-limit:])


class HelpScreen(ModalScreen[None]):
    """Keybind help modal."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
        Binding("q", "dismiss", "Close", show=False),
        Binding("question_mark", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        help_text = Table.grid(padding=(0, 2))
        help_text.add_column(style="bold cyan", no_wrap=True)
        help_text.add_column()
        for key, description in [
            ("j / k", "Move the highlighted model"),
            ("g / G", "Jump to the first or last model row"),
            ("space", "Toggle selection on the highlighted model"),
            ("enter", "Run selected models, or the highlighted model if nothing is selected"),
            ("d", "Open the highlighted model details in a modal"),
            ("A", "Select every available model"),
            ("c", "Clear the current selection"),
            ("m", "Sort models alphabetically"),
            ("s", "Cycle the active sort field"),
            ("r", "Reverse sort direction"),
            ("n / f / t", "Sort directly by model name, feature set, or last updated time"),
            ("a / l / b / u", "Sort directly by accuracy, log loss, Brier score, or ROC AUC"),
            ("?", "Show or close this help"),
            ("q", "Quit the TUI"),
        ]:
            help_text.add_row(key, description)

        yield Static(
            Panel(
                help_text,
                title="Keybinds",
                subtitle="Stateless model launcher and result viewer",
                border_style="cyan",
            ),
            id="help-screen",
        )

    def action_dismiss(self) -> None:
        self.dismiss()


class EvaluationTUIApp(App[None]):
    """Stateless evaluator TUI built on current files in reports/."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        height: auto;
        padding: 0 1;
        background: $surface;
        color: $text;
    }

    #body {
        height: 1fr;
    }

    #model-table {
        width: 1fr;
        min-width: 0;
    }

    #detail-wrap {
        width: 44;
        min-width: 32;
        border-left: tall $primary;
    }

    #detail-pane {
        padding: 0 1;
    }

    #help-screen {
        width: 80;
        height: auto;
        padding: 1 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j", "cursor_down", "Down", priority=True),
        Binding("k", "cursor_up", "Up", priority=True),
        Binding("g", "jump_top", show=False, priority=True),
        Binding("G", "jump_bottom", show=False, priority=True),
        Binding("space", "toggle_selection", "Toggle", priority=True),
        Binding("enter", "run_selected", "Run", priority=True),
        Binding("d", "show_details", "Details", priority=True),
        Binding("A", "select_all", "Select All", priority=True),
        Binding("c", "clear_selection", "Clear", priority=True),
        Binding("m", "sort_alpha", "A-Z", priority=True),
        Binding("s", "cycle_sort", "Sort", priority=True),
        Binding("r", "reverse_sort", "Reverse", priority=True),
        Binding("question_mark", "show_help", "Help", key_display="?", priority=True),
        Binding("n", "sort_by('model_name')", show=False, priority=True),
        Binding("f", "sort_by('feature_set')", show=False, priority=True),
        Binding("t", "sort_by('updated_at')", show=False, priority=True),
        Binding("a", "sort_by('accuracy')", show=False, priority=True),
        Binding("l", "sort_by('log_loss')", show=False, priority=True),
        Binding("b", "sort_by('brier')", show=False, priority=True),
        Binding("u", "sort_by('roc_auc')", show=False, priority=True),
    ]

    TITLE = "UFC Evaluation TUI"
    SUB_TITLE = "Stateless launcher and leaderboard"

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, object]] = []
        self.visible_rows: list[dict[str, object]] = []
        self.selected_models: set[str] = set()
        self.sort_field = "log_loss"
        self.sort_reverse = False
        self.status_text = "Ready."
        self.model_label_width = MIN_MODEL_LABEL_WIDTH
        self.feature_label_width = MIN_FEATURE_LABEL_WIDTH

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(id="status")
        with Horizontal(id="body"):
            yield DataTable(id="model-table", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="detail-wrap"):
                yield Static(id="detail-pane")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_rows()
        self._table().focus()

    def _table(self) -> DataTable:
        return self.query_one("#model-table", DataTable)

    def _detail_pane(self) -> Static:
        return self.query_one("#detail-pane", Static)

    def _detail_wrap(self) -> VerticalScroll:
        return self.query_one("#detail-wrap", VerticalScroll)

    def _status_widget(self) -> Static:
        return self.query_one("#status", Static)

    def _current_row(self) -> dict[str, object] | None:
        table = self._table()
        if not self.visible_rows or table.cursor_row < 0 or table.cursor_row >= len(self.visible_rows):
            return None
        return self.visible_rows[table.cursor_row]

    def _current_model_name(self) -> str | None:
        row = self._current_row()
        if row is None:
            return None
        return str(row["model_name"])

    def _sorted_rows(self) -> list[dict[str, object]]:
        present: list[dict[str, object]] = []
        missing: list[dict[str, object]] = []
        for row in self.rows:
            value = row[self.sort_field]
            if value is None:
                missing.append(row)
            else:
                present.append(row)

        def row_key(row: dict[str, object]) -> object:
            value = row[self.sort_field]
            if isinstance(value, str):
                return (value.lower(), row["model_name"])
            return (value, row["model_name"])

        present = sorted(present, key=row_key, reverse=self.sort_reverse)
        missing = sorted(missing, key=lambda row: str(row["model_name"]))
        return [*present, *missing]

    def _update_status(self) -> None:
        sort_label = dict(SORT_FIELDS)[self.sort_field]
        direction = "desc" if self.sort_reverse else "asc"
        selected_count = len(self.selected_models)
        evaluated_count = sum(1 for row in self.rows if row["accuracy"] is not None)
        text = (
            f"{len(self.rows)} models | "
            f"{evaluated_count} evaluated | "
            f"{selected_count} selected | "
            f"sorted by {sort_label} ({direction}) | "
            f"{self.status_text}"
        )
        self._status_widget().update(Text(text))

    def _table_cells(self, row: dict[str, object]) -> list[str]:
        model_name = str(row["model_name"])
        return [
            "[X]" if model_name in self.selected_models else "[ ]",
            _clip_label(model_name, self.model_label_width),
            _clip_label(str(row["feature_set"]), self.feature_label_width),
            _format_metric(row["accuracy"]),
            _format_metric(row["log_loss"]),
            _format_metric(row["brier"]),
            _format_metric(row["roc_auc"]),
        ]

    def _table_content_budget(self, width: int, *, detail_visible: bool) -> int:
        if not detail_visible:
            return max(72, width - 8)

        preferred_table_width = (
            3
            + PREFERRED_MODEL_LABEL_WIDTH
            + PREFERRED_FEATURE_LABEL_WIDTH
            + (4 * METRIC_COLUMN_WIDTH)
            + TABLE_FRAME_WIDTH
        )
        detail_width = max(
            DETAIL_PANEL_MIN_WIDTH,
            min(DETAIL_PANEL_MAX_WIDTH, width - preferred_table_width),
        )
        return max(72, width - detail_width - 8)

    def _column_widths(self, width: int, *, detail_visible: bool) -> tuple[int, int]:
        budget = self._table_content_budget(width, detail_visible=detail_visible)
        fixed_budget = 3 + (4 * METRIC_COLUMN_WIDTH) + TABLE_FRAME_WIDTH
        flexible_budget = max(
            MIN_MODEL_LABEL_WIDTH + MIN_FEATURE_LABEL_WIDTH,
            budget - fixed_budget,
        )
        preferred_model_width = PREFERRED_MODEL_LABEL_WIDTH if detail_visible else 42
        preferred_feature_width = PREFERRED_FEATURE_LABEL_WIDTH if detail_visible else 28
        model_width = max(MIN_MODEL_LABEL_WIDTH, int(flexible_budget * 0.60))
        feature_width = max(
            MIN_FEATURE_LABEL_WIDTH,
            flexible_budget - model_width,
        )
        return (
            min(preferred_model_width, model_width),
            min(preferred_feature_width, feature_width),
        )

    def _configure_table(self) -> None:
        table = self._table()
        detail_visible = self.size.width >= DETAIL_BREAKPOINT
        self.model_label_width, self.feature_label_width = self._column_widths(
            self.size.width,
            detail_visible=detail_visible,
        )
        table.clear(columns=True)
        table.add_column("Sel", width=3)
        table.add_column("Model", width=self.model_label_width)
        table.add_column("Feature", width=self.feature_label_width)
        table.add_column("Acc", width=METRIC_COLUMN_WIDTH)
        table.add_column("Loss", width=METRIC_COLUMN_WIDTH)
        table.add_column("Brier", width=METRIC_COLUMN_WIDTH)
        table.add_column("AUC", width=METRIC_COLUMN_WIDTH)

    def refresh_rows(self, focus_model: str | None = None) -> None:
        self.rows = load_model_rows()
        known_models = {str(row["model_name"]) for row in self.rows}
        self.selected_models &= known_models

        self._apply_layout(self.size.width)
        table = self._table()
        self._configure_table()
        self.visible_rows = self._sorted_rows()
        for row in self.visible_rows:
            table.add_row(*self._table_cells(row), key=str(row["model_name"]))

        if table.row_count:
            if focus_model and focus_model in known_models:
                table.move_cursor(row=table.get_row_index(focus_model), column=0)
            else:
                table.move_cursor(row=0, column=0)

        self._update_status()
        self._update_details()

    def _render_metric_panel(self, row: dict[str, object]) -> RenderableType:
        if row["metrics_error"] is not None:
            error = Table.grid()
            error.add_row(f"Could not read metrics JSON: {row['metrics_error']}")
            return Panel(error, title="Metrics", border_style="red")

        if row["accuracy"] is None:
            empty = Table.grid(padding=(0, 1))
            empty.add_row("No current evaluation output for this model.")
            empty.add_row("Press enter to run the highlighted model, or select several models first.")
            return Panel(empty, title="Metrics", border_style="yellow")

        metric_table = Table(expand=True)
        metric_table.add_column("Metric")
        metric_table.add_column("Model", justify="right")
        metric_table.add_column("Bookmaker", justify="right")
        metric_table.add_column("Delta", justify="right")
        metric_table.add_row(
            "Accuracy",
            _format_metric(row["accuracy"]),
            _format_metric(row["bookmaker_accuracy"]),
            _metric_delta_text(float(row["accuracy"]), float(row["bookmaker_accuracy"]), lower_is_better=False),
        )
        metric_table.add_row(
            "Log Loss",
            _format_metric(row["log_loss"]),
            _format_metric(row["bookmaker_log_loss"]),
            _metric_delta_text(float(row["log_loss"]), float(row["bookmaker_log_loss"]), lower_is_better=True),
        )
        metric_table.add_row(
            "Brier",
            _format_metric(row["brier"]),
            _format_metric(row["bookmaker_brier"]),
            _metric_delta_text(float(row["brier"]), float(row["bookmaker_brier"]), lower_is_better=True),
        )
        metric_table.add_row(
            "ROC AUC",
            _format_metric(row["roc_auc"]),
            _format_metric(row["bookmaker_roc_auc"]),
            _metric_delta_text(float(row["roc_auc"]), float(row["bookmaker_roc_auc"]), lower_is_better=False),
        )
        metric_table.add_row("Train rows", str(row["train_rows"]), "-", "-")
        metric_table.add_row("Test rows", str(row["test_rows"]), "-", "-")
        return Panel(metric_table, title="Metrics", border_style="green")

    def _render_prediction_panel(self, row: dict[str, object]) -> RenderableType:
        predictions_path = row["predictions_path"]
        if predictions_path is None:
            empty = Table.grid()
            empty.add_row("No prediction preview is available yet.")
            return Panel(empty, title="Recent Holdout Predictions", border_style="yellow")

        try:
            preview = _prediction_preview(Path(predictions_path))
        except Exception as exc:  # pragma: no cover - defensive UI path
            error = Table.grid()
            error.add_row(f"Could not read predictions: {exc}")
            return Panel(error, title="Recent Holdout Predictions", border_style="red")

        if preview is None or preview.empty:
            empty = Table.grid()
            empty.add_row("Prediction file is empty.")
            return Panel(empty, title="Recent Holdout Predictions", border_style="yellow")

        prediction_table = Table(expand=True)
        prediction_table.add_column("Date", no_wrap=True)
        prediction_table.add_column("Red")
        prediction_table.add_column("Blue")
        prediction_table.add_column("Win", no_wrap=True)
        prediction_table.add_column("Model", justify="right", no_wrap=True)
        prediction_table.add_column("Book", justify="right", no_wrap=True)
        prediction_table.add_column("M ok", no_wrap=True)
        prediction_table.add_column("B ok", no_wrap=True)

        for fight in preview.itertuples(index=False):
            prediction_table.add_row(
                str(fight.Date),
                str(fight.RedFighter),
                str(fight.BlueFighter),
                str(fight.Winner),
                f"{float(fight.ModelProbRed):.3f}",
                f"{float(fight.BookmakerProbRed):.3f}",
                str(fight.ModelOK),
                str(fight.BookOK),
            )
        return Panel(
            prediction_table,
            title="Recent Holdout Predictions",
            subtitle="latest rows from the current predictions CSV",
            border_style="blue",
        )

    def _detail_renderable(self, row: dict[str, object]) -> RenderableType:
        summary = Table.grid(padding=(0, 1))
        summary.add_row("Model", f"[bold]{row['model_name']}[/bold]")
        summary.add_row("Feature set", str(row["feature_set"]))
        summary.add_row("Status", str(row["status"]))
        summary.add_row("Updated", str(row["updated_display"]))
        summary.add_row("Selected", "yes" if str(row["model_name"]) in self.selected_models else "no")
        return Group(
            Panel(summary, title="Model", border_style="cyan"),
            self._render_metric_panel(row),
            self._render_prediction_panel(row),
        )

    def _update_details(self) -> None:
        model_name = self._current_model_name()
        if model_name is None:
            self._detail_pane().update(Panel("No models available.", title="Details"))
            return

        row = next(item for item in self.rows if item["model_name"] == model_name)
        self._detail_pane().update(self._detail_renderable(row))

    def _apply_layout(self, width: int) -> None:
        detail_visible = width >= DETAIL_BREAKPOINT
        self._detail_wrap().styles.display = "block" if detail_visible else "none"
        if detail_visible:
            preferred_table_width = (
                3
                + PREFERRED_MODEL_LABEL_WIDTH
                + PREFERRED_FEATURE_LABEL_WIDTH
                + (4 * METRIC_COLUMN_WIDTH)
                + TABLE_FRAME_WIDTH
            )
            detail_width = max(
                DETAIL_PANEL_MIN_WIDTH,
                min(DETAIL_PANEL_MAX_WIDTH, width - preferred_table_width),
            )
            self._detail_wrap().styles.width = detail_width

    def on_resize(self, event: events.Resize) -> None:
        self.refresh_rows(focus_model=self._current_model_name())

    def action_cursor_down(self) -> None:
        table = self._table()
        if table.row_count:
            next_row = min(table.cursor_row + 1, table.row_count - 1)
            table.move_cursor(row=next_row, column=0)

    def action_cursor_up(self) -> None:
        table = self._table()
        if table.row_count:
            next_row = max(table.cursor_row - 1, 0)
            table.move_cursor(row=next_row, column=0)

    def action_toggle_selection(self) -> None:
        model_name = self._current_model_name()
        if model_name is None:
            return
        if model_name in self.selected_models:
            self.selected_models.remove(model_name)
        else:
            self.selected_models.add(model_name)
        self.refresh_rows(focus_model=model_name)

    def action_jump_top(self) -> None:
        table = self._table()
        if table.row_count:
            table.move_cursor(row=0, column=0)

    def action_jump_bottom(self) -> None:
        table = self._table()
        if table.row_count:
            table.move_cursor(row=table.row_count - 1, column=0)

    def action_select_all(self) -> None:
        self.selected_models = {str(row["model_name"]) for row in self.rows}
        self.status_text = "Selected all models."
        self.refresh_rows(focus_model=self._current_model_name())

    def action_clear_selection(self) -> None:
        self.selected_models.clear()
        self.status_text = "Cleared model selection."
        self.refresh_rows(focus_model=self._current_model_name())

    def action_sort_alpha(self) -> None:
        self.sort_field = "model_name"
        self.sort_reverse = False
        self.status_text = "Sorting alphabetically."
        self.refresh_rows(focus_model=self._current_model_name())

    def action_cycle_sort(self) -> None:
        current_index = next(index for index, item in enumerate(SORT_FIELDS) if item[0] == self.sort_field)
        self.sort_field = SORT_FIELDS[(current_index + 1) % len(SORT_FIELDS)][0]
        self.status_text = f"Sorting by {dict(SORT_FIELDS)[self.sort_field]}."
        self.refresh_rows(focus_model=self._current_model_name())

    def action_reverse_sort(self) -> None:
        self.sort_reverse = not self.sort_reverse
        direction = "descending" if self.sort_reverse else "ascending"
        self.status_text = f"Sort direction is now {direction}."
        self.refresh_rows(focus_model=self._current_model_name())

    def action_sort_by(self, field: str) -> None:
        self.sort_field = field
        self.status_text = f"Sorting by {dict(SORT_FIELDS)[field]}."
        self.refresh_rows(focus_model=self._current_model_name())

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    @work(thread=True, exclusive=True)
    def _run_models(self, model_names: list[str]) -> None:
        total = len(model_names)
        try:
            for index, model_name in enumerate(model_names, start=1):
                self.call_from_thread(
                    self._set_status_text,
                    f"Running {index}/{total}: {model_name}",
                )
                completed = _run_evaluation_subprocess(model_name)
                if completed.returncode != 0:
                    stderr = completed.stderr.strip()
                    stdout = completed.stdout.strip()
                    detail = stderr or stdout or f"Evaluation exited with status {completed.returncode}."
                    raise RuntimeError(detail)
            final_message = f"Finished evaluating {total} model(s)."
            focus_model = model_names[-1]
        except Exception as exc:  # pragma: no cover - defensive UI path
            final_message = "Evaluation failed. See notification for the error."
            focus_model = model_names[-1]
            self.call_from_thread(
                self.notify,
                _tail_lines(str(exc)),
                title="Evaluation Error",
                severity="error",
            )
        self.call_from_thread(self._finish_run, final_message, focus_model)

    def _set_status_text(self, status_text: str) -> None:
        self.status_text = status_text
        self._update_status()

    def _finish_run(self, status_text: str, focus_model: str) -> None:
        self.status_text = status_text
        self.refresh_rows(focus_model=focus_model)
        self.notify(status_text, title="Evaluation")

    def action_run_selected(self) -> None:
        if self.workers:
            self.notify("An evaluation is already running.", title="Busy", severity="warning")
            return

        highlighted_model = self._current_model_name()
        if self.selected_models:
            model_names = [
                str(row["model_name"])
                for row in self._sorted_rows()
                if str(row["model_name"]) in self.selected_models
            ]
        else:
            model_names = [highlighted_model] if highlighted_model else []
        if not model_names:
            self.notify("No models are available to run.", title="Nothing To Do", severity="warning")
            return

        self.status_text = f"Queued {len(model_names)} model(s) for evaluation."
        self._update_status()
        self._run_models(model_names)

    def action_show_details(self) -> None:
        model_name = self._current_model_name()
        if model_name is None:
            return
        row = next(item for item in self.rows if item["model_name"] == model_name)
        self.push_screen(DetailScreen(self._detail_renderable(row), str(row["model_name"])))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._update_details()


class DetailScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
        Binding("q", "dismiss", "Close", show=False),
        Binding("d", "dismiss", "Close", show=False),
    ]

    def __init__(self, renderable: RenderableType, model_name: str) -> None:
        super().__init__()
        self.renderable = renderable
        self.model_name = model_name

    def compose(self) -> ComposeResult:
        yield VerticalScroll(Static(self.renderable), id="detail-modal")

    def action_dismiss(self) -> None:
        self.dismiss()


def main() -> int:
    EvaluationTUIApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
