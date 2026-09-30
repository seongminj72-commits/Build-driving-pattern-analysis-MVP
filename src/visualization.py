"""Streamlit 및 파일 저장에서 재사용하는 matplotlib 시각화 함수."""
import matplotlib.pyplot as plt
import pandas as pd


def _new_figure():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_time_series(df: pd.DataFrame, column: str, title: str, ylabel: str):
    fig, ax = _new_figure()
    ax.plot(df["timestamp"], df[column], linewidth=1.2)
    ax.set(title=title, xlabel="Time", ylabel=ylabel)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_acceleration_events(df: pd.DataFrame):
    fig, ax = _new_figure()
    ax.plot(df["timestamp"], df["acceleration"], label="Acceleration", color="#2563eb")
    for flag, label, color in [("hard_acceleration", "Hard acceleration", "#dc2626"), ("hard_braking", "Hard braking", "#f59e0b")]:
        points = df[df[flag]]
        ax.scatter(points["timestamp"], points["acceleration"], label=label, color=color, s=24)
    ax.set(title="Acceleration and events", xlabel="Time", ylabel="m/s²")
    ax.legend()
    fig.autofmt_xdate(); fig.tight_layout()
    return fig


def plot_driver_metric(features: pd.DataFrame, metric: str, title: str):
    aggregated = features.groupby("driver_id")[metric].mean().sort_values()
    fig, ax = _new_figure()
    aggregated.plot(kind="bar", ax=ax, color="#0f766e")
    ax.set(title=title, xlabel="Driver", ylabel=metric)
    fig.tight_layout()
    return fig


def plot_cluster_feature_means(summary: pd.DataFrame, columns: list[str] | None = None):
    columns = columns or ["average_speed", "speed_std", "hard_acceleration_count", "hard_braking_count"]
    available = [column for column in columns if column in summary.columns]
    fig, ax = _new_figure()
    if available:
        normalized = summary.set_index("cluster")[available]
        normalized = (normalized - normalized.mean()) / normalized.std().replace(0, 1)
        normalized.plot(kind="bar", ax=ax)
    ax.set(title="Cluster feature means (standardized for display)", xlabel="Cluster", ylabel="Relative mean")
    fig.tight_layout()
    return fig


def plot_silhouette_scores(scores: dict[int, float]):
    fig, ax = _new_figure()
    if scores:
        ax.plot(list(scores), list(scores.values()), marker="o")
    ax.set(title="Silhouette score by k", xlabel="k", ylabel="Silhouette score")
    fig.tight_layout()
    return fig


def save_overview_figures(events: pd.DataFrame, features: pd.DataFrame, summary: pd.DataFrame,
                          scores: dict[int, float], output_dir: str = "results/figures") -> None:
    from pathlib import Path
    folder = Path(output_dir); folder.mkdir(parents=True, exist_ok=True)
    plots = [(plot_driver_metric(features, "average_speed", "Average speed by driver"), "average_speed_by_driver.png"),
             (plot_driver_metric(features, "hard_acceleration_count", "Hard acceleration events by driver"), "hard_acceleration_by_driver.png"),
             (plot_driver_metric(features, "hard_braking_count", "Hard braking events by driver"), "hard_braking_by_driver.png"),
             (plot_cluster_feature_means(summary), "cluster_feature_means.png"),
             (plot_silhouette_scores(scores), "silhouette_scores.png")]
    # 첫 주행의 시계열을 대표 예시로 저장한다.
    first = events.iloc[0:0]
    if not events.empty:
        first = events[(events.driver_id == events.iloc[0].driver_id) & (events.trip_id == events.iloc[0].trip_id)]
        plots += [(plot_time_series(first, "speed", "Speed over time", "km/h"), "speed_over_time.png"),
                  (plot_acceleration_events(first), "acceleration_over_time.png")]
    for fig, filename in plots:
        fig.savefig(folder / filename, dpi=130)
        plt.close(fig)
