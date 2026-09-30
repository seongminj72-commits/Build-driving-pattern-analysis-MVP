"""운전자/주행 단위 요약 Feature 생성."""
import pandas as pd


def build_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """측정 행을 driver_id/trip_id 단위의 ML Feature로 집계한다."""
    if df.empty:
        raise ValueError("Feature를 만들 주행 데이터가 없습니다.")
    grouped = df.groupby(["driver_id", "trip_id"], sort=True)
    features = grouped.agg(
        average_speed=("speed", "mean"), max_speed=("speed", "max"), speed_std=("speed", "std"),
        average_acceleration=("acceleration", "mean"), max_acceleration=("acceleration", "max"),
        min_acceleration=("acceleration", "min"), hard_acceleration_count=("hard_acceleration", "sum"),
        hard_braking_count=("hard_braking", "sum"), overspeed_count=("overspeed", "sum"),
        average_rpm=("rpm", "mean"), rpm_std=("rpm", "std"), point_count=("timestamp", "count"),
        start_time=("timestamp", "min"), end_time=("timestamp", "max"),
    ).reset_index()
    durations = (features["end_time"] - features["start_time"]).dt.total_seconds().clip(lower=0)
    features["driving_time"] = durations
    denominator = features["point_count"].replace(0, 1)
    for count_col, ratio_col in [
        ("hard_acceleration_count", "hard_acceleration_ratio"),
        ("hard_braking_count", "hard_braking_ratio"), ("overspeed_count", "overspeed_ratio"),
    ]:
        features[ratio_col] = features[count_col] / denominator
    features["speed_std"] = features["speed_std"].fillna(0)
    features["rpm_std"] = features["rpm_std"].fillna(0)
    return features
