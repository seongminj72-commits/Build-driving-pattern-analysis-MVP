"""주행 CSV 로드와 기본 정제 함수."""
from pathlib import Path
from typing import Any
import pandas as pd

REQUIRED_COLUMNS = [
    "driver_id", "trip_id", "timestamp", "speed", "acceleration", "brake",
    "rpm", "steering_angle", "latitude", "longitude",
]
NUMERIC_COLUMNS = ["speed", "acceleration", "brake", "rpm", "steering_angle", "latitude", "longitude"]


def load_and_preprocess_data(csv_path: str | Path, verbose: bool = True) -> pd.DataFrame:
    """CSV를 복사해 정제하고 driver/trip/time 순서로 반환한다."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"원본 CSV 파일을 찾을 수 없습니다: {path}. 먼저 python generate_sample_data.py를 실행하세요.")
    df = pd.read_csv(path).copy()
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")
    initial = len(df)
    df = df.drop_duplicates().copy()
    duplicated_removed = initial - len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 식별자/시간/핵심 측정값이 없는 행은 분석할 수 없어 제거한다.
    essential = ["driver_id", "trip_id", "timestamp", "speed", "acceleration", "rpm"]
    before = len(df)
    df = df.dropna(subset=essential).copy()
    essential_removed = before - len(df)
    # 물리적으로 말이 되지 않는 속도와 RPM은 결측 처리 후 주행별 선형 보정한다.
    invalid = (df["speed"] < 0) | (df["speed"] > 300) | (df["rpm"] < 0) | (df["rpm"] > 20000)
    invalid_count = int(invalid.sum())
    df.loc[invalid, ["speed", "rpm"]] = float("nan")
    df = df.sort_values(["driver_id", "trip_id", "timestamp"]).reset_index(drop=True)
    for col in ["speed", "acceleration", "rpm", "steering_angle"]:
        df[col] = df.groupby(["driver_id", "trip_id"], sort=False)[col].transform(
            lambda values: values.interpolate(limit_direction="both")
        )
    # GPS 결측은 위치 시각화에서 제외하며, brake 결측은 미입력으로 처리한다.
    df["brake"] = df["brake"].fillna(0).clip(0, 1).round().astype(int)
    if verbose:
        print(f"전처리: 입력 {initial:,}행, 중복 제거 {duplicated_removed:,}행, 필수값 결측 제거 {essential_removed:,}행, 이상 속도/RPM 보정 {invalid_count:,}건, 최종 {len(df):,}행")
    if df.empty:
        raise ValueError("전처리 후 데이터가 비었습니다. CSV의 필수 컬럼과 측정값을 확인하세요.")
    return df
