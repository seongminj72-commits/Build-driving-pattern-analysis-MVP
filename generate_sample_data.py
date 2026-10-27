"""재현 가능한 교육용 가상 차량 주행 데이터를 생성한다."""
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).parent / "data" / "raw" / "driving_data.csv"
SEED = 42
VALIDATION_COLUMNS = ['driver_id', 'trip_id', 'timestamp', 'speed_kmh']


def _fixture_frame(trip_id: str, seconds: list[int], speeds: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        'driver_id': ['SYNTH001'] * len(seconds),
        'trip_id': [trip_id] * len(seconds),
        'timestamp': [f'2026-01-01T00:00:{second:02d}Z' for second in seconds],
        'speed_kmh': speeds,
    }, columns=VALIDATION_COLUMNS)


def generate_validation_fixtures(output_dir: str | Path = Path(__file__).parent / 'tests' / 'fixtures' / 'trips') -> dict[str, pd.DataFrame]:
    """Write deterministic time/speed fixtures for analysis tests.

    These fixtures validate demonstration calculations; they are not real
    vehicle or road measurements.
    """
    fixtures = {
        'constant_speed': _fixture_frame('constant_speed', [0, 1, 3, 6], [36.0, 36.0, 36.0, 36.0]),
        'constant_acceleration': _fixture_frame('constant_acceleration', [0, 1, 3, 6], [0.0, 10.8, 32.4, 64.8]),
        'stop_time_gap': _fixture_frame('stop_time_gap', [0, 1, 2, 4, 15, 16, 18], [28.8, 14.4, 0.0, 0.0, 118.8, 118.8, 118.8]),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in fixtures.items():
        frame.to_csv(directory / f'{name}.csv', index=False, lineterminator='\n')
    return fixtures


def generate_sample_data(output_path: str | Path = OUTPUT_PATH, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    # 각 운전자의 생성 파라미터는 데이터 생성에만 사용하며 군집 모델에 정답으로 제공하지 않는다.
    profiles = [("D001", 48, 7, 0.45), ("D002", 64, 15, 1.1), ("D003", 56, 23, 1.8),
                ("D004", 42, 10, 1.35), ("D005", 72, 19, 0.8), ("D006", 52, 6, 0.3)]
    base = pd.Timestamp("2026-01-01 08:00:00")
    for driver_index, (driver, mean_speed, speed_noise, event_chance) in enumerate(profiles):
        for trip_number in range(1, 5):
            trip = f"T{trip_number:03d}"
            n = 90
            time_offset = (driver_index * 4 + trip_number - 1) * 7200
            timestamps = base + pd.to_timedelta(time_offset + np.arange(n) * 5, unit="s")
            speed = np.clip(mean_speed + rng.normal(0, speed_noise, n) + 5 * np.sin(np.arange(n) / 9), 0, 145)
            acceleration = np.diff(speed, prepend=speed[0]) / 3.6 / 5 + rng.normal(0, 0.28, n)
            event_mask = rng.random(n) < event_chance * 0.035
            acceleration[event_mask] += rng.choice([-4.2, 3.1], event_mask.sum())
            speed = np.clip(speed + np.cumsum(acceleration * 0.12), 0, 150)
            rpm = np.clip(850 + speed * 34 + acceleration * 220 + rng.normal(0, 180, n), 700, 6500)
            lat = 37.50 + driver_index * 0.005 + np.cumsum(rng.normal(0, 0.00008, n))
            lon = 127.02 + trip_number * 0.003 + np.cumsum(rng.normal(0, 0.00009, n))
            for i in range(n):
                rows.append({"driver_id": driver, "trip_id": trip, "timestamp": timestamps[i],
                             "speed": round(float(speed[i]), 2), "acceleration": round(float(acceleration[i]), 3),
                             "brake": int(acceleration[i] < -1.1), "rpm": round(float(rpm[i]), 1),
                             "steering_angle": round(float(rng.normal(0, 7)), 2),
                             "latitude": round(float(lat[i]), 6), "longitude": round(float(lon[i]), 6)})
    df = pd.DataFrame(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"가상 주행 데이터 생성 완료: {path} ({len(df):,}행, seed={seed})")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-fixtures', action='store_true',
                        help='분석 계약 검증용 결정적 합성 CSV 3개를 생성합니다.')
    args = parser.parse_args()
    if args.validation_fixtures:
        generate_validation_fixtures()
    else:
        generate_sample_data()
