"""교육용 예시 기준을 이용한 주행 이벤트 탐지."""
import pandas as pd

# 교육용 분석 예시 임계값이며 실제 법규/교통안전 기준이 아니다.
HARD_ACCELERATION_THRESHOLD = 2.5  # m/s²
HARD_BRAKING_THRESHOLD = -3.0      # m/s²
OVERSPEED_THRESHOLD = 100.0        # km/h (예시 기준)


def detect_driving_events(df: pd.DataFrame) -> pd.DataFrame:
    """급가속, 급감속, 과속 여부를 행별 Boolean으로 추가한다."""
    result = df.copy()
    result["hard_acceleration"] = result["acceleration"].ge(HARD_ACCELERATION_THRESHOLD).fillna(False)
    result["hard_braking"] = result["acceleration"].le(HARD_BRAKING_THRESHOLD).fillna(False)
    result["overspeed"] = result["speed"].ge(OVERSPEED_THRESHOLD).fillna(False)
    return result
