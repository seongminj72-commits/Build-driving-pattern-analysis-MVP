"""StandardScaler와 KMeans를 사용한 비지도 군집화."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ML_FEATURES = [
    "average_speed", "max_speed", "speed_std", "average_acceleration", "max_acceleration",
    "min_acceleration", "hard_acceleration_count", "hard_braking_count", "overspeed_count",
    "average_rpm", "rpm_std", "driving_time", "hard_acceleration_ratio", "hard_braking_ratio", "overspeed_ratio",
]


def fit_best_kmeans(features: pd.DataFrame, model_dir: str | Path = "models") -> tuple[pd.DataFrame, pd.DataFrame, dict[int, float], KMeans | None, StandardScaler | None]:
    """실루엣 점수를 비교해 KMeans를 학습하고 결과/모델을 반환한다."""
    if features.empty:
        raise ValueError("군집화할 Feature 데이터가 없습니다.")
    missing = sorted(set(ML_FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"군집화 Feature 컬럼이 없습니다: {', '.join(missing)}")
    result = features.copy()
    matrix = result[ML_FEATURES].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median()).fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    n = len(result)
    scores: dict[int, float] = {}
    candidates = [k for k in range(2, min(6, n - 1) + 1) if k < n]
    best_model = None
    best_k = None
    best_score = -1.0
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        if len(set(labels)) < 2 or len(set(labels)) >= n:
            continue
        try:
            score = float(silhouette_score(scaled, labels))
        except ValueError:
            continue
        scores[k] = score
        if score > best_score:
            best_score, best_k, best_model = score, k, model
    if best_model is None:
        print("군집 샘플이 부족하거나 실루엣 점수를 계산할 수 없어 단일 군집으로 처리합니다.")
        result["cluster"] = 0
        summary = result.groupby("cluster")[ML_FEATURES].mean().reset_index()
        return result, summary, scores, None, scaler
    result["cluster"] = best_model.labels_
    summary = result.groupby("cluster")[ML_FEATURES].mean().reset_index()
    folder = Path(model_dir)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, folder / "kmeans_model.pkl")
    joblib.dump(scaler, folder / "scaler.pkl")
    print(f"최적 cluster 수: k={best_k}, silhouette={best_score:.4f}")
    return result, summary, scores, best_model, scaler


def describe_cluster(cluster_id: int, summary: pd.DataFrame) -> str:
    """군집 평균값을 전체 군집 평균과 비교해 중립적인 설명을 만든다."""
    if summary.empty or "cluster" not in summary:
        return "군집 통계를 확인할 수 없습니다."
    row = summary.loc[summary["cluster"] == cluster_id]
    if row.empty:
        return "선택한 주행의 군집 통계를 찾을 수 없습니다."
    row = row.iloc[0]
    overall = summary.drop(columns="cluster").mean(numeric_only=True)
    traits = []
    if row.get("speed_std", 0) > overall.get("speed_std", 0): traits.append("속도 변동이 상대적으로 큽니다")
    else: traits.append("속도 변동이 상대적으로 작거나 평균 수준입니다")
    event_total = sum(float(row.get(c, 0)) for c in ["hard_acceleration_ratio", "hard_braking_ratio"])
    baseline = sum(float(overall.get(c, 0)) for c in ["hard_acceleration_ratio", "hard_braking_ratio"])
    if event_total > baseline: traits.append("급격한 가감속 이벤트 비율이 상대적으로 높게 관찰됩니다")
    else: traits.append("급격한 가감속 이벤트 비율이 평균 수준이거나 낮게 관찰됩니다")
    return "이 그룹에서는 " + "; ".join(traits) + ". 군집 번호는 안전성 또는 위험 등급을 뜻하지 않습니다."
