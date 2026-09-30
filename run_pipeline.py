"""전체 데이터 분석 파이프라인 실행 진입점."""
from pathlib import Path
from src.preprocess import load_and_preprocess_data
from src.event_detector import detect_driving_events
from src.feature_engineering import build_trip_features
from src.clustering import fit_best_kmeans
from src.visualization import save_overview_figures

ROOT = Path(__file__).resolve().parent


def main() -> None:
    raw_path = ROOT / "data/raw/driving_data.csv"
    processed_dir = ROOT / "data/processed"
    results_dir = ROOT / "results"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print("[1/7] CSV 로드 및 전처리")
    clean = load_and_preprocess_data(raw_path)
    print("[2/7] 주행 이벤트 탐지")
    events = detect_driving_events(clean)
    print("[3/7] 주행별 Feature 생성")
    features = build_trip_features(events)
    feature_path = processed_dir / "driving_features.csv"
    features.to_csv(feature_path, index=False)
    print(f"Feature 저장: {feature_path}")
    print("[4/7] StandardScaler + KMeans + silhouette 탐색")
    clustered, summary, scores, _, _ = fit_best_kmeans(features, ROOT / "models")
    output_columns = ["driver_id", "trip_id", "average_speed", "max_speed", "hard_acceleration_count",
                      "hard_braking_count", "overspeed_count", "cluster"]
    clustered.to_csv(results_dir / "cluster_result.csv", index=False)
    summary.to_csv(results_dir / "cluster_summary.csv", index=False)
    events.to_csv(processed_dir / "driving_events.csv", index=False)
    print("[5/7] 결과 저장")
    print(f"군집 결과: {results_dir / 'cluster_result.csv'} (포함 컬럼: {', '.join(output_columns)})")
    print("[6/7] 시각화 저장")
    save_overview_figures(events, clustered, summary, scores, str(results_dir / "figures"))
    print("[7/7] 파이프라인 완료")


if __name__ == "__main__":
    main()
