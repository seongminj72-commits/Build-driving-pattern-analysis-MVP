"""자동차 AX 기반 지능형 운전 패턴 분석 Streamlit 앱."""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess import load_and_preprocess_data
from src.event_detector import detect_driving_events
from src.feature_engineering import build_trip_features
from src.clustering import fit_best_kmeans, describe_cluster
from src.visualization import plot_time_series, plot_acceleration_events, plot_cluster_feature_means, plot_silhouette_scores

st.set_page_config(page_title="Intelligent Driving Pattern Analysis", page_icon="🚗", layout="wide")
st.title("🚗 Intelligent Driving Pattern Analysis")
st.caption("차량 주행 데이터의 특성을 살펴보고 유사한 주행 패턴을 군집으로 비교합니다.")
raw_path = ROOT / "data/raw/driving_data.csv"
if not raw_path.exists():
    st.info("데이터가 없습니다. 프로젝트 루트에서 `python generate_sample_data.py`와 `python run_pipeline.py`를 실행하세요.")
    st.stop()
try:
    clean = load_and_preprocess_data(raw_path, verbose=False)
    events = detect_driving_events(clean)
    features = build_trip_features(events)
    clustered, summary, scores, model, scaler = fit_best_kmeans(features, ROOT / "models")
except (ValueError, FileNotFoundError, KeyError) as exc:
    st.error(f"데이터 분석을 준비하지 못했습니다: {exc}")
    st.stop()

st.sidebar.header("주행 선택")
driver_ids = sorted(events["driver_id"].astype(str).unique())
driver_id = st.sidebar.selectbox("운전자", driver_ids)
driver_trips = sorted(events.loc[events.driver_id.astype(str) == driver_id, "trip_id"].astype(str).unique())
trip_id = st.sidebar.selectbox("주행 기록", driver_trips)
selected = events[(events.driver_id.astype(str) == driver_id) & (events.trip_id.astype(str) == trip_id)].copy()
feature_row = clustered[(clustered.driver_id.astype(str) == driver_id) & (clustered.trip_id.astype(str) == trip_id)]
if selected.empty or feature_row.empty:
    st.warning("선택한 주행 기록을 찾을 수 없습니다.")
    st.stop()
f = feature_row.iloc[0]

st.subheader(f"주행 요약 · {driver_id} / {trip_id}")
kpis = [("평균 속도", f"{f.average_speed:.1f} km/h"), ("최고 속도", f"{f.max_speed:.1f} km/h"),
        ("평균 RPM", f"{f.average_rpm:.0f}"), ("급가속", f"{int(f.hard_acceleration_count)}회"),
        ("급감속", f"{int(f.hard_braking_count)}회"), ("과속", f"{int(f.overspeed_count)}회")]
cols = st.columns(6)
for col, (label, value) in zip(cols, kpis): col.metric(label, value)
st.caption(f"주행 시간: {f.driving_time / 60:.1f}분 · 기록 포인트: {int(f.point_count):,}개")

left, right = st.columns(2)
with left:
    st.subheader("Speed Analysis")
    st.pyplot(plot_time_series(selected, "speed", "Speed over time", "km/h"), clear_figure=True)
with right:
    st.subheader("Acceleration Analysis")
    st.pyplot(plot_acceleration_events(selected), clear_figure=True)

st.subheader("Driving Events")
event_cols = st.columns(3)
for col, (label, field) in zip(event_cols, [("급가속", "hard_acceleration"), ("급감속", "hard_braking"), ("과속", "overspeed")]):
    col.metric(label, f"{int(selected[field].sum())}회")

st.subheader("AI Driving Pattern Analysis")
cluster_id = int(f.cluster)
st.write(f"**Driving Pattern Cluster: {cluster_id}**")
st.write(describe_cluster(cluster_id, summary))
st.caption("KMeans 군집 번호와 이 설명은 통계적 유사성의 요약이며 안전성/위험성 판정이 아닙니다.")
with st.expander("군집별 Feature 평균 및 k 탐색 점수"):
    st.pyplot(plot_cluster_feature_means(summary), clear_figure=True)
    if scores: st.pyplot(plot_silhouette_scores(scores), clear_figure=True)
    st.dataframe(summary.round(3), use_container_width=True)

st.subheader("GPS 주행 위치")
gps = selected.dropna(subset=["latitude", "longitude"])
if not gps.empty:
    st.map(gps[["latitude", "longitude"]].rename(columns={"latitude": "lat", "longitude": "lon"}), use_container_width=True)
else:
    st.info("선택된 주행에는 표시할 GPS 위치 정보가 없습니다.")
