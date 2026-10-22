"""용인시 운전 습관 분석 대시보드."""
from pathlib import Path
import sys
import io
import pandas as pd
import streamlit as st
import pydeck as pdk
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.trip_analysis import Settings, analyze, normalize_hotspots, nearby_events, route_segments, feedback


def read_csv(upload):
    payload = upload.getvalue()
    try:
        return pd.read_csv(io.BytesIO(payload), encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(payload), encoding='cp949')


def main():
    st.set_page_config(page_title='용인 운전 습관 분석', page_icon='🚗', layout='wide')
    st.title('🚗 용인시 운전 습관 분석')
    st.caption('통학생과 주민을 위한 주행 기록 분석 · 속도 변화와 주의구간을 함께 확인하세요.')
    st.info('실제 용인시 사고지점 원본은 아직 확보되지 않았습니다. 기본 예제는 합성 주행이며, 지역 사고 현황을 나타내지 않습니다.')
    upload = st.file_uploader('주행 CSV 업로드', type=['csv'])
    use_sample = st.checkbox('합성 주행 예제로 살펴보기', value=upload is None)
    sample_name = st.selectbox('합성 예제', ['수지·죽전 연결 시연', '기존 군집 분석 예제']) if use_sample and not upload else '수지·죽전 연결 시연'
    sample_path = ROOT / 'data/raw' / ('yongin_demo_trip.csv' if sample_name == '수지·죽전 연결 시연' else 'driving_data.csv')
    with st.sidebar:
        st.header('분석 설정')
        st.caption('프로젝트 시연용 기준 · 법적·공인 안전 기준이 아닙니다.')
        accel = st.number_input('급가속 기준 (m/s²)', min_value=0.1, value=2.5)
        brake = st.number_input('급감속 기준 (m/s²)', max_value=-0.1, value=-3.0)
        gap = st.number_input('최대 측정 간격 (초)', min_value=0.1, value=10.)
        radius = st.number_input('사고지점 근접 반경 (m)', min_value=1., value=200.)
    hotspot_upload = st.file_uploader('사고다발지역 CSV (선택 · WGS84 경위도)', type=['csv'])
    demo = st.checkbox('가상 사고지점으로 연결 기능 시연 (실제 사고 자료 아님)', value=False)
    try:
        raw = read_csv(upload) if upload else pd.read_csv(sample_path) if use_sample else None
        if raw is None:
            st.info('timestamp와 speed_kmh(또는 speed)가 있는 CSV를 업로드하세요. GPS는 선택입니다.')
            return
        points, events, _, report = analyze(raw, Settings(accel, brake, gap, radius))
        st.subheader('데이터 점검')
        st.write(report)
        choices = list(points[['driver_id','trip_id']].drop_duplicates().itertuples(index=False, name=None))
        driver, trip = st.selectbox('분석할 주행', choices, format_func=lambda v: f'{v[0]} / {v[1]}')
        points = points[(points.driver_id == driver) & (points.trip_id == trip)].copy()
        # 선택된 주행의 계산된 구간을 유지한다.
        events = events[(events.driver_id == driver) & (events.trip_id == trip)].copy()
        duration = float(points.interval_seconds.sum())
        summary = dict(duration_seconds=duration, average_speed=float(points.weighted_speed.sum()/duration) if duration else None,
                       max_speed=float(points.speed_kmh.max()), acceleration_count=int(events.event_type.eq('급가속').sum()),
                       braking_count=int(events.event_type.eq('급감속').sum()))
        hotspots = pd.DataFrame(columns=['latitude','longitude'])
        if hotspot_upload or demo:
            source = read_csv(hotspot_upload) if hotspot_upload else pd.read_csv(ROOT/'data/regional/demo_hotspots.csv')
            hotspots, excluded = normalize_hotspots(source)
            st.caption(f'사고지점 유효 {len(hotspots)}개 · 좌표 오류 제외 {excluded}개')
            st.dataframe(hotspots, hide_index=True)
            if hotspots.is_demo.fillna(True).any():
                st.warning('가상 또는 출처 상태 미확인 지점이 포함되어 있습니다. 실제 용인시 사고 현황으로 해석하지 마세요.')
        matches = nearby_events(events, hotspots, radius)
        cols = st.columns(5)
        values = [('유효 주행 시간', f'{duration:.1f}초'),
                  ('시간 가중 평균 속도', '계산 불가' if summary['average_speed'] is None else f"{summary['average_speed']:.1f} km/h"),
                  ('최고 속도',f"{summary['max_speed']:.1f} km/h"),('급가속',f"{summary['acceleration_count']}회"),
                  ('급감속',f"{summary['braking_count']}회")]
        for col, (label, value) in zip(cols,values):
            col.metric(label,value)
        st.caption('평균 속도는 유효 인접 구간의 양 끝 속도 평균 × 시간의 합 / 유효 시간 합입니다. 긴 공백은 제외합니다.')
        st.subheader('속도 변화')
        # 각 시간 구간을 별도 계열로 표시하여 공백을 연결하지 않는다.
        chart = points.pivot(index='timestamp', columns='segment', values='speed_kmh')
        chart.columns = [f'구간 {c}' for c in chart.columns]
        st.line_chart(chart)
        st.subheader('이벤트 상세')
        st.caption('연속 초과 구간을 1회로 집계합니다. 시작은 첫 초과 구간의 이전 측정 시각, 대표 위치는 최대 절대 가속도 행의 좌표입니다. 해당 좌표가 없으면 위치를 추정하지 않습니다.')
        st.dataframe(events, hide_index=True)
        st.download_button('이벤트 CSV 다운로드', events.to_csv(index=False).encode('utf-8-sig'), 'driving_events.csv','text/csv')
        st.subheader('경로와 주의구간')
        gps = points.dropna(subset=['latitude','longitude'])
        layers = [pdk.Layer('PathLayer', route_segments(points), get_path='path', get_color=[40,130,230], width_min_pixels=4),
                  pdk.Layer('ScatterplotLayer',gps,get_position='[longitude, latitude]',get_radius=8,get_fill_color=[40,130,230])]
        if not hotspots.empty:
            layers.append(pdk.Layer('ScatterplotLayer',hotspots,get_position='[longitude, latitude]',get_radius=radius,
                                    get_fill_color=[240,160,30,65],pickable=True))
        located = events.dropna(subset=['latitude','longitude'])
        layers.append(pdk.Layer('ScatterplotLayer',located,get_position='[longitude, latitude]',get_radius=25,
                                get_fill_color=[220,50,60],pickable=True))
        if not gps.empty:
            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=float(gps.latitude.mean()),
                             longitude=float(gps.longitude.mean()),zoom=13), tooltip={'text':'{event_type} {location_name}'}))
            st.caption('파랑: 주행 · 빨강: 이벤트 대표 위치 · 주황: 업로드/가상 사고지점 근접 반경. GPS 결측·긴 공백은 연결하지 않습니다.')
        else:
            st.info('유효한 GPS가 없어 위치 분석을 생략합니다. 속도·이벤트 분석은 가능합니다.')
        st.subheader('사고지점 인근 이벤트')
        st.dataframe(matches, hide_index=True)
        st.caption('대표 위치와 지점 중심 간 Haversine 직선거리입니다. 도로 연결 관계·실제 사고 영역·이벤트 전체 궤적은 반영하지 않습니다.')
        st.subheader('계산 근거에 따른 피드백')
        for sentence in feedback(summary,matches,not hotspots.empty):
            st.write('• '+sentence)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        st.error(f'CSV를 확인하세요: {exc}')


if __name__ == '__main__':
    main()
