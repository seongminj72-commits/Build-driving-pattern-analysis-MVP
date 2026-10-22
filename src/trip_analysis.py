"""CSV 업로드용 분석. 모든 임계값은 프로젝트 시연용이다."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Settings:
    acceleration: float = 2.5
    braking: float = -3.0
    max_gap_seconds: float = 10.0
    radius_m: float = 200.0


def analyze(raw, settings=Settings()):
    if settings.acceleration <= 0 or settings.braking >= 0 or settings.max_gap_seconds <= 0:
        raise ValueError('임계값과 최대 측정 간격을 확인하세요.')
    df = raw.copy()
    if 'speed_kmh' not in df and 'speed' in df:
        df['speed_kmh'] = df['speed']
    missing = {'timestamp', 'speed_kmh'} - set(df.columns)
    if missing:
        raise ValueError('필수 컬럼: timestamp, speed_kmh (기존 speed 허용). 누락: ' + ', '.join(sorted(missing)))
    if 'trip_id' not in df:
        df['trip_id'] = 'uploaded-trip'
    if 'driver_id' not in df:
        df['driver_id'] = 'uploaded-driver'
    keys = ['driver_id', 'trip_id']
    df['timestamp'] = pd.to_datetime(df.timestamp, errors='coerce', utc=True, format='mixed')
    df['speed_kmh'] = pd.to_numeric(df.speed_kmh, errors='coerce')
    invalid = (df[keys].isna().any(axis=1) | df.timestamp.isna() |
               ~np.isfinite(df.speed_kmh) | df.speed_kmh.lt(0))
    report = {'입력 행': len(df), '식별자·시각·속도 오류 제외': int(invalid.sum())}
    # 오류 행 제거 후에도 그 구간을 가로질러 차분하지 않는다.
    df['_invalid'] = invalid
    df = df.sort_values(keys + ['timestamp'], kind='stable')
    df['_barrier'] = df.groupby(keys, dropna=False)['_invalid'].cumsum()
    df = df.loc[~invalid].copy()
    duplicates = df.duplicated(keys + ['timestamp'], keep='first')
    report['동일 주행 중복 시각 제외'] = int(duplicates.sum())
    df = df.loc[~duplicates].reset_index(drop=True)
    if df.empty:
        raise ValueError('유효한 측정값이 없습니다. timestamp와 0 이상 유한한 속도를 확인하세요.')
    for c in keys:
        df[c] = df[c].astype(str)
    for c in ['latitude', 'longitude']:
        df[c] = pd.to_numeric(df[c], errors='coerce') if c in df else np.nan
    gps = df.latitude.between(-90, 90) & df.longitude.between(-180, 180)
    report['GPS 결측·오류 (위치 분석만 제외)'] = int((~gps).sum())
    df.loc[~gps, ['latitude', 'longitude']] = np.nan
    grouped = df.groupby(keys, sort=False)
    dt = grouped.timestamp.diff().dt.total_seconds()
    valid = dt.gt(0) & dt.le(settings.max_gap_seconds) & grouped['_barrier'].diff().eq(0)
    df['interval_seconds'] = dt.where(valid, 0.)
    df['acceleration_mps2'] = (grouped.speed_kmh.diff() / 3.6 / dt).where(valid)
    df['segment'] = (~valid).cumsum()
    report['긴 측정 공백'] = int(dt.gt(settings.max_gap_seconds).sum())
    report['분석 행'] = len(df)
    previous_speed = grouped.speed_kmh.shift()
    df['weighted_speed'] = ((df.speed_kmh + previous_speed) / 2 * df.interval_seconds).fillna(0)
    df['event_type'] = np.select([df.acceleration_mps2.ge(settings.acceleration),
                                 df.acceleration_mps2.le(settings.braking)], ['급가속', '급감속'], default='')
    runs = (df.event_type.ne(df.event_type.shift()) | df.segment.ne(df.segment.shift())).cumsum()
    records = []
    for _, block in df.loc[df.event_type.ne('')].groupby(runs):
        peak = block.loc[block.acceleration_mps2.abs().idxmax()]
        first = block.iloc[0]
        records.append(dict(event_id=f'E{len(records)+1:04}', driver_id=first.driver_id,
                            trip_id=first.trip_id, event_type=first.event_type,
                            start_time=first.timestamp-pd.Timedelta(seconds=first.interval_seconds),
                            end_time=block.timestamp.iloc[-1], duration_seconds=block.interval_seconds.sum(),
                            peak_acceleration_mps2=peak.acceleration_mps2,
                            latitude=peak.latitude, longitude=peak.longitude))
    columns = ['event_id','driver_id','trip_id','event_type','start_time','end_time',
               'duration_seconds','peak_acceleration_mps2','latitude','longitude']
    events = pd.DataFrame(records, columns=columns)
    duration = float(df.interval_seconds.sum())
    summary = dict(duration_seconds=duration,
                   average_speed=float(df.weighted_speed.sum()/duration) if duration else None,
                   max_speed=float(df.speed_kmh.max()),
                   acceleration_count=int(events.event_type.eq('급가속').sum()),
                   braking_count=int(events.event_type.eq('급감속').sum()))
    return df, events, summary, report


def route_segments(df):
    """GPS 누락, 주행 경계, 시간 공백을 넘지 않는 [경도, 위도] 경로."""
    paths, current, last_segment = [], [], None
    for row in df.itertuples():
        valid = pd.notna(row.latitude) and pd.notna(row.longitude)
        if row.segment != last_segment or not valid:
            if len(current) > 1:
                paths.append({'path': current})
            current = []
        if valid:
            current.append([row.longitude, row.latitude])
        last_segment = row.segment
    if len(current) > 1:
        paths.append({'path': current})
    return paths

HOTSPOT_FIELDS = ['hotspot_id','location_name','latitude','longitude','accident_type',
                  'accident_count','reference_year','source_name','source_url','is_demo']
PUBLIC_MAPPING = dict(zip(['사고지역관리번호','사고지역위치명','위도','경도','사고유형구분','사고건수','사고연도'],
                          ['hotspot_id','location_name','latitude','longitude','accident_type','accident_count','reference_year']))


def normalize_hotspots(raw):
    df = raw.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns={k:v for k,v in PUBLIC_MAPPING.items() if v not in df})
    if not {'latitude','longitude'} <= set(df):
        raise ValueError('사고지점 CSV에 latitude/longitude 또는 위도/경도가 필요합니다. WGS84 경위도만 지원합니다.')
    for c in HOTSPOT_FIELDS:
        if c not in df:
            df[c] = pd.NA
    for c in ['latitude','longitude','accident_count','reference_year']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.loc[df.accident_count.lt(0), 'accident_count'] = np.nan
    flags = df.is_demo.astype('string').str.lower()
    if not flags.dropna().isin(['true','false','1','0']).all():
        raise ValueError('is_demo는 true/false 또는 1/0이어야 합니다.')
    df['is_demo'] = flags.map({'true':True,'false':False,'1':True,'0':False}).astype('boolean')
    valid = df.latitude.between(-90,90) & df.longitude.between(-180,180)
    return df.loc[valid, HOTSPOT_FIELDS].reset_index(drop=True), int((~valid).sum())


def nearby_events(events, hotspots, radius_m=200):
    records = []
    if radius_m <= 0:
        raise ValueError('근접 반경은 양수여야 합니다.')
    for e in events.dropna(subset=['latitude','longitude']).itertuples():
        lat1, lat2 = np.radians(e.latitude), np.radians(hotspots.latitude.to_numpy(dtype=float))
        dlat = lat2-lat1
        dlon = np.radians(hotspots.longitude.to_numpy(dtype=float)-e.longitude)
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        distances = 6371000 * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))
        for i in np.flatnonzero(distances <= radius_m):
            record = hotspots.iloc[i].to_dict()
            record.update(event_id=e.event_id, distance_m=round(float(distances[i]),1))
            records.append(record)
    return pd.DataFrame(records, columns=['event_id','distance_m']+HOTSPOT_FIELDS)


def feedback(summary, matches, has_hotspots):
    text = [f"급가속 {summary['acceleration_count']}회, 급감속 {summary['braking_count']}회가 시연 기준을 충족했습니다.",
            '급감속은 보행자·신호 등에 대한 적절한 대응일 수 있습니다. 당시 도로 상황과 함께 확인하세요.',
            '제한속도 데이터가 없어 과속 여부는 판정하지 않습니다.']
    if has_hotspots:
        text.append(f'대표 위치 기준 사고지점 인근 이벤트는 {matches.event_id.nunique()}회입니다. 근접성은 사고 원인이나 운전자 과실을 의미하지 않습니다.')
    else:
        text.append('사고지점 데이터가 없어 지역 연계 결과는 계산하지 않았습니다.')
    return text
