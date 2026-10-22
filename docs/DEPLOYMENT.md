# Streamlit Community Cloud 배포

대상 저장소: https://github.com/seongminj72-commits/Build-driving-pattern-analysis-MVP

## 앱 설정

- Repository: `seongminj72-commits/Build-driving-pattern-analysis-MVP`
- Branch: `main`
- Main file path: `dashboard/app.py`
- Python: `3.12` 권장
- Secrets: 필요 없음 (외부 사고 데이터 API 연동 없음)

https://share.streamlit.io/ 에서 로그인하고 Create app으로 위 설정을 입력한다.
서비스 약관 동의나 GitHub 접근 권한 승인이 새로 필요하면 계정 소유자가 확인해야 한다.
저장소는 공개이며 배포 앱도 공개 시연 용도로 사용한다. 업로드 CSV는 앱 메모리에서만 처리한다.

## 배포 후 점검

1. 기본 합성 예제에서 요약 지표와 지도를 확인한다.
2. 가상 사고지점 시연을 켜면 근접 이벤트가 2회인지 확인한다.
3. GPS 없는 CSV에서 기본 분석이 되는지 확인한다.
4. 실제 용인시 사고자료 미확보 안내와 가상 데이터 표시를 확인한다.

배포를 생성한 뒤에는 실제 주소를 열어 앱 실행을 확인해야 완료로 간주한다.
main 업데이트는 Community Cloud의 연결된 앱에서 자동으로 반영된다.
