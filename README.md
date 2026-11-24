# Macro Quant Analyzer — 무형자산·거시 GDP 괴리 및 퀀트 포트폴리오 검증 저장소

경제 블로그 **혜안** 포스팅의 정량 근거를 재현하고 검증하기 위한 파이프라인입니다.

- **분석 기간**: `2006-01-01` ~ `2026-11-23` (20년 패널, 10년 회귀 검정 및 3년 누적 전환율 스크리닝 내장)
- **핵심 검증 항목**:
  1. **거시 GDP 괴리 (`Delta Y_Gap`)**: 무형자본 축적(`dK_u`)과 비관측 TFP(`dTFP_unmeasured`)
  2. **회계 비용 처리 왜곡 & H0 검정**: R&D/SG&A 내 무형투자 자본화 후 `Adjusted PBR`, `Adjusted ROIC`, `OCF` 비교
  3. **수급 및 무형투자 비중 55% 임계치**: 55% 돌파 구간과 펀드 플로우 상관성
  4. **진부화(Cliff Effect) & 테일 리스크**: 손상차손(`impairment_loss`) 및 변동성 P95 비교
  5. **퀀트 스크리닝 & 한계기업 배제**: 3년 전환율(`>= 1.5x`), `FCF Margin`, `Sharpe Ratio` 및 **차입 증가 + GPM 정체 기업 배제**
  6. **시계열 무결성**: US/KR 거래일·시점 정렬, 빈 데이터/휴장일 안전 처리, 시그널 지연(`lag`) 테스트

## 실행
```bash
pip install -r requirements.txt
pytest -q
python -m src.run_all --refresh-cache
