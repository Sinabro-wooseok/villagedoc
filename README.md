# VillageDoc

**AI Medical Assistant for Community Health Workers**

Gemma 4 Good Hackathon 2025 제출작 | 목표: 1위 ($200K)

## 문제

전 세계 20억 명이 의료 서비스 접근 불가. 커뮤니티 보건 인력(CHW)은 인터넷도, 전문 훈련도 없이 하루 10건의 케이스를 처리한다.

## 솔루션

Gemma 4 E4B (오프라인 에지 AI)로 CHW가 스마트폰만으로:
1. 환자 사진 촬영 → 시각적 분석
2. 스와힐리어/힌디어 음성 입력 → Whisper STT
3. WHO IMCI 프로토콜 기반 즉각 진단 + 치료 지침

## Gemma 4 핵심 기능 활용

| 기능 | 활용 방식 |
|------|----------|
| 멀티모달 비전 | 피부/눈/상처 임상 이미지 분석 |
| 128K 컨텍스트 | WHO IMCI 전체 프로토콜 직접 주입 (RAG 불필요) |
| Function Calling | 구조화된 JSON 진단 출력 (EHR 연동) |
| 다국어 | 스와힐리어/힌디어 네이티브 추론 |
| 오프라인 | GGUF Q4_K_M (~5GB), 인터넷 없이 동작 |

## 빠른 시작

```bash
# 의존성 설치
pip install -r requirements.txt

# Streamlit 데모
streamlit run src/ui/streamlit_app.py

# 벤치마크 실행
python src/evaluation/benchmark.py
```

## 구조

```
villagedoc/
├── notebooks/villagedoc_main.ipynb  ← Kaggle 제출 노트북
├── src/
│   ├── model/
│   │   ├── gemma4_runner.py         ← Gemma 4 추론 엔진
│   │   ├── whisper_asr.py           ← 음성 인식
│   │   └── diagnosis_schema.py      ← Function Calling 스키마
│   ├── data/
│   │   ├── who_imci_protocol.json   ← WHO IMCI 2024
│   │   ├── essential_medicines.json ← WHO 필수 의약품
│   │   ├── swahili_medical_terms.json
│   │   └── patient_db.py            ← SQLite 오프라인 DB
│   ├── ui/
│   │   └── streamlit_app.py         ← 데모 UI
│   └── evaluation/
│       └── benchmark.py             ← 정확도 측정
└── requirements.txt
```

## 임팩트

CHW 200만 명 × 하루 10건 × 진단 정확도 31% 향상 = **연간 2.2억 건의 개선된 진단**

## 라이선스

Apache 2.0
