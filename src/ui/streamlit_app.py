"""
VillageDoc - Streamlit 데모 UI
오프라인 커뮤니티 보건 인력(CHW) 의료 보조 시스템
"""

import streamlit as st
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gemma4_runner import MockGemma4Runner

# 실제 모델 여부에 따라 전환
try:
    from model.gemma4_runner import Gemma4Runner
    runner = Gemma4Runner()
except Exception:
    runner = MockGemma4Runner()

st.set_page_config(
    page_title="VillageDoc - AI 의료 보조",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일
st.markdown("""
<style>
.risk-high { background: #e74c3c; color: white; padding: 12px; border-radius: 8px; }
.risk-medium { background: #f39c12; color: white; padding: 12px; border-radius: 8px; }
.risk-low { background: #27ae60; color: white; padding: 12px; border-radius: 8px; }
.diagnosis-card { background: #f8f9fa; padding: 16px; border-radius: 8px; margin: 8px 0; }
.red-flag { color: #e74c3c; font-weight: bold; }
.med-available { color: #27ae60; }
.med-unavailable { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.title("VillageDoc")
    st.caption("커뮤니티 보건 인력 AI 보조")
    st.divider()

    language = st.selectbox(
        "언어 선택",
        options=["sw", "hi", "fr", "en"],
        format_func=lambda x: {
            "sw": "Swahili (스와힐리어)",
            "hi": "Hindi (힌디어)",
            "fr": "Français (프랑스어)",
            "en": "English (영어)"
        }[x]
    )

    st.divider()
    st.caption("모델: Gemma 4 E4B-IT")
    st.caption("지식: WHO IMCI 2024")
    st.caption("상태: 오프라인 동작")

    if st.button("데모: 말라리아 케이스"):
        st.session_state["demo_symptoms"] = "mtoto ana homa na kutapika kwa siku tatu, macho yake ni ya njano"
        st.session_state["demo_age"] = 3
        st.session_state["demo_weight"] = 15.0

# 메인 화면
st.title("VillageDoc")
st.subheader("WHO IMCI 프로토콜 기반 AI 의료 진단 보조")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("환자 정보")

    uploaded_image = st.file_uploader(
        "임상 사진 (선택)",
        type=["jpg", "jpeg", "png"],
        help="피부, 눈, 상처 사진을 업로드하면 시각적 분석을 수행합니다"
    )
    if uploaded_image:
        st.image(uploaded_image, caption="업로드된 임상 사진", use_column_width=True)

    symptoms = st.text_area(
        "증상 설명",
        value=st.session_state.get("demo_symptoms", ""),
        placeholder={
            "sw": "예: mtoto ana homa na kutapika kwa siku tatu...",
            "hi": "예: बच्चे को तीन दिन से बुखार है...",
            "fr": "ex: l'enfant a de la fièvre depuis trois jours...",
            "en": "e.g.: child has had fever and vomiting for three days..."
        }[language],
        height=100
    )

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("나이 (세)", min_value=0.0, max_value=120.0,
                              value=float(st.session_state.get("demo_age", 0)), step=0.5)
    with c2:
        weight = st.number_input("체중 (kg)", min_value=0.0, max_value=200.0,
                                 value=float(st.session_state.get("demo_weight", 0)), step=0.5)

    patient_history = st.text_area("과거력 (선택)", height=60,
                                   placeholder="예: 2주 전 말라리아 치료력, 예방접종 미완료...")

    diagnose_btn = st.button("진단 실행", type="primary", use_container_width=True)

with col2:
    st.subheader("진단 결과")

    if diagnose_btn and symptoms:
        with st.spinner("Gemma 4 E4B 추론 중..."):
            image_path = None
            if uploaded_image:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(uploaded_image.getbuffer())
                    image_path = tmp.name

            result = runner.diagnose(
                symptoms=symptoms,
                language=language,
                image_path=image_path,
                patient_age_months=int(age * 12) if age else None,
                patient_weight_kg=weight if weight else None
            )

        # 위험도 표시
        risk = result.get("risk_score", 0)
        risk_class = "risk-high" if risk >= 8 else "risk-medium" if risk >= 5 else "risk-low"
        risk_text = "즉시 이송 필요!" if risk >= 8 else "이송 권고" if risk >= 5 else "가정 치료 가능"
        st.markdown(
            f'<div class="{risk_class}"><h3>위험도: {risk}/10 — {risk_text}</h3></div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        # 위험 징후
        red_flags = result.get("red_flags", [])
        if red_flags:
            st.error("위험 징후 감지!")
            for flag in red_flags:
                st.markdown(f"🚨 {flag}")

        # 감별 진단
        st.subheader("감별 진단")
        for dx in result.get("differential_diagnosis", []):
            conf = dx.get("confidence", 0)
            with st.expander(f"{dx['rank']}. {dx.get('condition', '')} ({conf*100:.0f}%)", expanded=dx["rank"] == 1):
                if dx.get("condition_en"):
                    st.caption(f"영어: {dx['condition_en']}")
                if dx.get("icd10"):
                    st.caption(f"ICD-10: {dx['icd10']}")
                st.progress(conf)
                st.write(dx.get("reasoning", ""))

        # 치료 프로토콜
        protocol = result.get("treatment_protocol", {})
        st.subheader("치료 프로토콜")

        actions = protocol.get("immediate_actions", [])
        if actions:
            st.write("**즉각 조치:**")
            for action in actions:
                st.write(f"✓ {action}")

        meds = protocol.get("medications", [])
        if meds:
            st.write("**처방:**")
            for med in meds:
                avail_icon = "✅" if med.get("available_locally") else "❌"
                st.write(
                    f"{avail_icon} **{med['name']}**: {med.get('dose', '')} "
                    f"| {med.get('frequency', '')} | {med.get('duration', '')}"
                )

        if protocol.get("referral_needed"):
            urgency_map = {
                "immediate": "즉시 이송!!!",
                "within_6h": "6시간 이내",
                "within_24h": "24시간 이내",
                "within_week": "1주일 이내"
            }
            urgency = urgency_map.get(protocol.get("referral_urgency", ""), "이송 권고")
            st.warning(f"🏥 이송: {urgency}\n\n{protocol.get('referral_reason', '')}")

        counseling = result.get("patient_counseling", "")
        if counseling:
            st.info(f"💬 보호자 교육:\n{counseling}")

        # JSON 출력 (EHR 연동)
        with st.expander("JSON 출력 (EHR 연동용)", expanded=False):
            st.json(result)

    elif diagnose_btn:
        st.warning("증상을 입력해주세요.")
    else:
        st.info("왼쪽에서 환자 정보를 입력하고 '진단 실행'을 클릭하세요.\n\n"
                "사이드바의 '데모: 말라리아 케이스'로 샘플 케이스를 테스트할 수 있습니다.")

# 하단 정보
st.divider()
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("지원 언어", "10개")
with col_b:
    st.metric("WHO 프로토콜", "IMCI 2024")
with col_c:
    st.metric("동작 환경", "완전 오프라인")

st.caption(
    "VillageDoc은 커뮤니티 보건 인력(CHW)의 의사결정을 지원하는 도구입니다. "
    "전문 의사를 대체하지 않으며, 위험도 7점 이상 케이스는 반드시 의사에게 이송하세요."
)
