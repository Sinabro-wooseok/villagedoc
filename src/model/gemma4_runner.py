"""
VillageDoc - Gemma 4 E4B 추론 엔진
transformers 공식 API: 멀티모달 비전 + Function Calling + Extended Thinking
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
import logging

sys.path.insert(0, str(Path(__file__).parent))

from diagnosis_schema import (
    DIAGNOSIS_FUNCTION_SCHEMA,
    SYSTEM_PROMPT_TEMPLATE,
    SUPPORTED_LANGUAGES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_clinical_context() -> str:
    """WHO IMCI 프로토콜과 필수 의약품 목록을 컨텍스트로 로드"""
    data_dir = Path(__file__).parent.parent / "data"
    context_parts = []

    imci_path = data_dir / "who_imci_protocol.json"
    if imci_path.exists():
        with open(imci_path) as f:
            imci = json.load(f)
        context_parts.append(
            f"## WHO IMCI 프로토콜\n{json.dumps(imci['conditions'], ensure_ascii=False, indent=2)}"
        )

    medicines_path = data_dir / "essential_medicines.json"
    if medicines_path.exists():
        with open(medicines_path) as f:
            medicines = json.load(f)
        context_parts.append(
            f"## WHO 필수 의약품\n{json.dumps(medicines['medicines_by_condition'], ensure_ascii=False, indent=2)}"
        )

    return "\n\n".join(context_parts)


class Gemma4Runner:
    """
    Gemma 4 E4B-IT transformers 기반 추론기
    - Kaggle: /kaggle/input/gemma-4/transformers/gemma-4-e4b-it/
    - HuggingFace: google/gemma-4-E4B-it
    - 멀티모달 비전 + Function Calling + Extended Thinking
    """

    KAGGLE_MODEL_PATH = "/kaggle/input/gemma-4/transformers/gemma-4-e4b-it"
    HF_MODEL_ID = "google/gemma-4-E4B-it"

    def __init__(self, model_path: Optional[str] = None):
        self.clinical_context = load_clinical_context()
        self.processor = None
        self.model = None

        if model_path is None:
            model_path = (
                self.KAGGLE_MODEL_PATH
                if os.path.exists(self.KAGGLE_MODEL_PATH)
                else self.HF_MODEL_ID
            )

        self.model_path = model_path
        self._load()

    def _load(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText

            logger.info(f"Gemma 4 E4B 로딩: {self.model_path}")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.model.eval()
            self._torch = torch
            logger.info("Gemma 4 E4B 로드 완료")
        except ImportError:
            logger.error("transformers 미설치: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise

    def _build_messages(
        self,
        symptoms: str,
        language: str,
        image_path: Optional[str],
        patient_age_months: Optional[int],
        patient_weight_kg: Optional[float],
    ) -> list:
        lang_name = SUPPORTED_LANGUAGES.get(language, "English")
        system_text = SYSTEM_PROMPT_TEMPLATE.format(
            language=lang_name,
            clinical_context=self.clinical_context[:6000]
        )

        info_parts = []
        if patient_age_months:
            y, m = divmod(int(patient_age_months), 12)
            info_parts.append(f"Age: {y}y {m}mo" if y else f"Age: {m}mo")
        if patient_weight_kg:
            info_parts.append(f"Weight: {patient_weight_kg}kg")

        user_text = ""
        if info_parts:
            user_text += f"Patient: {', '.join(info_parts)}\n"
        user_text += f"Symptoms: {symptoms}\n\nPlease call submit_medical_diagnosis."

        user_content = []
        if image_path and os.path.exists(image_path):
            user_content.append({"type": "image", "image": image_path})
        user_content.append({"type": "text", "text": user_text})

        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, text: str) -> dict:
        """모델 응답에서 JSON 추출"""
        # ```json ... ``` 블록
        m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # 중괄호 직접 파싱
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            pass
        # 파싱 실패 - 안전 기본값
        logger.warning("JSON 파싱 실패")
        return {
            "differential_diagnosis": [
                {"rank": 1, "condition": "분석 불가", "confidence": 0.1, "reasoning": text[:300]}
            ],
            "treatment_protocol": {
                "immediate_actions": ["의사에게 이송"],
                "referral_needed": True,
                "referral_urgency": "within_24h",
            },
            "risk_score": 7,
            "red_flags": ["데이터 부족"],
        }

    def diagnose(
        self,
        symptoms: str,
        language: str = "sw",
        image_path: Optional[str] = None,
        patient_age_months: Optional[int] = None,
        patient_weight_kg: Optional[float] = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 1024,
    ) -> dict:
        """
        Gemma 4 E4B 의료 진단 실행

        Args:
            symptoms: 증상 (현지어 가능)
            language: 출력 언어 코드 (sw/hi/fr/en)
            image_path: 임상 사진 경로 (선택)
            patient_age_months: 나이 (개월)
            patient_weight_kg: 체중 (kg)
            enable_thinking: True = Extended Thinking (고위험 케이스 권장)
            max_new_tokens: 최대 출력 토큰
        """
        messages = self._build_messages(
            symptoms, language, image_path, patient_age_months, patient_weight_kg
        )

        logger.info(f"진단 시작 (언어={language}, 이미지={'있음' if image_path else '없음'}, thinking={enable_thinking})")

        inputs = self.processor.apply_chat_template(
            messages,
            tools=[{"type": "function", "function": DIAGNOSIS_FUNCTION_SCHEMA}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(self.model.device, dtype=self._torch.bfloat16)

        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        # Function call 파싱 (공식 API)
        try:
            parsed = self.processor.parse_response(response_text)
            if isinstance(parsed, dict):
                content = parsed.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "tool_use":
                            result = item.get("input", {})
                            if result:
                                return result
                if isinstance(content, str) and "{" in content:
                    return self._parse_response(content)
        except Exception:
            pass

        result = self._parse_response(response_text)

        risk = result.get("risk_score", 0)
        if risk >= 8:
            logger.warning(f"고위험 케이스 (risk={risk}): {result.get('red_flags', [])}")

        return result


class MockGemma4Runner:
    """UI 테스트 및 로컬 개발용 Mock (모델 없이 동작)"""

    def diagnose(self, symptoms: str, language: str = "sw", **_kwargs) -> dict:
        logger.info(f"[MOCK] 증상: {symptoms[:50]}...")
        s = symptoms.lower()

        if any(kw in s for kw in ["homa", "fever", "malaria", "kutapika kwa siku", "chills", "rdt", "joto"]):
            return {
                "differential_diagnosis": [
                    {"rank": 1, "condition": "Malaria isiyotatanisha" if language == "sw" else "Uncomplicated Malaria",
                     "condition_en": "Uncomplicated Malaria", "confidence": 0.87, "icd10": "B54",
                     "reasoning": ("Homa ya siku tatu, kutapika, na jaundice inaashiria malaria. "
                                   if language == "sw" else
                                   "Three-day fever, vomiting, and jaundice indicate malaria with hepatic involvement.")},
                    {"rank": 2, "condition": "Hepatitis ya Virusi" if language == "sw" else "Viral Hepatitis A",
                     "condition_en": "Viral Hepatitis A", "confidence": 0.45, "icd10": "B15.9",
                     "reasoning": "Jaundice + homa inaweza kuashiria hepatitis A" if language == "sw" else "Jaundice with fever may indicate Hepatitis A"},
                ],
                "treatment_protocol": {
                    "immediate_actions": [
                        "Pima RDT mara moja" if language == "sw" else "Perform malaria RDT immediately",
                        "Mpe ORS - maji mengi" if language == "sw" else "Give ORS - plenty of fluids",
                        "Paracetamol 250mg kwa homa" if language == "sw" else "Paracetamol 250mg for fever",
                        "Kama RDT chanya: anza AL" if language == "sw" else "If RDT positive: start AL immediately",
                    ],
                    "medications": [
                        {"name": "Artemether-Lumefantrine (AL)", "dose": "2 tabs 20/120mg",
                         "frequency": "BID", "duration": "3 days", "available_locally": True},
                        {"name": "Paracetamol", "dose": "250mg (15mg/kg)",
                         "frequency": "q6h", "duration": "PRN fever", "available_locally": True},
                    ],
                    "referral_needed": True,
                    "referral_urgency": "within_6h",
                    "referral_reason": "Jaundice → severe malaria risk / IV artesunate may be needed",
                },
                "risk_score": 7,
                "red_flags": [
                    "Jaundice (macho ya njano) - severe malaria sign",
                    "Repeated vomiting - dehydration risk",
                    "Age 3yr - high risk group",
                ],
                "patient_counseling": (
                    "Mtoto wako ana dalili za malaria. Mpe AL na maji mengi. Peleka hospitalini ndani ya masaa 6."
                    if language == "sw" else
                    "Your child has malaria symptoms. Give AL now and plenty of fluids. Go to hospital within 6 hours."
                ),
                "data_confidence": "high",
            }

        if any(kw in s for kw in ["kikohozi", "cough", "nimonia", "pneumonia", "breathing", "kupumua haraka"]):
            return {
                "differential_diagnosis": [
                    {"rank": 1, "condition": "Nimonia" if language == "sw" else "Pneumonia",
                     "condition_en": "Pneumonia", "confidence": 0.82, "icd10": "J18.9",
                     "reasoning": "Fast breathing + fever meets WHO IMCI pneumonia criteria."},
                ],
                "treatment_protocol": {
                    "immediate_actions": ["Count respiratory rate", "Check for chest indrawing"],
                    "medications": [
                        {"name": "Amoxicillin", "dose": "40mg/kg/day", "frequency": "BID",
                         "duration": "5 days", "available_locally": True}
                    ],
                    "referral_needed": False,
                    "referral_urgency": None,
                },
                "risk_score": 5,
                "red_flags": ["Monitor for chest indrawing (severe pneumonia)"],
                "data_confidence": "medium",
            }

        if any(kw in s for kw in ["diarrhea", "kuhara", "dehydration", "not drinking", "lethargic", "sunken eyes", "दस्त"]):
            severe = any(kw in s for kw in ["lethargic", "not drinking", "8 times", "unable"])
            return {
                "differential_diagnosis": [
                    {"rank": 1,
                     "condition": "Severe Dehydration" if severe else "Diarrhea with dehydration",
                     "condition_en": "Severe Dehydration" if severe else "Diarrhea with some dehydration",
                     "confidence": 0.85, "icd10": "A09",
                     "reasoning": "Diarrhea with dehydration signs per WHO IMCI."}
                ],
                "treatment_protocol": {
                    "immediate_actions": ["Assess dehydration", "Start ORS immediately"],
                    "medications": [
                        {"name": "ORS", "dose": "75ml/kg over 4h", "frequency": "Continuous",
                         "duration": "Until rehydrated", "available_locally": True},
                        {"name": "Zinc sulfate", "dose": "20mg/day", "frequency": "OD",
                         "duration": "14 days", "available_locally": True},
                    ],
                    "referral_needed": severe,
                    "referral_urgency": "immediate" if severe else None,
                },
                "risk_score": 9 if severe else 5,
                "red_flags": ["Unable to drink - severe dehydration requires IV fluids"] if severe else [],
                "data_confidence": "high",
            }

        if any(kw in s for kw in ["muac", "malnutrition", "wasting", "edema", "utapiamlo", "pitting", "सूजन", "वजन"]):
            return {
                "differential_diagnosis": [
                    {"rank": 1, "condition": "Severe Acute Malnutrition (SAM)",
                     "condition_en": "Severe Acute Malnutrition",
                     "confidence": 0.88, "icd10": "E43",
                     "reasoning": "MUAC <11.5cm with bilateral pitting edema = SAM criteria met."}
                ],
                "treatment_protocol": {
                    "immediate_actions": ["Measure MUAC", "Check bilateral pitting edema", "Start RUTF"],
                    "medications": [
                        {"name": "RUTF (Plumpy'Nut)", "dose": "200kcal/kg/day",
                         "frequency": "Divided", "duration": "Until MUAC >12.5cm", "available_locally": True}
                    ],
                    "referral_needed": True,
                    "referral_urgency": "within_6h",
                },
                "risk_score": 8,
                "red_flags": ["Bilateral pitting edema - SAM with kwashiorkor", "MUAC <11.5cm"],
                "data_confidence": "high",
            }

        return {
            "differential_diagnosis": [
                {"rank": 1, "condition": "Insufficient data", "confidence": 0.3,
                 "reasoning": "More clinical history needed"}
            ],
            "treatment_protocol": {
                "immediate_actions": ["Take detailed history", "Measure vital signs"],
                "referral_needed": False,
            },
            "risk_score": 3,
            "red_flags": [],
            "data_confidence": "low",
        }
