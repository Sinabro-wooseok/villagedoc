"""
VillageDoc - Gemma 4 E4B 진단 스키마 (Function Calling용 JSON Schema)
커뮤니티 보건 인력(CHW)을 위한 구조화된 의료 진단 출력
"""

from typing import TypedDict, List, Optional

# Gemma 4 function calling용 JSON Schema
DIAGNOSIS_FUNCTION_SCHEMA = {
    "name": "submit_medical_diagnosis",
    "description": "커뮤니티 보건 인력을 위한 구조화된 의료 진단 및 치료 프로토콜 제출",
    "parameters": {
        "type": "object",
        "properties": {
            "differential_diagnosis": {
                "type": "array",
                "description": "가능성 높은 순서로 정렬된 감별 진단 목록",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer", "description": "우선순위 순위 (1=가장 가능성 높음)"},
                        "condition": {"type": "string", "description": "질환명 (현지어로)"},
                        "condition_en": {"type": "string", "description": "질환명 (영어)"},
                        "confidence": {"type": "number", "description": "신뢰도 0.0-1.0"},
                        "icd10": {"type": "string", "description": "ICD-10 코드"},
                        "reasoning": {"type": "string", "description": "임상적 근거"}
                    },
                    "required": ["rank", "condition", "confidence", "reasoning"]
                }
            },
            "treatment_protocol": {
                "type": "object",
                "properties": {
                    "immediate_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "즉각적 조치 사항 (현지어)"
                    },
                    "medications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "dose": {"type": "string"},
                                "frequency": {"type": "string"},
                                "duration": {"type": "string"},
                                "available_locally": {"type": "boolean", "description": "WHO 필수 의약품 목록 포함 여부"}
                            }
                        }
                    },
                    "referral_needed": {"type": "boolean"},
                    "referral_urgency": {
                        "type": "string",
                        "enum": ["immediate", "within_6h", "within_24h", "within_week", None],
                        "description": "이송 긴급도"
                    },
                    "referral_reason": {"type": "string"},
                    "follow_up_hours": {"type": "integer", "description": "재방문 권장 시간"}
                },
                "required": ["immediate_actions", "referral_needed"]
            },
            "risk_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "위험도 점수 (1=낮음, 10=매우 위험)"
            },
            "red_flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "즉각 이송이 필요한 위험 징후 목록"
            },
            "patient_counseling": {
                "type": "string",
                "description": "환자/보호자에게 전달할 교육 메시지 (현지어)"
            },
            "data_confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "입력 데이터 품질 기반 진단 신뢰도"
            }
        },
        "required": [
            "differential_diagnosis",
            "treatment_protocol",
            "risk_score",
            "red_flags",
            "data_confidence"
        ]
    }
}

# 시스템 프롬프트 - WHO IMCI 프로토콜 기반
SYSTEM_PROMPT_TEMPLATE = """당신은 커뮤니티 보건 인력(CHW, Community Health Worker)을 보조하는 AI 의료 어시스턴트입니다.
WHO IMCI(Integrated Management of Childhood Illness) 프로토콜과 WHO 필수 의약품 목록을 기반으로 진단합니다.

## 핵심 원칙
1. 오진 방지를 위해 항상 감별 진단을 제공
2. 중증 징후(danger signs)를 절대 놓치지 않기
3. 현지에서 구할 수 있는 약품만 처방
4. 의심스러울 때는 이송 권고
5. 응답 언어: {language}

## 제공된 임상 지식
{clinical_context}

## 출력 형식
반드시 `submit_medical_diagnosis` 함수를 호출하여 구조화된 JSON으로만 응답하세요.
절대 일반 텍스트로 답변하지 마세요.

## 중요 면책사항
이 AI는 전문 의사를 대체할 수 없습니다. 위험도 7점 이상 케이스는 반드시 의사에게 이송하세요."""

# 지원 언어 코드
SUPPORTED_LANGUAGES = {
    "sw": "Swahili (스와힐리어)",
    "hi": "Hindi (힌디어)",
    "fr": "French (프랑스어)",
    "en": "English (영어)",
    "ko": "Korean (한국어)",
    "am": "Amharic (암하라어)",
    "ha": "Hausa (하우사어)",
    "yo": "Yoruba (요루바어)",
    "pt": "Portuguese (포르투갈어)",
    "id": "Indonesian (인도네시아어)"
}
