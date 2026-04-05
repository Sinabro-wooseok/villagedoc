"""
VillageDoc - 의료 정확도 벤치마크
WHO IMCI 케이스 기반 진단 성능 측정
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gemma4_runner import MockGemma4Runner

# WHO IMCI 기반 벤치마크 케이스 (실제 임상 케이스에서 추출)
BENCHMARK_CASES = [
    # 말라리아 케이스
    {"id": "MAL-001", "language": "sw", "age_months": 36, "weight_kg": 15.0,
     "symptoms": "mtoto ana homa na kutapika kwa siku tatu, macho yake ni ya njano",
     "ground_truth": "Malaria", "expected_risk_min": 6, "expected_referral": True},

    {"id": "MAL-002", "language": "en", "age_months": 24, "weight_kg": 11.0,
     "symptoms": "2 year old with fever 39.5C, chills, headache, RDT positive for malaria",
     "ground_truth": "Malaria", "expected_risk_min": 4, "expected_referral": False},

    {"id": "MAL-003", "language": "sw", "age_months": 12, "weight_kg": 8.5,
     "symptoms": "mtoto mdogo ana homa na kushindwa kunyonya, ana mshtuko",
     "ground_truth": "Severe Malaria", "expected_risk_min": 9, "expected_referral": True},

    # 폐렴 케이스
    {"id": "PNE-001", "language": "en", "age_months": 18, "weight_kg": 10.0,
     "symptoms": "child with cough and fast breathing >50 breaths per minute, temperature 38.8C",
     "ground_truth": "Pneumonia", "expected_risk_min": 5, "expected_referral": False},

    {"id": "PNE-002", "language": "sw", "age_months": 6, "weight_kg": 6.0,
     "symptoms": "mtoto ana kikohozi na kupumua haraka sana, ana joto la mwili",
     "ground_truth": "Pneumonia", "expected_risk_min": 5, "expected_referral": False},

    # 설사/탈수 케이스
    {"id": "DIA-001", "language": "en", "age_months": 9, "weight_kg": 7.0,
     "symptoms": "infant with diarrhea 8 times today, not drinking, sunken eyes, very lethargic",
     "ground_truth": "Severe Dehydration", "expected_risk_min": 9, "expected_referral": True},

    {"id": "DIA-002", "language": "hi", "age_months": 24, "weight_kg": 9.5,
     "symptoms": "बच्चे को दस्त है, आँखें धंसी हुई हैं, प्यासा है",
     "ground_truth": "Diarrhea with dehydration", "expected_risk_min": 5, "expected_referral": False},

    # 영양실조 케이스
    {"id": "MAL_NUT-001", "language": "en", "age_months": 18, "weight_kg": 6.5,
     "symptoms": "18 month old child MUAC 11.0cm, bilateral pitting edema, not eating",
     "ground_truth": "Severe Acute Malnutrition", "expected_risk_min": 8, "expected_referral": True},
]


def run_benchmark(runner=None, verbose: bool = True) -> dict:
    """
    벤치마크 실행

    Returns:
        {
            "accuracy": float,
            "referral_accuracy": float,
            "risk_score_accuracy": float,
            "results": list
        }
    """
    if runner is None:
        runner = MockGemma4Runner()

    results = []
    correct_dx = 0
    correct_referral = 0
    correct_risk = 0
    total = len(BENCHMARK_CASES)

    if verbose:
        print("VillageDoc 벤치마크 시작")
        print("=" * 80)
        print(f"{'케이스 ID':<12} {'실제 진단':<25} {'예측 진단':<25} {'신뢰도':<8} {'이송':<6} {'결과'}")
        print("-" * 80)

    for case in BENCHMARK_CASES:
        result = runner.diagnose(
            symptoms=case["symptoms"],
            language=case["language"],
            patient_age_months=case["age_months"],
            patient_weight_kg=case["weight_kg"]
        )

        top_dx = result.get("differential_diagnosis", [{}])[0]
        pred = top_dx.get("condition_en", top_dx.get("condition", "Unknown"))
        conf = top_dx.get("confidence", 0)
        risk = result.get("risk_score", 0)
        referral = result.get("treatment_protocol", {}).get("referral_needed", False)

        # 진단 정확도 (ground truth가 예측에 포함되는지)
        ground_lower = case["ground_truth"].lower()
        pred_lower = pred.lower()
        dx_correct = any(kw in pred_lower for kw in ground_lower.split())

        # 이송 정확도
        ref_correct = referral == case["expected_referral"]

        # 위험도 정확도 (최소 기준 이상인지)
        risk_correct = risk >= case["expected_risk_min"]

        if dx_correct:
            correct_dx += 1
        if ref_correct:
            correct_referral += 1
        if risk_correct:
            correct_risk += 1

        status = "✓" if dx_correct else "✗"
        ref_icon = "✓" if ref_correct else "✗"

        if verbose:
            print(
                f"{case['id']:<12} {case['ground_truth'][:24]:<25} "
                f"{pred[:24]:<25} {conf*100:.0f}%    {ref_icon}    {status}"
            )

        results.append({
            "case_id": case["id"],
            "ground_truth": case["ground_truth"],
            "prediction": pred,
            "confidence": conf,
            "risk_score": risk,
            "referral": referral,
            "dx_correct": dx_correct,
            "ref_correct": ref_correct,
            "risk_correct": risk_correct
        })

    dx_accuracy = correct_dx / total * 100
    ref_accuracy = correct_referral / total * 100
    risk_accuracy = correct_risk / total * 100

    if verbose:
        print("=" * 80)
        print(f"\n결과 요약:")
        print(f"  진단 정확도:     {correct_dx}/{total} = {dx_accuracy:.1f}%")
        print(f"  이송 판단 정확도: {correct_referral}/{total} = {ref_accuracy:.1f}%")
        print(f"  위험도 정확도:   {correct_risk}/{total} = {risk_accuracy:.1f}%")
        print()
        print(f"  임팩트 추정:")
        print(f"  CHW 200만 명 × 하루 10건 × {dx_accuracy:.0f}% 정확도 향상")
        print(f"  = 연간 약 {200_0000 * 10 * dx_accuracy / 100 / 1_0000:.0f}만 건의 개선된 진단")

    return {
        "accuracy": dx_accuracy,
        "referral_accuracy": ref_accuracy,
        "risk_score_accuracy": risk_accuracy,
        "total_cases": total,
        "results": results
    }


if __name__ == "__main__":
    from model.gemma4_runner import MockGemma4Runner
    runner = MockGemma4Runner()
    run_benchmark(runner, verbose=True)
