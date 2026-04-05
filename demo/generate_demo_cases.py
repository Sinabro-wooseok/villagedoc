"""
VillageDoc 데모 케이스 이미지 생성
실제 임상 이미지 대신 Kaggle 제출용 대표 이미지 생성
"""

from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cases")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_case_image(filename: str, title: str, color: tuple, symptoms: list):
    """데모 케이스 카드 이미지 생성"""
    img = Image.new("RGB", (400, 300), color=color)
    draw = ImageDraw.Draw(img)

    # 제목
    draw.rectangle([0, 0, 400, 60], fill=(0, 0, 0, 180))
    draw.text((10, 10), title, fill="white")

    # 증상 목록
    y = 80
    for symptom in symptoms:
        draw.text((20, y), f"• {symptom}", fill=(30, 30, 30))
        y += 30

    img.save(os.path.join(OUTPUT_DIR, filename))
    print(f"생성: {filename}")


cases = [
    {
        "filename": "malaria_jaundice.jpg",
        "title": "CASE 1: Malaria with Jaundice",
        "color": (255, 220, 150),
        "symptoms": [
            "Fever 39.2°C (3 days)",
            "Repeated vomiting",
            "Yellow sclera (jaundice)",
            "Patient: 3yr, 15kg",
            "Location: Rural Tanzania",
        ],
    },
    {
        "filename": "pneumonia_child.jpg",
        "title": "CASE 2: Childhood Pneumonia",
        "color": (180, 210, 240),
        "symptoms": [
            "Cough (2 days)",
            "Fast breathing 48/min",
            "Temperature 38.8°C",
            "Patient: 2yr, 12kg",
            "Location: Rural Bihar, India",
        ],
    },
    {
        "filename": "sam_malnutrition.jpg",
        "title": "CASE 3: Severe Acute Malnutrition",
        "color": (220, 200, 180),
        "symptoms": [
            "MUAC: 10.8cm (<11.5cm)",
            "Bilateral pitting edema",
            "Not eating (5 days)",
            "Patient: 18mo, 6.5kg",
            "Location: South Sudan",
        ],
    },
    {
        "filename": "diarrhea_dehydration.jpg",
        "title": "CASE 4: Severe Dehydration",
        "color": (200, 230, 200),
        "symptoms": [
            "Diarrhea 8x today",
            "Unable to drink",
            "Sunken eyes",
            "Patient: 9mo, 7kg",
            "Location: DRC",
        ],
    },
]

for case in cases:
    make_case_image(**case)

print(f"\n데모 케이스 이미지 {len(cases)}개 생성 완료: {OUTPUT_DIR}")
