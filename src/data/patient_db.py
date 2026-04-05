"""
VillageDoc - SQLite 오프라인 환자 기록 DB
인터넷 없이 환자 데이터 저장 및 조회
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent.parent / "data" / "villagedoc.db"


def get_connection() -> sqlite3.Connection:
    os.makedirs(DB_PATH.parent, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """DB 초기화 및 테이블 생성"""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_code TEXT UNIQUE NOT NULL,
                chw_id TEXT,
                village TEXT,
                age_months INTEGER,
                weight_kg REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                symptoms TEXT NOT NULL,
                language TEXT DEFAULT 'en',
                image_path TEXT,
                diagnosis_json TEXT,
                risk_score INTEGER,
                referral_needed INTEGER DEFAULT 0,
                referral_urgency TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            );

            CREATE TABLE IF NOT EXISTS follow_ups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id INTEGER,
                notes TEXT,
                outcome TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (consultation_id) REFERENCES consultations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_consultations_patient
                ON consultations(patient_id);
            CREATE INDEX IF NOT EXISTS idx_consultations_risk
                ON consultations(risk_score);
        """)
    print(f"DB 초기화 완료: {DB_PATH}")


def save_consultation(
    symptoms: str,
    diagnosis: dict,
    language: str = "en",
    patient_code: Optional[str] = None,
    chw_id: Optional[str] = None,
    village: Optional[str] = None,
    age_months: Optional[int] = None,
    weight_kg: Optional[float] = None,
    image_path: Optional[str] = None
) -> int:
    """진단 결과를 DB에 저장. 상담 ID 반환."""
    with get_connection() as conn:
        # 환자 레코드 처리
        patient_id = None
        if patient_code:
            cursor = conn.execute(
                "SELECT id FROM patients WHERE patient_code = ?", (patient_code,)
            )
            row = cursor.fetchone()
            if row:
                patient_id = row["id"]
            else:
                cursor = conn.execute(
                    """INSERT INTO patients (patient_code, chw_id, village, age_months, weight_kg)
                       VALUES (?, ?, ?, ?, ?)""",
                    (patient_code, chw_id, village, age_months, weight_kg)
                )
                patient_id = cursor.lastrowid

        # 상담 기록 저장
        protocol = diagnosis.get("treatment_protocol", {})
        cursor = conn.execute(
            """INSERT INTO consultations
               (patient_id, symptoms, language, image_path, diagnosis_json, risk_score, referral_needed, referral_urgency)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                patient_id,
                symptoms,
                language,
                image_path,
                json.dumps(diagnosis, ensure_ascii=False),
                diagnosis.get("risk_score", 0),
                int(protocol.get("referral_needed", False)),
                protocol.get("referral_urgency")
            )
        )
        return cursor.lastrowid


def get_high_risk_cases(risk_threshold: int = 7) -> list:
    """위험도 임계값 이상 케이스 조회"""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT c.*, p.patient_code, p.village
               FROM consultations c
               LEFT JOIN patients p ON c.patient_id = p.id
               WHERE c.risk_score >= ?
               ORDER BY c.created_at DESC""",
            (risk_threshold,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_daily_stats(date: Optional[str] = None) -> dict:
    """일별 통계 조회"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    with get_connection() as conn:
        row = conn.execute(
            """SELECT
               COUNT(*) as total_consultations,
               SUM(referral_needed) as referrals,
               AVG(risk_score) as avg_risk,
               MAX(risk_score) as max_risk
               FROM consultations
               WHERE date(created_at) = ?""",
            (date,)
        ).fetchone()
    return dict(row) if row else {}


if __name__ == "__main__":
    init_db()
    print("환자 DB 초기화 완료")
