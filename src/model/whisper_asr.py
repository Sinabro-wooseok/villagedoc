"""
VillageDoc - Whisper-tiny 음성 인식
오프라인 다국어 STT: 스와힐리어, 힌디어, 프랑스어 등 지원
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WhisperASR:
    """
    Whisper-tiny 오프라인 음성-텍스트 변환
    모델 크기: ~39MB (CPU에서도 빠른 추론)
    지원 언어: 99개 언어 (아프리카 언어 포함)
    """

    SUPPORTED_LANGUAGE_CODES = {
        "sw": "swahili",
        "hi": "hindi",
        "fr": "french",
        "en": "english",
        "am": "amharic",
        "ha": "hausa",
        "yo": "yoruba",
        "pt": "portuguese",
        "id": "indonesian",
        "ar": "arabic"
    }

    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        """
        Args:
            model_size: "tiny" (39MB), "base" (74MB), "small" (244MB)
            device: "cpu" 또는 "cuda"
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Whisper 모델 로드"""
        try:
            import whisper
            logger.info(f"Whisper-{self.model_size} 로딩...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper 모델 로드 완료")
        except ImportError:
            logger.warning("openai-whisper 미설치. pip install openai-whisper")
            self.model = None
        except Exception as e:
            logger.error(f"Whisper 로드 실패: {e}")
            self.model = None

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        auto_detect: bool = True
    ) -> dict:
        """
        음성 파일을 텍스트로 변환

        Args:
            audio_path: 오디오 파일 경로 (wav, mp3, m4a 등)
            language: 언어 코드 (None이면 자동 감지)
            auto_detect: True면 언어 자동 감지 후 한 번 더 전사

        Returns:
            {"text": str, "language": str, "confidence": float}
        """
        if not self.model:
            logger.warning("Whisper 모델 없음 - 텍스트 입력 모드 사용")
            return {"text": "", "language": language or "unknown", "confidence": 0.0}

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일 없음: {audio_path}")

        whisper_lang = None
        if language and language in self.SUPPORTED_LANGUAGE_CODES:
            whisper_lang = self.SUPPORTED_LANGUAGE_CODES[language]

        logger.info(f"음성 인식 시작: {audio_path} (언어: {whisper_lang or '자동감지'})")

        result = self.model.transcribe(
            audio_path,
            language=whisper_lang,
            fp16=(self.device == "cuda"),
            verbose=False
        )

        detected_lang = result.get("language", language or "unknown")
        text = result.get("text", "").strip()

        # 평균 토큰 로그 확률로 신뢰도 추정
        segments = result.get("segments", [])
        if segments:
            avg_logprob = sum(s.get("avg_logprob", -1) for s in segments) / len(segments)
            confidence = min(1.0, max(0.0, (avg_logprob + 1.0)))
        else:
            confidence = 0.5

        logger.info(f"인식 결과: '{text[:100]}...' (언어: {detected_lang}, 신뢰도: {confidence:.2f})")

        return {
            "text": text,
            "language": detected_lang,
            "confidence": confidence
        }

    def transcribe_from_microphone(self, duration_seconds: int = 10, language: Optional[str] = None) -> dict:
        """
        마이크에서 실시간 녹음 후 전사 (Streamlit/Android 환경)

        Args:
            duration_seconds: 녹음 시간 (초)
            language: 언어 코드
        """
        try:
            import sounddevice as sd
            import numpy as np
            import soundfile as sf
            import tempfile

            sample_rate = 16000
            logger.info(f"녹음 시작... ({duration_seconds}초)")
            audio = sd.rec(
                int(duration_seconds * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32"
            )
            sd.wait()
            logger.info("녹음 완료")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, sample_rate)
                return self.transcribe(tmp.name, language=language)

        except ImportError:
            logger.error("sounddevice/soundfile 미설치. pip install sounddevice soundfile")
            raise


class MockWhisperASR:
    """UI 테스트용 Mock STT"""

    DEMO_RESPONSES = {
        "sw": "mtoto ana homa na kutapika kwa siku tatu, macho yake ni ya njano",
        "hi": "बच्चे को तीन दिन से बुखार और उल्टी है, आँखें पीली हैं",
        "en": "child has fever and vomiting for three days, eyes are yellow"
    }

    def transcribe(self, audio_path: str, language: str = "sw", **kwargs) -> dict:
        text = self.DEMO_RESPONSES.get(language, self.DEMO_RESPONSES["en"])
        return {"text": text, "language": language, "confidence": 0.95}

    def transcribe_from_microphone(self, duration_seconds: int = 10, language: str = "sw") -> dict:
        text = self.DEMO_RESPONSES.get(language, self.DEMO_RESPONSES["en"])
        return {"text": text, "language": language, "confidence": 0.95}
