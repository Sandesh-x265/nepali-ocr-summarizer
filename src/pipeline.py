import time

from .ocr_service import NepaliOCRService
from .translation_service import TranslationService
from .summarization_service import (
    SummarizationService
)


class NepaliDocumentPipeline:
    def __init__(self):
        self.ocr_service = (
            NepaliOCRService()
        )

        self.translation_service = (
            TranslationService()
        )

        self.summarization_service = (
            SummarizationService()
        )

    def process(self, image):
        overall_start = time.time()

        ocr_result = (
            self.ocr_service.extract_text(
                image
            )
        )

        nepali_text = ocr_result["text"]

        if not nepali_text.strip():
            return {
                "annotated_image": image,
                "nepali_text": "",
                "english_translation": "",
                "summary": "No text detected.",
                "confidence": 0,
                "processing_time": 0
            }

        english_translation = (
            self.translation_service
            .translate_ne_to_en(
                nepali_text
            )
        )

        summary = (
            self.summarization_service
            .summarize(
                english_translation
            )
        )

        total_time = round(
            time.time() - overall_start,
            2
        )

        return {
            "annotated_image": (
                ocr_result["annotated_image"]
            ),
            "nepali_text": nepali_text,
            "english_translation": (
                english_translation
            ),
            "summary": summary,
            "confidence": (
                ocr_result["confidence"]
            ),
            "processing_time": total_time
        }