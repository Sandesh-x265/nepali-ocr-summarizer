from transformers import pipeline
from transformers.pipelines import Pipeline

from .config import (
    SUMMARIZATION_MODEL,
    MAX_SUMMARY_LENGTH,
    MIN_SUMMARY_LENGTH
)


class SummarizationService:
    def __init__(self):

        self.summarizer: Pipeline = pipeline(
            task="summarization",
            model=SUMMARIZATION_MODEL,
            device=-1
        )

    def summarize(
        self,
        text: str
    ) -> str:

        if not text.strip():
            return ""

        word_count = len(text.split())

        if word_count < 60:
            return (
                "Text too short for summarization."
            )

        summary = self.summarizer(
            text,
            max_length=MAX_SUMMARY_LENGTH,
            min_length=MIN_SUMMARY_LENGTH,
            do_sample=False
        )

        return summary[0]["summary_text"]