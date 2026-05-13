from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

from .config import TRANSLATION_MODEL


class TranslationService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL
        )

    def chunk_text(
        self,
        text,
        max_words=150
    ):
        words = text.split()

        chunks = []

        for i in range(0, len(words), max_words):
            chunk = " ".join(
                words[i:i + max_words]
            )

            chunks.append(chunk)

        return chunks

    def translate_ne_to_en(
        self,
        text: str
    ) -> str:

        if not text.strip():
            return ""

        chunks = self.chunk_text(text)

        translated_chunks = []

        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True
            )

            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=(
                    self.tokenizer.convert_tokens_to_ids(
                        "eng_Latn"
                    )
                ),
                max_length=512
            )

            translated_text = (
                self.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True
                )[0]
            )

            translated_chunks.append(
                translated_text
            )

        return " ".join(translated_chunks)