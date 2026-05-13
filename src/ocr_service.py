import time
import cv2
import pytesseract
import numpy as np

from PIL import Image
from PIL import ImageDraw

from .config import OCR_LANGUAGES


# WINDOWS ONLY
# Uncomment and update if needed

# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )


class NepaliOCRService:
    def __init__(self):
        self.languages = OCR_LANGUAGES

    def preprocess_image(self, image: Image.Image):
        image = image.convert("RGB")

        img_np = np.array(image)

        gray = cv2.cvtColor(
            img_np,
            cv2.COLOR_RGB2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        return thresh

    def extract_text(self, image: Image.Image):
        start_time = time.time()

        processed = self.preprocess_image(image)

        data = pytesseract.image_to_data(
            processed,
            lang=self.languages,
            output_type=pytesseract.Output.DICT
        )

        extracted_text = []

        confidences = []

        annotated_image = image.copy()

        draw = ImageDraw.Draw(annotated_image)

        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()

            conf = data["conf"][i]

            if text != "" and int(conf) > 0:
                extracted_text.append(text)

                confidences.append(int(conf))

                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]

                draw.rectangle(
                    [(x, y), (x + w, y + h)],
                    outline="red",
                    width=2
                )

        avg_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0
        )

        processing_time = round(
            time.time() - start_time,
            2
        )

        return {
            "text": " ".join(extracted_text),
            "confidence": round(avg_confidence, 2),
            "annotated_image": annotated_image,
            "processing_time": processing_time
        }