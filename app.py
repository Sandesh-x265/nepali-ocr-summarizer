import gradio as gr

from src.pipeline import (
    NepaliDocumentPipeline
)

print("Loading pipeline...")

pipeline = NepaliDocumentPipeline()

print("Pipeline loaded successfully.")


def process_document(image):

    if image is None:
        return (
            None,
            "",
            "",
            "",
            ""
        )

    try:

        result = pipeline.process(image)

        metrics = (
            f"OCR Confidence: "
            f"{result['confidence']}%\n"
            f"Processing Time: "
            f"{result['processing_time']} sec"
        )

        return (
            result["annotated_image"],
            result["nepali_text"],
            result["english_translation"],
            result["summary"],
            metrics
        )

    except Exception as error:

        return (
            None,
            "",
            "",
            f"Error: {str(error)}",
            "Processing Failed"
        )


demo = gr.Interface(
    fn=process_document,

    inputs=gr.Image(
        type="pil",
        label="Upload Nepali Text Image"
    ),

    outputs=[
        gr.Image(
            label="Detected Text Regions"
        ),

        gr.Textbox(
            label="Extracted Nepali Text",
            lines=6
        ),

        gr.Textbox(
            label="English Translation",
            lines=6
        ),

        gr.Textbox(
            label="Summary",
            lines=5
        ),

        gr.Textbox(
            label="Metrics",
            lines=2
        )
    ],

    title="🇳🇵 Nepali OCR + Translation + Summarization",

    description=(
        "Upload an image containing Nepali text "
        "to extract, translate, and summarize it."
    ),

    flagging_mode="never"
)

if __name__ == "__main__":

    demo.queue()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        debug=True
    )