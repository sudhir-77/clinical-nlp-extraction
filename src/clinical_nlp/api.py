"""FastAPI endpoint that returns schema-validated condition JSON."""

from __future__ import annotations

from fastapi import FastAPI

from clinical_nlp.pipeline import ExtractionPipeline
from clinical_nlp.schema import ExtractRequest, ExtractResponse


def create_app(pipeline: ExtractionPipeline) -> FastAPI:
    app = FastAPI(title="Clinical NLP extraction", version="1.0.0")

    @app.post("/extract", response_model=ExtractResponse)
    def extract(body: ExtractRequest) -> ExtractResponse:
        return pipeline.extract(body.text)

    return app
