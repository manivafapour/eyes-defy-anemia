"""Framework-agnostic two-stage inference pipeline.

    raw eye photo -> [Stage 1: segment] -> mask
                  -> [synthesize crop]   -> conjunctiva crop
                  -> [Stage 2: classify] -> P(anemic) -> label

No FastAPI, no HTTP, no file I/O beyond PIL -- so this class is unit-testable and
reusable from a CLI, a notebook, or a batch job. The API layer (``app.api``) is a
thin wrapper around :meth:`InferencePipeline.run`.

Models are injected (dependency inversion): the pipeline talks only to the
``BaseSegmenter`` / ``BaseClassifier`` interfaces, so mock and real backends are
interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from app.core.exceptions import SegmentationQualityError
from app.core.preprocessing import synthesize_crop
from app.models.base import (
    BaseClassifier,
    BaseSegmenter,
    ClassificationResult,
    SegmentationResult,
)


@dataclass(frozen=True)
class PipelineResult:
    """Everything a caller (API, CLI, test) needs from one inference."""

    label: str
    probability: float
    threshold: float
    is_anemic: bool
    crop_used: Image.Image
    segmentation: SegmentationResult
    classification: ClassificationResult
    warnings: list[str] = field(default_factory=list)

    @property
    def is_mock(self) -> bool:
        return self.segmentation.is_mock or self.classification.is_mock


class InferencePipeline:
    def __init__(
        self,
        segmenter: BaseSegmenter,
        classifier: BaseClassifier,
        *,
        min_coverage: float = 0.005,
        max_coverage: float = 0.60,
    ) -> None:
        self.segmenter = segmenter
        self.classifier = classifier
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage

    def run(self, image: Image.Image) -> PipelineResult:
        # --- Stage 1: segmentation ---
        seg = self.segmenter.segment(image)
        warnings = self._quality_gate(seg)

        # --- Bridge: synthesize the crop the classifier expects (PARITY step) ---
        crop = synthesize_crop(image, seg.mask)

        # --- Stage 2: classification ---
        cls = self.classifier.predict(crop)

        if seg.is_mock:
            warnings.append(
                "Stage 1 segmentation is MOCKED: the crop is a fixed placeholder "
                "region, not a real conjunctiva localization."
            )

        return PipelineResult(
            label=cls.label,
            probability=cls.probability,
            threshold=cls.threshold,
            is_anemic=cls.label == "anemic",
            crop_used=crop,
            segmentation=seg,
            classification=cls,
            warnings=warnings,
        )

    def _quality_gate(self, seg: SegmentationResult) -> list[str]:
        """Reject implausible masks before wasting a classifier forward pass."""
        if seg.coverage < self.min_coverage:
            raise SegmentationQualityError(
                "No conjunctiva tissue detected. Please upload a clear, well-lit "
                "close-up of the lower inner eyelid.",
                coverage=seg.coverage,
            )
        if seg.coverage > self.max_coverage:
            raise SegmentationQualityError(
                "Segmented region is implausibly large; the image may not be a "
                "close-up eye photo.",
                coverage=seg.coverage,
            )
        return []
