"""Domain exceptions for the inference layer (framework-agnostic).

These carry no HTTP semantics; the API layer maps them to status codes so the
core stays reusable from a CLI, notebook, or batch job.
"""
from __future__ import annotations


class PipelineError(Exception):
    """Base class for all inference-pipeline failures."""


class InvalidImageError(PipelineError):
    """Uploaded bytes could not be decoded as an image."""


class SegmentationQualityError(PipelineError):
    """Segmentation produced an implausible mask (empty or near-full-frame).

    Almost always means the upload isn't a usable close-up eye photo -- the
    documented failure mode where a segmenter flags a whole frame on
    out-of-distribution input.
    """

    def __init__(self, message: str, coverage: float) -> None:
        super().__init__(message)
        self.coverage = coverage
