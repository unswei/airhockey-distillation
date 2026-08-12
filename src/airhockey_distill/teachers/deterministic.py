"""Narrow adapters for reproducible Dreamer teacher inference."""

from __future__ import annotations

from typing import Any


_ORIGINAL_SAMPLE_ATTRIBUTE = "_airhockey_original_sample"


def enable_deterministic_dreamer_inference(
    aggregate_output_type: type[Any],
) -> None:
    """Use distribution predictions for RSSM state and policy action outputs.

    DreamerV3 wraps both the categorical RSSM posterior and the policy action
    distribution in embodied.jax.outs.Agg. Its policy path calls the sample
    method even in evaluation mode. Replacing that method with pred selects
    the categorical mode and the bounded-normal action mean.
    """

    if hasattr(aggregate_output_type, _ORIGINAL_SAMPLE_ATTRIBUTE):
        return
    original_sample = getattr(aggregate_output_type, "sample", None)
    if not callable(original_sample) or not callable(
        getattr(aggregate_output_type, "pred", None)
    ):
        raise TypeError("aggregate output type must define callable sample and pred")

    setattr(aggregate_output_type, _ORIGINAL_SAMPLE_ATTRIBUTE, original_sample)

    def deterministic_sample(
        self: Any,
        seed: Any,
        shape: tuple[int, ...] = (),
    ) -> Any:
        del seed
        if shape:
            raise ValueError(
                "deterministic Dreamer inference does not support sample shapes"
            )
        return self.pred()

    aggregate_output_type.sample = deterministic_sample
