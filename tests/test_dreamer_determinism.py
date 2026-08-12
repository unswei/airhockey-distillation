import pytest

from airhockey_distill.teachers import enable_deterministic_dreamer_inference


def test_deterministic_dreamer_inference_uses_distribution_prediction() -> None:
    class FakeAggregateOutput:
        def pred(self):
            return ("prediction", 7)

        def sample(self, seed, shape=()):
            return ("sample", seed, shape)

    output = FakeAggregateOutput()
    assert output.sample(11) == ("sample", 11, ())

    enable_deterministic_dreamer_inference(FakeAggregateOutput)

    assert output.sample(11) == ("prediction", 7)
    assert output.sample(99) == ("prediction", 7)


def test_deterministic_dreamer_inference_is_idempotent_and_rejects_shapes() -> None:
    class FakeAggregateOutput:
        def pred(self):
            return 3

        def sample(self, seed, shape=()):
            return seed, shape

    enable_deterministic_dreamer_inference(FakeAggregateOutput)
    deterministic_sample = FakeAggregateOutput.sample
    enable_deterministic_dreamer_inference(FakeAggregateOutput)

    assert FakeAggregateOutput.sample is deterministic_sample
    with pytest.raises(ValueError, match="does not support sample shapes"):
        FakeAggregateOutput().sample(1, shape=(2,))
