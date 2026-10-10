from types import SimpleNamespace

import torch

from megatron.core.inference import utils as inference_utils
from megatron.core.inference.utils import Counter


class _FakeMoELayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._inference_token_dispatcher = SimpleNamespace(_runs_metadata_sync=True)


class TestInferenceUtils:

    def test_counter(self):
        counter = Counter()
        r = next(counter)
        assert r == 0, f'Counter return value should be 0 but it is {r}'
        assert counter.counter == 1, f'Counter should be 1 but it is {counter.counter}'
        counter.reset()
        assert counter.counter == 0, f'Counter should be 0 but it is {counter.counter}'

    def test_moe_cache_rebinds_for_new_model(self, monkeypatch):
        monkeypatch.setattr(inference_utils, "MoELayer", _FakeMoELayer)
        monkeypatch.setattr(inference_utils, "moe_layer_cache", None)
        monkeypatch.setattr(inference_utils, "_moe_layer_cache_model", None)
        monkeypatch.setattr(inference_utils, "_moe_metadata_sync_model", None)

        first_layers = [_FakeMoELayer(), _FakeMoELayer()]
        first_model = torch.nn.Sequential(*first_layers)
        inference_utils.set_moe_metadata_sync(first_model)
        assert inference_utils.moe_layer_cache == first_layers
        assert inference_utils._moe_layer_cache_model is first_model
        assert inference_utils._moe_metadata_sync_model is first_model
        assert first_layers[0]._inference_token_dispatcher._runs_metadata_sync
        assert not first_layers[1]._inference_token_dispatcher._runs_metadata_sync

        second_layers = [_FakeMoELayer(), _FakeMoELayer()]
        second_model = torch.nn.Sequential(*second_layers)
        inference_utils.set_moe_metadata_sync(second_model)

        assert inference_utils.moe_layer_cache == second_layers
        assert inference_utils._moe_layer_cache_model is second_model
        assert inference_utils._moe_metadata_sync_model is second_model
        assert second_layers[0]._inference_token_dispatcher._runs_metadata_sync
        assert not second_layers[1]._inference_token_dispatcher._runs_metadata_sync
        # Rebinding the cache must not mutate the first model's dispatchers.
        assert first_layers[0]._inference_token_dispatcher._runs_metadata_sync
        assert not first_layers[1]._inference_token_dispatcher._runs_metadata_sync
