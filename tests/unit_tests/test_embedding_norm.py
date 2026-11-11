"""Tests for --embedding-norm (a norm with a learnable gain on the embedding output)."""

import argparse

from megatron.core.transformer.transformer_config import TransformerConfig


def test_argument_and_default():
    from megatron.training.arguments import add_megatron_arguments

    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    assert parser.parse_args([]).embedding_norm is False
    assert parser.parse_args(["--embedding-norm"]).embedding_norm is True
    assert TransformerConfig.__dataclass_fields__["embedding_norm"].default is False
