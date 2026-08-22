# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan import (
    FastHunyuanConfig,
    HunyuanConfig,
)


def test_fast_hunyuan_defaults_to_parallel_tiled_vae_decode():
    assert HunyuanConfig().vae_config.parallel_decode_mode == "auto"
    assert FastHunyuanConfig().vae_config.parallel_decode_mode == "tiled"


def test_fast_hunyuan_parallel_decode_mode_can_be_overridden():
    config = FastHunyuanConfig()

    config.update_config_from_dict({"vae_config.parallel_decode_mode": "spatial_shard"})

    assert config.vae_config.parallel_decode_mode == "spatial_shard"
