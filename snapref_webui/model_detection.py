# SPDX-License-Identifier: GPL-3.0-only
from .i18n import SnapRefError

BASE_BLOCKS_29B = (0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19,
                   20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 39)


def block_map(indices, count):
    if set(indices) != set(range(28)) or count not in (28, 40):
        raise SnapRefError("unsupported_layout", adapter=len(set(indices)), blocks=count)
    return tuple(range(28)) if count == 28 else BASE_BLOCKS_29B


def detect_engine(engine):
    from backend.diffusion_engine.anima import Anima as AnimaEngine
    from backend.nn.anima import Anima as AnimaTransformer
    if not isinstance(engine, AnimaEngine):
        raise SnapRefError("anima_only")
    dm = engine.forge_objects.unet.get_model_object("diffusion_model")
    if not isinstance(dm, AnimaTransformer):
        raise SnapRefError("anima_only")
    block_map(range(28), len(dm.blocks))
    if dm.patch_spatial != 2 or dm.patch_temporal != 1 or dm.out_channels != 16:
        raise SnapRefError("unsupported_structure")
    for block in dm.blocks:
        attn = block.self_attn
        if block.x_dim != 2048 or (attn.n_heads, attn.head_dim) != (16, 128) or block.cross_attn.context_dim != 1024:
            raise SnapRefError("unsupported_structure")
    return dm
