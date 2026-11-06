__version__ = "2.1.0"

from flash_mla.flash_mla_interface import (
    FlashMLASchedMeta,
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_mla_sparse_fwd,
    flash_mla_sparse_quantized_fwd,
    flash_mla_sparse_quantized_decode_fwd,
)

from flash_mla import fused_norm_rope_attn_rope_cast
from flash_mla_cuda import QuantizedAttentionConfig

__all__ = [
    "QuantizedAttentionConfig",
    "flash_mla_sparse_quantized_fwd",
    "flash_mla_sparse_quantized_decode_fwd",
    "FlashMLASchedMeta",
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "flash_mla_sparse_fwd",
    "fused_norm_rope_attn_rope_cast",
]
