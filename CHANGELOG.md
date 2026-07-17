# FlashMLA 2.0.0 for PPU CHANGELOG

## [2.0.0] - 2026-06

This release adds DeepSeek V4 support for PPU hardware with major API refactoring for metadata initialization and performance optimizations for sparse attention.

### Interface Changes

#### `flash_mla_interface.py`

**`get_mla_metadata()` - Simplified to Stub**

- Now returns a `FlashMLASchedMeta` placeholder instead of computing metadata immediately
- Actual metadata generated lazily on first `flash_mla_with_kvcache()` call
- Accepts `*args, **kwargs` for backward compatibility

**`flash_mla_with_kvcache()` - Enhanced API**

*New Parameters:*
- `attn_sink`: Attention sink scaling for DeepSeek V4
- `extra_k_cache` / `extra_indices_in_kvcache`: Extra KV cache for MODEL1 support
- `topk_length` / `extra_topk_length`: Per-query variable topk for efficiency
- `out`: Pre-allocated output buffer

*Changes:*
- `block_table` and `cache_seqlens` now optional (for sparse attention)
- `tile_scheduler_metadata` type changed to `FlashMLASchedMeta`
- `num_splits` must be `None`

**`flash_mla_sparse_fwd()` - New Features**

- Added `attn_sink`, `topk_length`, and `out` parameters
- Head dimension `d_v` fixed at 512

### Performance Optimizations

- New tile sizes: kBlockM=16&32 for head dims 8, 16, 32
- Memory dependency reduction for scale operations
- Eliminated DSA decode stack overhead
- Optimized FP8 to BF16 conversion
