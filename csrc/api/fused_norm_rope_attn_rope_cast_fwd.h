#pragma once

#include <torch/extension.h>
#include <optional>
#include <vector>

std::vector<at::Tensor> fused_norm_rope_attn_rope_cast_fwd(
    const at::Tensor& q, const at::Tensor& kv, const at::Tensor& indices,
    double sm_scale, int64_t d_v, const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length, bool enable_q_norm,
    double rms_norm_eps, const at::Tensor& token_positions, bool is_rope_neox_style,
    int64_t rope_dim, const at::Tensor& cos_sin_cache, int64_t n_wv_group,
    int64_t num_per_channels, bool use_tma_aligned_col_major_sf,
    bool round_sf, bool use_packed_ue8m0);

std::vector<at::Tensor> fused_norm_rope_attn_rope_cast_decode(
    const at::Tensor& q, const at::Tensor& kv, const at::Tensor& indices,
    double sm_scale, int64_t d_v, const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length, const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length, bool enable_q_norm,
    double rms_norm_eps, const at::Tensor& token_positions, bool is_rope_neox_style,
    int64_t rope_dim, const at::Tensor& cos_sin_cache, int64_t n_wv_group,
    int64_t num_per_channels, bool use_tma_aligned_col_major_sf,
    bool round_sf, bool use_packed_ue8m0);

std::vector<at::Tensor> permute_q_b_proj(
    const at::Tensor& q_b_proj, const at::Tensor& scale_factors, int64_t h_q, int64_t d_q);

std::vector<at::Tensor> permute_wv_proj(
    const at::Tensor& wv_proj, const at::Tensor& scale_factors,
    int64_t wv_group_size, int64_t d_o);
