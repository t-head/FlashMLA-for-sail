#pragma once
#include "ppu/sparse_native_fp8_attention.cuh"
#include "common.h"
#include "sparse_native_fp8_config.h"
#include <cmath>
#include <limits>
#include <variant>
#include <c10/cuda/CUDAException.h>

namespace flashmla::quantized_attention {

// Storage encoding is independent of the tensor's physical dtype (e.g. uint8
// can hold packed values). Compute formats describe the QK and PV operands.
// Only E4M3 compute is implemented today; unsupported configurations fail before
// launching. New formats can add a backend without adding a parallel public API.
struct Config {
    using Scale = std::variant<at::Tensor, double>;

    std::string q_storage = "auto";
    std::string kv_storage = "auto";
    std::string qk_compute = "e4m3";
    std::string probability = "e4m3";
    Scale q_scale = 1.0;
    Scale kv_scale = 1.0;
    // Tensor scales and logical block shapes reserve space for future backends.
    // The current E4M3 backend rejects them, including scalar tensor scales.
    std::vector<int64_t> q_scale_block_shape;
    std::vector<int64_t> kv_scale_block_shape;

    void validate(const at::Tensor &q, const at::Tensor &kv) const {
        TORCH_CHECK(qk_compute == "e4m3" && probability == "e4m3",
            "quantized attention: unsupported compute/probability format; available backend: e4m3");
        const auto storage_matches = [](const at::Tensor &t, const std::string &format) {
            if (format == "auto") return true;
            if (format == "e4m3")
                return t.scalar_type() == torch::kFloat8_e4m3fn || t.scalar_type() == torch::kUInt8;
            return format == "bf16" && t.scalar_type() == torch::kBFloat16;
        };
        TORCH_CHECK(storage_matches(q, q_storage) && storage_matches(kv, kv_storage),
            "quantized attention: unsupported or mismatched storage format");
        TORCH_CHECK(q_scale_block_shape.empty() && kv_scale_block_shape.empty() &&
                        std::holds_alternative<double>(q_scale) &&
                        std::holds_alternative<double>(kv_scale),
            "quantized attention: the E4M3 backend currently supports per-tensor scalar scales only");
        const double qs = std::get<double>(q_scale), ks = std::get<double>(kv_scale);
        TORCH_CHECK(std::isfinite(qs) && qs > 0 && std::isfinite(ks) && ks > 0,
            "quantized attention: scale must be finite and positive");
    }
};

} // namespace flashmla::quantized_attention

namespace flashmla::quantized_attention::detail {

inline float checked_scale(double value, const char *name) {
    // Kernels use FP32 with fast math: subnormal scales may be flushed to zero.
    TORCH_CHECK(std::isfinite(value) && value >= std::numeric_limits<float>::min() &&
                    value <= std::numeric_limits<float>::max(),
                "quantized attention: ", name, " must be finite and positive in the FP32 normal range");
    return static_cast<float>(value);
}

inline void check_inputs(const at::Tensor &q, const at::Tensor &kv, const at::Tensor &indices) {
    CHECK_DEVICE(q);
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && indices.dim() == 3,
                "sparse attention: invalid Q/KV/indices rank");
    for (const auto &tensor : {q, kv, indices}) {
        TORCH_CHECK(tensor.device() == q.device(), "sparse attention: input device mismatch");
        for (auto size : tensor.sizes())
            TORCH_CHECK(size <= INT_MAX, "sparse attention: shape exceeds kernel index range");
    }
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "sparse attention: indices must be int32");
    TORCH_CHECK(q.size(-2) > 0 && kv.size(-2) > 0 && q.size(-2) % kv.size(-2) == 0,
                "sparse attention: KV heads must divide Q heads");
    TORCH_CHECK(indices.size(0) == q.size(0) && indices.size(1) == kv.size(1),
                "sparse attention: incompatible indices shape");
}

inline void check_alignment(const at::Tensor &tensor, const char *name, int bytes = 16) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % bytes == 0,
                "sparse attention: ", name, " must be ", bytes, "-byte aligned");
}

inline void check_optional(const std::optional<at::Tensor> &tensor, const at::Tensor &q,
                           at::ScalarType dtype, at::IntArrayRef shape, const char *name,
                           int alignment = 1) {
    if (!tensor.has_value())
        return;
    TORCH_CHECK(tensor->device() == q.device(), "sparse attention: ", name, " device mismatch");
    TORCH_CHECK(tensor->scalar_type() == dtype, "sparse attention: ", name, " dtype mismatch");
    TORCH_CHECK(tensor->is_contiguous(), "sparse attention: ", name, " must be contiguous");
    TORCH_CHECK(tensor->sizes() == shape, "sparse attention: invalid ", name, " shape");
    check_alignment(*tensor, name, alignment);
}

inline at::Tensor output(const at::Tensor &q, const std::optional<at::Tensor> &out,
                         at::IntArrayRef shape, at::ScalarType dtype, int alignment = 1) {
    check_optional(out, q, dtype, shape, "out", alignment);
    return out.has_value() ? *out : torch::empty(shape, q.options().dtype(dtype));
}

} // namespace flashmla::quantized_attention::detail

namespace flashmla::sparse_native_fp8 {
using quantized_attention::detail::check_alignment;
using quantized_attention::detail::check_optional;

inline void check_device_capability() {
    auto [major, minor] = get_compute_capability(get_current_device());
    TORCH_CHECK(major == 8 && minor >= 9,
                "quantized E4M3 attention requires sm_89 (PPU ZW-M890P); got sm_", major, minor);
}

struct InputShape {
    int queries, heads, dim, kv_rows, width;
};

inline InputShape check_inputs(const at::Tensor &q, const at::Tensor &kv, const at::Tensor &indices,
                               int max_width = MaxTopk, bool allow_bf16_kv = false,
                               bool allow_strided_q = false) {
    quantized_attention::detail::check_inputs(q, kv, indices);
    const int n = q.size(0), h = q.size(1), d = q.size(2), rows = kv.size(0), w = indices.size(2);
    TORCH_CHECK(
        h > 0 && h % 8 == 0 && (d == 512 || d == 576),
        "quantized attention: heads must be a positive multiple of 8 and D must be 512 or 576");
    TORCH_CHECK(rows > 0 && kv.size(1) == 1 && kv.size(2) == d && indices.size(0) == n &&
                    indices.size(1) == 1,
                "quantized attention: incompatible Q/KV/indices shape");
    TORCH_CHECK(w > 0 && w % TileKeys == 0 && w <= max_width,
                "quantized attention: width must be a positive multiple of 64 and <= ", max_width);
    const auto is_e4m3 = [](const at::Tensor &tensor) {
        return tensor.scalar_type() == torch::kFloat8_e4m3fn ||
               tensor.scalar_type() == torch::kUInt8;
    };
    const bool q_bf16 = q.scalar_type() == torch::kBFloat16;
    const bool kv_bf16 = kv.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(q_bf16 || is_e4m3(q), "quantized attention: Q must be BF16 or E4M3");
    TORCH_CHECK(is_e4m3(kv) || (allow_bf16_kv && kv_bf16),
                "quantized attention: unsupported KV storage dtype");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32,
                "quantized attention: indices must be int32");
    CHECK_CONTIGUOUS(indices);
    check_alignment(q, "Q");
    TORCH_CHECK(q.stride(0) <= INT_MAX && q.stride(1) <= INT_MAX,
                "quantized attention: Q stride exceeds kernel index range");
    if (allow_strided_q && q_bf16) {
        TORCH_CHECK(q.stride(2) == 1 && q.stride(0) >= d && q.stride(1) >= d &&
                        q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0,
                    "quantized attention: Q rows must be dense and 16-byte aligned");
    } else {
        CHECK_CONTIGUOUS(q);
    }
    if (kv_bf16) {
        TORCH_CHECK(kv.stride(2) == 1 && kv.stride(0) >= d && kv.stride(0) % 2 == 0,
                    "quantized attention: BF16 KV rows must be dense and 4-byte aligned");
        check_alignment(kv, "KV", 4);
    } else {
        CHECK_CONTIGUOUS(kv);
        check_alignment(kv, "KV");
    }
    return {n, h, d, rows, w};
}

inline void check_auxiliary(const at::Tensor &q, int queries, int heads,
                            const std::optional<at::Tensor> &sink,
                            const std::optional<at::Tensor> &lengths) {
    check_optional(sink, q, torch::kFloat32, {heads}, "sink");
    check_optional(lengths, q, torch::kInt32, {queries}, "lengths");
}

struct AttentionOutput {
    at::Tensor out, lse;
    AttentionOutput(const at::Tensor &q, int queries, int heads,
                    const std::optional<at::Tensor> &output, int lse_queries = 0,
                    bool output_only = false)
        : out(quantized_attention::detail::output(
              q, output, {queries, heads, ValueDim}, torch::kBFloat16, 16)) {
        TORCH_CHECK(lse_queries >= 0 && (!lse_queries || queries % lse_queries == 0),
                    "quantized attention: invalid paged LSE query count");
        lse = output_only   ? torch::empty({0}, q.options().dtype(torch::kFloat32))
              : lse_queries ? torch::empty({queries / lse_queries, heads, lse_queries},
                                           q.options().dtype(torch::kFloat32))
                            : torch::empty({queries, heads}, q.options().dtype(torch::kFloat32));
    }
};

struct SplitWorkspace {
    at::Tensor output, lse, maxima;
    SplitWorkspace(const at::Tensor &out, const at::Tensor &final_lse, int splits,
                   const at::Tensor &final_max = at::Tensor()) {
        const int64_t n = out.size(0), h = out.size(1), d = out.size(2);
        output = splits > 1 ? torch::empty({n, splits, h, d}, out.options()) : out;
        lse = splits > 1 ? torch::empty({n, splits, h}, final_lse.options()) : final_lse;
        if (final_max.defined())
            maxima = splits > 1 ? torch::empty({n, splits, h}, final_max.options()) : final_max;
    }
};
} // namespace flashmla::sparse_native_fp8

namespace flashmla::quantized_attention {
using sparse_native_fp8::AttentionParams;
using sparse_native_fp8::AttentionOutput;
using sparse_native_fp8::SplitWorkspace;

enum class OutputMode { Decode, Prefill, OutputOnly };

// Tensor ownership stays at the API boundary. All native schedules consume the
// same prepared launch contract, without calling another public tensor API.
struct Request {
    at::cuda::CUDAGuard guard;
    AttentionParams params;
    at::Tensor output, lse, maxima;
    const at::Tensor &q;
    const std::optional<at::Tensor> &sink;

    Request(const at::Tensor &query, const at::Tensor &kv, const at::Tensor &indices,
            double sm_scale, const Config &config, int64_t d_v,
            const std::optional<at::Tensor> &attn_sink, const std::optional<at::Tensor> &lengths,
            const std::optional<at::Tensor> &out, OutputMode mode = OutputMode::Decode,
            int64_t lse_queries = 0)
        : guard(checked_device(query)), q(query), sink(attn_sink) {
        using namespace sparse_native_fp8;
        config.validate(q, kv);
        TORCH_CHECK(std::isfinite(sm_scale) && sm_scale > 0,
                    "quantized attention: softmax scale must be finite and positive");
        params.scale = detail::checked_scale(sm_scale * std::get<double>(config.q_scale) *
                                                std::get<double>(config.kv_scale) * 1.4426950408889634,
                                            "combined QK scale");
        params.os = detail::checked_scale(std::get<double>(config.kv_scale), "KV output scale");
        check_device_capability();
        const bool prefill = mode != OutputMode::Decode;
        const auto [n, h, d, rows, width] =
            check_inputs(q, kv, indices, prefill ? 2176 : MaxTopk, prefill, prefill);
        TORCH_CHECK(d_v == ValueDim, "quantized attention: d_v must be 512");
        check_auxiliary(q, n, h, attn_sink, lengths);
        TORCH_CHECK(lse_queries >= 0 && lse_queries <= INT_MAX,
                    "quantized attention: invalid paged LSE query count");
        AttentionOutput result(q, n, h, out, lse_queries, mode == OutputMode::OutputOnly);
        output = result.out;
        lse = result.lse;
        maxima = mode == OutputMode::Decode ? at::Tensor()
                 : mode == OutputMode::Prefill
                     ? torch::empty({n, h}, q.options().dtype(torch::kFloat32))
                     : torch::empty({0}, q.options().dtype(torch::kFloat32));
        auto &p = params;
        p.q = q.data_ptr();
        p.kv = kv.data_ptr();
        p.idx = indices.data_ptr<int>();
        p.sink = get_optional_tensor_ptr<float>(attn_sink);
        p.lengths = get_optional_tensor_ptr<int>(lengths);
        p.out = output.data_ptr();
        p.lse = mode == OutputMode::Decode || mode == OutputMode::Prefill ? lse.data_ptr<float>()
                                                                          : nullptr;
        p.maxima = mode == OutputMode::Prefill ? maxima.data_ptr<float>() : nullptr;
        p.dim = d;
        p.queries = n;
        p.heads = h;
        p.width = width;
        p.kvrows = rows;
        p.qbf = q.scalar_type() == torch::kBFloat16;
        p.qst = q.stride(0);
        p.qsh = q.stride(1);
        p.stream = at::cuda::getCurrentCUDAStream();
        p.lse_queries = lse_queries;
        const auto *properties = at::cuda::getCurrentDeviceProperties();
        p.compute_units = properties->multiProcessorCount;
        p.cache_bytes = properties->l2CacheSize;
    }

    static at::Device checked_device(const at::Tensor &q) {
        CHECK_DEVICE(q);
        return q.device();
    }

    void set_extra(const at::Tensor &extra, const at::Tensor &indices,
                   const std::optional<at::Tensor> &lengths) {
        using namespace sparse_native_fp8;
        const auto shape = check_inputs(q, extra, indices);
        check_optional(lengths, q, torch::kInt32, {params.queries}, "extra_lengths");
        TORCH_CHECK(int64_t(params.kvrows) + shape.kv_rows <= INT_MAX,
                    "quantized attention: too many KV rows");
        params.extra = extra.data_ptr();
        params.extra_idx = indices.data_ptr<int>();
        params.extra_lengths = get_optional_tensor_ptr<int>(lengths);
        params.extra_width = shape.width;
        params.extra_rows = shape.kv_rows;
    }
};

inline void execute_decode(AttentionParams params, const at::Tensor &out, const at::Tensor &lse,
                           const at::Tensor &maxima = at::Tensor()) {
    using namespace sparse_native_fp8;
    if (!params.queries)
        return;
    params.splits = choose_splits(params.queries, params.heads, params.width + params.extra_width,
                                  params.compute_units);
    params.two_stage =
        prefer_two_stages(params.queries, params.splits, params.heads, params.compute_units);
    SplitWorkspace workspace(out, lse, params.splits, maxima);
    params.part = workspace.output.data_ptr();
    params.plse = workspace.lse.data_ptr<float>();
    params.partial_maxima = maxima.defined() ? workspace.maxima.data_ptr<float>() : nullptr;
    if (params.extra)
        run_native_fp8_attention_extra(params);
    else
        run_native_fp8_attention_decode(params);
}

inline void execute_verify(Request &request, int64_t queries_per_request, bool shifted) {
    using namespace sparse_native_fp8;
    auto params = request.params;
    const int n = params.queries, h = params.heads, d = params.dim, w = params.width;
    TORCH_CHECK(queries_per_request > 1 && queries_per_request <= MaxVerifyQueries &&
                    n % queries_per_request == 0,
                "native verify: invalid queries/request");
    TORCH_CHECK(!params.qbf, "native verify: requires E4M3 Q");
    TORCH_CHECK((d == 512 || (d == 576 && !shifted)) && (h == 8 || h == 16),
                "native verify: unsupported heads/dim");
    TORCH_CHECK(shifted || !params.extra, "native verify: extra pool requires window union");
    params.qpr = queries_per_request;
    params.batch = n / params.qpr;
    params.padded_width = shifted ? ((w + params.qpr - 1 + 63) / 64) * 64 : w;
    const int union_width = params.padded_width + params.extra_width;
    TORCH_CHECK(union_width <= MaxTopk, "native verify: union exceeds supported width");
    if (!n)
        return;
    // The same allocation holds classifier indices, per-query masks and status
    // for ordinary and sliding-window verification, including graph replay.
    const int64_t union_elements = int64_t(params.batch) * union_width;
    const int64_t mask_elements = int64_t(params.batch) * params.qpr * (union_width / 32);
    auto workspace = torch::empty({union_elements + mask_elements + params.batch},
                                  request.q.options().dtype(torch::kInt32));
    params.united = workspace.data_ptr<int>();
    params.masks = reinterpret_cast<unsigned *>(params.united + union_elements);
    params.errors = params.united + union_elements + mask_elements;
    auto folded_sink =
        !shifted && request.sink.has_value() ? request.sink->repeat({params.qpr}) : at::Tensor();
    params.fold_sink = folded_sink.defined() ? folded_sink.data_ptr<float>() : nullptr;
    params.two_stage = prefer_two_stages(n, 1, h, params.compute_units);
    if (shifted)
        run_native_fp8_attention_window(params);
    else
        run_native_fp8_attention_verify(params);
}
} // namespace flashmla::quantized_attention

void prepare_sparse_prefill_fp8_kv(const void *, void *, int, int, int64_t, hggcStream_t);
void prepare_sparse_prefill_index_bounds(const int *, int *, int, int, int, hggcStream_t);

std::vector<at::Tensor> sparse_quantized_fwd(
    const at::Tensor &q,       // [s_q, h_q, d_qk=512|576] bf16/e4m3
    const at::Tensor &kv,      // [s_kv, h_kv=1, d_qk=512|576] bf16/e4m3
    const at::Tensor &indices, // [s_q, h_kv=1, topk] int32
    double sm_scale, const std::optional<flashmla::quantized_attention::Config> &quantization,
    int64_t d_v, const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length, const std::optional<at::Tensor> &out_opt,
    bool compute_aux, bool trusted_full_topk) {
    using namespace flashmla::quantized_attention;
    const auto mode = compute_aux ? OutputMode::Prefill : OutputMode::OutputOnly;
    Request request(q, kv, indices, sm_scale, quantization.value_or(Config{}), d_v, attn_sink,
                    topk_length, out_opt, mode);
    auto &launch = request.params;
    const int s_q = launch.queries, h_q = launch.heads, d_qk = launch.dim, s_kv = launch.kvrows,
              topk = launch.width;
    constexpr int h_kv = 1;
    const bool kv_is_bf16 = kv.scalar_type() == torch::kBFloat16;
    auto stream = launch.stream;
    at::Tensor kv_fp8 = kv;
    auto lengths = topk_length;
    if (kv_is_bf16 && s_q > 0) {
        kv_fp8 = torch::empty({s_kv, h_kv, d_qk}, kv.options().dtype(torch::kFloat8_e4m3fn));
        prepare_sparse_prefill_fp8_kv(kv.data_ptr(), kv_fp8.data_ptr(), s_kv, d_qk, kv.stride(0),
                                      stream);
        if (!lengths.has_value()) {
            lengths = torch::empty({s_q}, indices.options());
            prepare_sparse_prefill_index_bounds(indices.data_ptr<int>(), lengths->data_ptr<int>(),
                                                s_q, topk, s_kv, stream);
        }
    }
    launch.kv = kv_fp8.data_ptr();
    launch.lengths = get_optional_tensor_ptr<int>(lengths);
    launch.trusted_full_topk = trusted_full_topk;
    // Small full-head grids can amortize at least four key partitions and merge.
    const bool split_prefill =
        s_q > 0 && d_qk == 512 && h_q % 64 == 0 && !launch.qbf && !kv_is_bf16 &&
        flashmla::sparse_native_fp8::choose_splits(s_q, h_q, topk, launch.compute_units) >= 4;
    if (split_prefill)
        execute_decode(launch, request.output, request.lse,
                       compute_aux ? request.maxima : at::Tensor());
    else
        run_sparse_prefill_fp8_fwd(launch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {request.output, request.maxima, request.lse};
}

std::vector<at::Tensor> sparse_quantized_decode_fwd(
    const at::Tensor &q, const at::Tensor &kv, const at::Tensor &indices, double sm_scale,
    const std::optional<flashmla::quantized_attention::Config> &quantization, int64_t d_v,
    const std::optional<at::Tensor> &sink, const std::optional<at::Tensor> &lengths,
    const std::optional<at::Tensor> &out) {
    using namespace flashmla::quantized_attention;
    Request request(q, kv, indices, sm_scale, quantization.value_or(Config{}), d_v, sink, lengths,
                    out);
    execute_decode(request.params, request.output, request.lse);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {request.output, request.lse};
}

namespace flashmla::quantized_attention {

// Request grouping comes solely from Q[B,S,H,D]. Flattened [B*S,1,H,D]
// inputs remain independent queries; divisibility cannot recover grouping.
inline bool use_verify_union(int batch, int queries, int heads, int width, int compute_units,
                             int64_t cache_bytes, int dim, int64_t kv_rows) {
    using namespace flashmla::sparse_native_fp8;
    if (!(queries >= 2 && queries <= MaxVerifyQueries && width == MaxTopk &&
          (heads == 8 || heads == 16) && batch >= compute_units))
        return false;
    if (heads == 8)
        return true;
    const int folded_heads = queries * heads;
    const int64_t pool = kv_rows * dim;
    if (dim == 512 && pool > 0 && pool <= cache_bytes && folded_heads > 64 &&
        folded_heads % 64 == 32)
        return true;
    if (dim == 512 && folded_heads > 64 && folded_heads % 64 > 0 && folded_heads % 64 <= heads &&
        pool > cache_bytes && pool <= 16 * cache_bytes)
        return false;
    const int64_t working_set = int64_t(batch) * width * dim;
    const int tiles = (folded_heads + 63) / 64;
    const int64_t accessed = dim == 512 && kv_rows > 0 ? std::min(working_set, pool) : working_set;
    if (tiles == 1 && folded_heads < 64 && accessed < 8 * cache_bytes)
        return false;
    const bool streaming = dim == 576 && pool > 24 * cache_bytes;
    const int footprints = dim == 576 ? 3 : 4;
    const int reuse = working_set >= footprints * cache_bytes || streaming ? 3 : 4;
    return working_set >= 2 * cache_bytes && queries >= reuse * tiles;
}

inline std::optional<at::Tensor> flatten_lengths(const std::optional<at::Tensor> &lengths,
                                                 const at::Tensor &q, int64_t batch,
                                                 int64_t queries) {
    if (!lengths.has_value())
        return std::nullopt;
    TORCH_CHECK(lengths->device() == q.device() && lengths->scalar_type() == torch::kInt32,
                "quantized attention: lengths must be int32 on the Q device");
    if (lengths->numel() == batch && queries > 1)
        return lengths->reshape({batch}).repeat_interleave(queries).contiguous();
    return lengths->reshape({batch * queries}).contiguous();
}

inline std::vector<at::Tensor>
paged_attention(const at::Tensor &q, const at::Tensor &kv, int head_dim_v,
                const at::Tensor &indices, double scale, const std::optional<at::Tensor> &sink,
                const std::optional<at::Tensor> &lengths, const std::optional<at::Tensor> &extra,
                const std::optional<at::Tensor> &extra_indices,
                const std::optional<at::Tensor> &extra_lengths,
                const std::optional<at::Tensor> &out, bool enable_batch_invariant) {
    TORCH_CHECK_NOT_IMPLEMENTED(!enable_batch_invariant,
                                "enable_batch_invariant is not supported for native FP8 attention");
    CHECK_DEVICE(q);
    at::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(q.dim() == 4 && kv.dim() == 4 && kv.size(2) == 1,
                "quantized paged attention expects Q[B,S,H,D], cache[pages,page_size,1,D]");
    const int64_t batch = q.size(0), queries = q.size(1), heads = q.size(2), dim = q.size(3);
    TORCH_CHECK(kv.size(3) == dim && head_dim_v == 512,
                "quantized paged attention requires matching Q/KV dimensions and DV=512");
    TORCH_CHECK(batch * queries <= INT_MAX && heads <= INT_MAX,
                "quantized paged attention: query/head count exceeds kernel index range");
    TORCH_CHECK(indices.dim() == 3 && indices.size(0) == batch && indices.size(1) == queries,
                "quantized indices must be [batch,queries,sparse_width]");
    const int64_t width = indices.size(2);
    TORCH_CHECK(extra.has_value() == extra_indices.has_value(),
                "quantized extra cache and indices must be provided together");
    TORCH_CHECK(extra.has_value() || !extra_lengths.has_value(),
                "quantized extra lengths require an extra cache");
    std::optional<at::Tensor> flat_extra, flat_extra_ids;
    int64_t extra_width = 0;
    if (extra.has_value()) {
        TORCH_CHECK(extra->scalar_type() == kv.scalar_type() && extra->dim() == 4 &&
                        extra->size(2) == 1 && extra->size(3) == dim && extra_indices->dim() == 3 &&
                        extra_indices->size(0) == batch && extra_indices->size(1) == queries,
                    "quantized extra cache must use the same flat E4M3 layout");
        extra_width = extra_indices->size(2);
        flat_extra = extra->reshape({-1, 1, dim});
        flat_extra_ids = extra_indices->reshape({batch * queries, 1, extra_width}).contiguous();
    }
    const auto flat_q = q.reshape({batch * queries, heads, dim}).contiguous();
    const auto flat_kv = kv.reshape({-1, 1, dim});
    const auto flat_ids = indices.reshape({batch * queries, 1, width}).contiguous();
    std::optional<at::Tensor> flat_out;
    if (out.has_value()) {
        CHECK_SHAPE(out.value(), batch, queries, heads, head_dim_v);
        flat_out = out->view({batch * queries, heads, head_dim_v});
    }
    const auto flat_lengths = flatten_lengths(lengths, q, batch, queries);
    const auto flat_extra_lengths = flatten_lengths(extra_lengths, q, batch, queries);
    const auto *properties = at::cuda::getCurrentDeviceProperties();
    const int cus = properties->multiProcessorCount;
    const int64_t union_width = ((width + queries - 1 + 63) / 64) * 64 + extra_width;
    const int64_t folded_tiles = (queries * heads + 63) / 64;
    const bool q8 = q.scalar_type() == torch::kFloat8_e4m3fn || q.scalar_type() == torch::kUInt8;
    Request request(flat_q, flat_kv, flat_ids, scale, Config{}, head_dim_v, sink, flat_lengths,
                    flat_out, OutputMode::Decode, queries);
    if (extra.has_value())
        request.set_extra(*flat_extra, *flat_extra_ids, flat_extra_lengths);
    if (extra.has_value() && dim == 512 && heads == 8 && queries > 1 &&
        queries <= flashmla::sparse_native_fp8::MaxVerifyQueries && q8 && width % 64 == 0 &&
        union_width <= 2048 && batch * folded_tiles >= cus &&
        queries * heads * 4 >= folded_tiles * 64 * 3) {
        execute_verify(request, queries, true);
    } else if (!extra.has_value() && (dim == 512 || dim == 576) && q8 &&
               use_verify_union(batch, queries, heads, width, cus, properties->l2CacheSize, dim,
                                flat_kv.size(0))) {
        execute_verify(request, queries, false);
    } else if (!extra.has_value() || (dim == 512 && width + extra_width <= 2048)) {
        execute_decode(request.params, request.output, request.lse);
    } else {
        // Two independent pools with one final normalization. Pool views reuse
        // the validated request; they do not re-enter tensor API validation.
        auto run_pool = [&](bool secondary) {
            auto params = request.params;
            AttentionOutput result(flat_q, params.queries, params.heads, std::nullopt);
            if (secondary) {
                params.kv = params.extra;
                params.idx = params.extra_idx;
                params.lengths = params.extra_lengths;
                params.width = params.extra_width;
                params.kvrows = params.extra_rows;
            }
            params.extra = nullptr;
            params.extra_idx = nullptr;
            params.extra_lengths = nullptr;
            params.extra_width = params.extra_rows = 0;
            params.sink = nullptr;
            params.out = result.out.data_ptr();
            params.lse = result.lse.data_ptr<float>();
            params.lse_queries = 0;
            execute_decode(params, result.out, result.lse);
            return result;
        };
        const auto main_result = run_pool(false), extra_result = run_pool(true);
        auto part = at::stack({main_result.out, extra_result.out}, 1);
        auto plse = at::stack({main_result.lse, extra_result.lse}, 1);
        sparse_native_fp8::MergeParams merge;
        merge.part = part.data_ptr();
        merge.plse = plse.data_ptr<float>();
        merge.out = request.params.out;
        merge.lse = request.params.lse;
        merge.sink = request.params.sink;
        merge.tokens = request.params.queries;
        merge.splits = 2;
        merge.heads = heads;
        merge.stream = request.params.stream;
        merge.lse_queries = queries;
        run_native_fp8_attention_merge(merge);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {request.output.view({batch, queries, heads, head_dim_v}),
            request.lse.view({batch, heads, queries})};
}

} // namespace flashmla::quantized_attention
