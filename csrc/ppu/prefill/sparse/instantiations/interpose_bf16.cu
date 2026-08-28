#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <link.h>

// The device-only interposer intentionally has no Torch/CUDA-wrapper headers.
// Dispatch errors are impossible after the host API's shape validation, but
// retain fail-closed behavior for a mismatched caller.
#define TORCH_CHECK(condition, ...)        \
    do {                                   \
        if (!(condition)) { std::abort(); } \
    } while (false)

#ifndef CUDART_LN2_F
#define CUDART_LN2_F 0.69314718055994530942f
#endif

#include "params.h"
#include "kerutils/common/common.h"
#include "kerutils/common/static_switch.h"
#include "prefill/sparse/sparse_prefill_std.cuh"

// The production 2.0.0 FlashMLA wheel exports this template instantiation as
// a weak ELF symbol.  A strong definition lets an exact-shape kernel build be
// validated with LD_PRELOAD without replacing the wheel's dense/decode APIs.
// Keep the ABI name explicit: it is checked against the target wheel before
// the interposer is admitted by tools/build_h8_interposer.sh.
extern "C" __attribute__((visibility("default")))
void flashmla_sparse_prefill_dispatch_bf16(SparsePrefillParams& params)
    asm("_Z31run_sparse_prefill_fwd_dispatchIN7cutlass10bfloat16_tEEvR19SparsePrefillParams");

extern "C" __attribute__((visibility("default")))
void flashmla_sparse_prefill_h8_run(SparsePrefillParams* params);

namespace {

constexpr const char* kDispatchSymbol =
    "_Z31run_sparse_prefill_fwd_dispatchIN7cutlass10bfloat16_tEEvR19SparsePrefillParams";

struct FlashMlaModule {
    const char* path = nullptr;
};

int find_flash_mla_module(dl_phdr_info* info, size_t, void* opaque) {
    if (info->dlpi_name != nullptr &&
        std::strstr(info->dlpi_name, "flash_mla_cuda") != nullptr) {
        static_cast<FlashMlaModule*>(opaque)->path = info->dlpi_name;
        return 1;
    }
    return 0;
}

using DispatchFn = void (*)(SparsePrefillParams&);

DispatchFn resolve_original_dispatch() {
    FlashMlaModule module;
    dl_iterate_phdr(find_flash_mla_module, &module);
    if (module.path == nullptr) {
        std::fprintf(stderr, "flashmla-h8-interposer: flash_mla_cuda is not loaded\n");
        std::abort();
    }

    // PyTorch imports extension modules with RTLD_LOCAL.  RTLD_NEXT therefore
    // cannot see the wheel's weak dispatch symbol; query that object's own
    // handle so non-h8 calls retain the exact production implementation.
    void* handle = dlopen(module.path, RTLD_NOW | RTLD_NOLOAD);
    if (handle == nullptr) {
        std::fprintf(stderr, "flashmla-h8-interposer: dlopen(%s) failed: %s\n",
                     module.path, dlerror());
        std::abort();
    }
    dlerror();
    void* symbol = dlsym(handle, kDispatchSymbol);
    const char* error = dlerror();
    if (symbol == nullptr || error != nullptr) {
        std::fprintf(stderr, "flashmla-h8-interposer: dlsym(%s) failed: %s\n",
                     module.path, error ? error : "unknown error");
        std::abort();
    }
    return reinterpret_cast<DispatchFn>(symbol);
}

void run_h8(SparsePrefillParams& params) {
    constexpr int kBlockM = 8;
    constexpr int kBlockN = 64;
    constexpr int AtomLayoutQ = 1;
    constexpr int AtomLayoutP = 1;
    constexpr int kNwarps0 = AtomLayoutQ * (kBlockN / 16);
    constexpr int kNwarps = 16;
    // Match the existing PPU10 M8 contract: load Q with the M8-specific TSM
    // atom, retain it in registers, then let the KV pipeline reuse Q's SMEM.
    constexpr bool Is_Q_in_regs = true;
    constexpr bool Share_Q_K_smem = true;
    constexpr bool USE_MMA_M8 = true;

    DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
        DISPATCH_BOOLEAN_FLAG(params.topk_length != nullptr, HAVE_TOPK_LENGTH, [&]() {
            run_flash_sparse_prefill_fwd<Flash_fwd_kernel_traits<
                HEAD_DIM_QK, kBlockM, kBlockN, kNwarps,
                Is_Q_in_regs, Share_Q_K_smem, cutlass::bfloat16_t, 512,
                1, USE_MMA_M8, AtomLayoutQ, AtomLayoutP,
                kBlockN, 2, kNwarps0>, HAVE_TOPK_LENGTH>(params);
        });
    });
}

}  // namespace

extern "C" void flashmla_sparse_prefill_dispatch_bf16(SparsePrefillParams& params) {
    if (params.h_q != 8) {
        static DispatchFn next = resolve_original_dispatch();
        next(params);
        return;
    }
    run_h8(params);
}

extern "C" void flashmla_sparse_prefill_h8_run(SparsePrefillParams* params) {
    if (params == nullptr || params->h_q != 8) {
        std::abort();
    }
    run_h8(*params);
}
