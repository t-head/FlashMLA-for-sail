/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// A narrow build target for sparse-prefill kernel development.  It keeps the
// production operator and ABI while avoiding unrelated dense/decode host API
// compilation when validating a sparse-prefill-only change.
#include "sparse_fwd.h"

#include <torch/python.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA sparse-prefill development module";
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
}
