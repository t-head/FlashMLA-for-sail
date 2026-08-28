/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#include "dense_decode.h"
#include "sparse_fwd.h"
#ifdef FLASHMLA_C_ENABLE_DECODE_SPARSE
#include "sparse_decode.h"
#endif

#ifndef FLASH_MLA_CPP_INFER_BUILD

#ifdef FLASH_MLA_STANDALONE_BUILD

#include <torch/python.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    // New aligned names
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
#ifdef FLASHMLA_C_ENABLE_DECODE_SPARSE
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
#endif
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
}

#else

#include <Python.h>
#include "kerutils/supplemental/pytorch_shim.h"

TORCH_LIBRARY(_flashmla_C, m) {
    // New aligned names
    m.def("dense_decode_fwd", make_pytorch_shim(&dense_attn_decode_interface));
    m.impl("dense_decode_fwd", torch::kCUDA, make_pytorch_shim(&dense_attn_decode_interface));

#ifdef FLASHMLA_C_ENABLE_DECODE_SPARSE
    m.def("sparse_decode_fwd", make_pytorch_shim(&sparse_attn_decode_interface));
    m.impl("sparse_decode_fwd", torch::kCUDA, make_pytorch_shim(&sparse_attn_decode_interface));
#endif

    m.def("sparse_prefill_fwd", make_pytorch_shim(&sparse_attn_prefill_interface));
    m.impl("sparse_prefill_fwd", torch::kCUDA, make_pytorch_shim(&sparse_attn_prefill_interface));
}

PyMODINIT_FUNC PyInit__flashmla_C() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_flashmla_C", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
#endif // FLASH_MLA_STANDALONE_BUILD

#endif // FLASH_MLA_CPP_INFER_BUILD
