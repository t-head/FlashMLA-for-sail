/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#include "dense_decode.h"
#include "sparse_native_fp8.h"
#include "sparse_fwd.h"
#ifdef FLASHMLA_C_ENABLE_DECODE_SPARSE
#include "sparse_decode.h"
#endif

#ifndef FLASH_MLA_CPP_INFER_BUILD

#ifdef FLASH_MLA_STANDALONE_BUILD

#include <torch/python.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("native_fp8_verify_fwd", &native_fp8_verify_fwd,
        pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("indices"),
        pybind11::arg("queries_per_request"), pybind11::arg("sm_scale"),
        pybind11::arg("sink"), pybind11::arg("lengths"), pybind11::arg("out"),
        pybind11::arg("shifted") = false, pybind11::arg("extra") = pybind11::none(),
        pybind11::arg("extra_indices") = pybind11::none(), pybind11::arg("extra_lengths") = pybind11::none(),
        pybind11::arg("lse_queries") = 0);
    m.def("native_fp8_extra_fwd", &native_fp8_extra_fwd,
        pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("indices"),
        pybind11::arg("extra"), pybind11::arg("extra_indices"), pybind11::arg("sm_scale"),
        pybind11::arg("sink"), pybind11::arg("lengths"), pybind11::arg("extra_lengths"),
        pybind11::arg("out"), pybind11::arg("lse_queries") = 0);
    m.def("native_fp8_merge", &native_fp8_merge);
    m.def("native_fp8_lse_to_natural", &native_fp8_lse_to_natural);
    m.def("sparse_decode_fp8_fwd", &sparse_decode_fp8_fwd,
        pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("indices"),
        pybind11::arg("sm_scale"), pybind11::arg("q_scale"), pybind11::arg("kv_scale"),
        pybind11::arg("d_v"),
        pybind11::arg("attn_sink") = pybind11::none(),
        pybind11::arg("topk_length") = pybind11::none(), pybind11::arg("out") = pybind11::none(),
        pybind11::arg("lse_queries") = 0);
    m.def("sparse_prefill_fp8_fwd", &sparse_prefill_fp8_fwd);
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
    m.def("sparse_decode_fp8_fwd", make_pytorch_shim(&sparse_decode_fp8_fwd));
    m.impl("sparse_decode_fp8_fwd", torch::kCUDA, make_pytorch_shim(&sparse_decode_fp8_fwd));
    m.def("sparse_prefill_fp8_fwd", make_pytorch_shim(&sparse_prefill_fp8_fwd));
    m.impl("sparse_prefill_fp8_fwd", torch::kCUDA, make_pytorch_shim(&sparse_prefill_fp8_fwd));
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
