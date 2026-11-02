/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#include "dense_decode.h"
#include "sparse_quantized.h"
#include "sparse_fwd.h"
#ifdef FLASHMLA_C_ENABLE_DECODE_SPARSE
#include "sparse_decode.h"
#endif

#ifndef FLASH_MLA_CPP_INFER_BUILD

#ifdef FLASH_MLA_STANDALONE_BUILD

#include <torch/python.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    namespace py = pybind11;
    using flashmla::quantized_attention::Config;
    py::class_<Config>(m, "QuantizedAttentionConfig",
        "Storage encoding, QK compute, quantized probabilities for PV, and dequantization scales.")
        .def(py::init([](std::string q_storage, std::string kv_storage, std::string qk_compute,
            std::string probability, Config::Scale q_scale, Config::Scale kv_scale,
            std::vector<int64_t> q_scale_block_shape, std::vector<int64_t> kv_scale_block_shape) {
            return Config{q_storage, kv_storage, qk_compute, probability, q_scale, kv_scale,
                          q_scale_block_shape, kv_scale_block_shape};
        }), py::arg("q_storage") = "auto", py::arg("kv_storage") = "auto",
            py::arg("qk_compute") = "e4m3", py::arg("probability") = "e4m3",
            py::arg("q_scale") = 1.0, py::arg("kv_scale") = 1.0,
            py::arg("q_scale_block_shape") = std::vector<int64_t>{},
            py::arg("kv_scale_block_shape") = std::vector<int64_t>{})
        .def_readwrite("q_storage", &Config::q_storage)
        .def_readwrite("kv_storage", &Config::kv_storage)
        .def_readwrite("qk_compute", &Config::qk_compute)
        .def_readwrite("probability", &Config::probability)
        .def_readwrite("q_scale", &Config::q_scale)
        .def_readwrite("kv_scale", &Config::kv_scale)
        .def_readwrite("q_scale_block_shape", &Config::q_scale_block_shape)
        .def_readwrite("kv_scale_block_shape", &Config::kv_scale_block_shape);
    m.def("sparse_quantized_fwd", &sparse_quantized_fwd);
    m.def("sparse_quantized_decode_fwd", &sparse_quantized_decode_fwd);

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
