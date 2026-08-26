/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "kerutils/common/common.h"

#include <tuple>

#if !defined(__HGGCCC_RTC__)
#include <hggc_runtime.h>
#endif


inline int get_current_device() {
    int device;
    CHECK_CUDA(hggcGetDevice(&device));
    return device;
}

inline std::tuple<int, int> get_compute_capability(int device) {
    int capability_major, capability_minor;
    CHECK_CUDA(hggcDeviceGetAttribute(&capability_major, hggcDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(hggcDeviceGetAttribute(&capability_minor, hggcDevAttrComputeCapabilityMinor, device));
    return {capability_major, capability_minor};
}

inline int get_num_sm(int device) {
    int multiprocessor_count;
    CHECK_CUDA(hggcDeviceGetAttribute(&multiprocessor_count, hggcDevAttrMultiProcessorCount, device));
    return multiprocessor_count;
}
