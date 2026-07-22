/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <hggc_runtime.h>

#include "kerutils/common/common.h"

static inline bool is_sm89_or_newer() {
    int device = 0;
    hggcGetDevice(&device);
    int major = 0, minor = 0;
    hggcDeviceGetAttribute(&major, hggcDevAttrComputeCapabilityMajor, device);
    hggcDeviceGetAttribute(&minor, hggcDevAttrComputeCapabilityMinor, device);
    return (major > 8) || (major == 8 && minor >= 9);
}
