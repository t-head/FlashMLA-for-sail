/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <hggc_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "kerutils/common/common.h"

static inline bool is_sm89_or_newer() {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    return (dprops->major > 8) || (dprops->major == 8 && dprops->minor >= 9);
}
