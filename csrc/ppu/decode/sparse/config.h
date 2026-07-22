#pragma once

namespace Config {

static constexpr int HEAD_DIM_V = 512;

}

enum NamedBarriers : int {
    sScale0Ready = 5,
    sScale1Ready = 6,
    sP0Ready = 7,
    rO1sP0sV0RIssued = 8,
    sMInitialized = 9,
    mGroup0 = 10,
    mGroup1 = 11,
    tma_copy_issued = 12
};
