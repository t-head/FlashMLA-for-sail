// Common paired-block traversal. The actual iteration is wg_iteration_body.cuh.
#pragma once
#define FLASHMLA_WI_PAIR_RANGE(CONDITION, STEP) \
    for (; CONDITION; block_idx += 2) { STEP; }
#define FLASHMLA_WI_PAIR_TAIL(CONDITION, STEP) \
    if (CONDITION) { STEP; }
#define FLASHMLA_WI_PAIR_FINAL(STEP) STEP;
