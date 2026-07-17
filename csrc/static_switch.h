// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define IS_PAGE_POWER2(PAGE, E_PAGE,...)   \
  [&] {                                    \
    auto is_pow2 = [](int v) { return v > 0 && (v & (v - 1)) == 0; }; \
    bool page_ok = (PAGE == 0) || is_pow2(PAGE);   \
    bool epage_ok = (E_PAGE == 0) || is_pow2(E_PAGE); \
    if (page_ok && epage_ok && (is_pow2(PAGE) || is_pow2(E_PAGE))) { \
      constexpr static bool kPagePow2 = true;  \
      return __VA_ARGS__();                \
    } else {                                    \
      constexpr static bool kPagePow2 = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define SEQLENG_SWITCH_ALIGN(SEQLENG, ...)   \
  [&] {                                    \
    if (SEQLENG <= 16) {                  \
      constexpr static int kBlockM = 16;  \
      return __VA_ARGS__();                \
    } else if (SEQLENG <= 32) {            \
      constexpr static int kBlockM = 32;  \
      return __VA_ARGS__();                \
    } else {                              \
      constexpr static int kBlockM = 64;  \
      return __VA_ARGS__();               \
    }                                      \
  }()

#define SEQLENG_SWITCH(SEQLENG, ...)   \
  [&] {                                    \
    if (SEQLENG <= 16) {                  \
      constexpr static int kBlockM = 16;  \
      return __VA_ARGS__();                \
    } else if (SEQLENG <= 32) {            \
      constexpr static int kBlockM = 32;  \
      return __VA_ARGS__();                \
    } else if (SEQLENG <= 48) {            \
      constexpr static int kBlockM = 48;  \
      return __VA_ARGS__();               \
    } else {                              \
      constexpr static int kBlockM = 64;  \
      return __VA_ARGS__();               \
    }                                      \
  }()

#define THREADS_SWITCH(BLOCKS, ...)   \
  [&] {                                    \
    if (BLOCKS <= 32) {                   \
      constexpr static int kBlockM = 1;  \
      return __VA_ARGS__();                \
    } else if (BLOCKS <= 64) {            \
      constexpr static int kBlockM = 2;  \
      return __VA_ARGS__();               \
    } else {                              \
      constexpr static int kBlockM = 4;  \
      return __VA_ARGS__();               \
    }                                      \
  }()

#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...) \
  [&] {                                              \
    if (NUM_SPLITS <= 32) {                          \
      constexpr static int NAME = 32;                \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 64) {                   \
      constexpr static int NAME = 64;                \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 96) {                   \
      constexpr static int NAME = 96;                \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 128) {                  \
      constexpr static int NAME = 128;               \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 160) {                  \
      constexpr static int NAME = 160;               \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 320) {                  \
      constexpr static int NAME = 320;               \
      return __VA_ARGS__();                          \
    } else {                                         \
      FLASH_ASSERT(false);                           \
    }                                                \
  }()

#define DISPATCH_HEAD_DIM(HEAD_DIM, CONSTEXPR_NAME, ...) \
[&] () { \
    if (HEAD_DIM == 576) { \
        static constexpr int CONSTEXPR_NAME = 576; \
        return __VA_ARGS__(); \
    } else if (HEAD_DIM == 512) { \
        static constexpr int CONSTEXPR_NAME = 512; \
        return __VA_ARGS__(); \
    } else { \
        TORCH_CHECK(false, "Unsupported head_dim_qk: ", HEAD_DIM); \
    } \
} ();

#define DISPATCH_BOOLEAN_FLAG(FLAG, CONSTEXPR_NAME, ...) \
    [&] () { \
        if (FLAG) { \
            static constexpr bool CONSTEXPR_NAME = true; \
            return __VA_ARGS__(); \
        } else { \
            static constexpr bool CONSTEXPR_NAME = false; \
            return __VA_ARGS__(); \
        } \
    } ();