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

#define SEQLENG_SWITCH(SEQLENG, ...)   \
  [&] {                                    \
    if (SEQLENG <= 8) {                   \
      constexpr static int kBlockM = 8;  \
      return __VA_ARGS__();                \
    } else if (SEQLENG <= 16) {            \
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
