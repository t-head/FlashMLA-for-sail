#pragma once

#include <cstdio>
#include <cstdlib>

#include <hggc_runtime.h>

#define FLASH_DEVICE_ASSERT(cond)                                                 \
    do {                                                                          \
        if (not (cond)) {                                                        \
            printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
            asm("ppu.trap;");                                                        \
        }                                                                         \
    } while(0)

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        hggcError_t status_ = call;                                               \
        if (status_ != hggcSuccess) {                                             \
            fprintf(stderr, "HGGC error (%s:%d): %s\n", __FILE__, __LINE__,      \
                    hggcGetErrorString(status_));                                 \
            exit(1);                                                             \
        }                                                                         \
    } while (0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(hggcGetLastError())

#define FLASH_ASSERT(cond)                                                       \
    do {                                                                          \
        if (not (cond)) {                                                        \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                             \
        }                                                                         \
    } while(0)

namespace kerutils {}

namespace ku = kerutils;
