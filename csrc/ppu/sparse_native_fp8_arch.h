#pragma once

// Select native PPU0015 definitions before including any actlize headers.
// This override is local to native FP8 translation units; existing kernels
// retain the build's ACOMPUTE_VERSION=10000 configuration.
#ifdef ACOMPUTE_VERSION
#undef ACOMPUTE_VERSION
#endif
#define ACOMPUTE_VERSION 10500
