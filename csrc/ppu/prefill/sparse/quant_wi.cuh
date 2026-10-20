/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cute/util/debug.hpp>
#include "sparse_quant_prefill.h"
#include "quant_traits.h"
#include "acc_vreg_fraga.h"
#include "kernel_traits.h"
#include "utils.h"
#include "kerutils/device/ppu/softmax.cuh"
#include "kerutils/device/ppu/mask.cuh"
#include "kerutils/device/ppu/dequant.cuh"
#include "wg_common_ops.cuh"
#include "wg_iteration_schedule.cuh"
#if defined(FLASHMLA_V40_PREFILL_MODEL1)
#include "v40_model1_format.cuh"
#endif

namespace flashmla::v41prefill {
using namespace cute;
using cutlass::arch::NamedBarrier;

static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;

using ::flashmla::sparse_wi::get_AorC_row_idx;

using ::flashmla::sparse_wi::convert_acc;

// Exact E2M1 FP4 magnitudes and signs for eight nibbles in a word.
// Share sign expansion across four BF16x2 output pairs.
__forceinline__ __device__ uint4 v41_prefill_fp4_word_shared_sign(uint32_t word) {
    const uint32_t hsel=((word>>1)&0x77777777u)|(word&0x11111111u);
    const uint32_t selectors0=word&0x7777u;
    const uint32_t selectors1=(word>>16)&0x7777u;
    const uint32_t low0=__byte_perm(0xc0800000u,0xc0804000u,selectors0);
    uint32_t high0;
    asm volatile("ppu.prmt.b32 %0, %1, %2, %3;\n"
        : "=r"(high0) : "n"(0x40403f00u), "n"(0xc0c0bf80u), "r"(hsel));
    const uint32_t low1=__byte_perm(0xc0800000u,0xc0804000u,selectors1);
    const uint32_t high1=__byte_perm(0x40403f00u,0xc0c0bf80u,hsel>>16);
    return uint4{__byte_perm(low0,high0,0x5140),__byte_perm(low0,high0,0x7362),
    __byte_perm(low1,high1,0x5140),__byte_perm(low1,high1,0x7362)};
}
// Each producer owns an AoS24 slot in the unused shared-output tail.
#if defined(FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32) || defined(FLASHMLA_V41_QUANT_DECODE_SCOPE_ROW32)
// Row-base byte offset and scale-plane delta; only the issue site forms pointers.
struct V41QuantRowAddress32 {
    uint32_t row_payload_offset;
    uint32_t scale_delta;
};
static_assert(sizeof(V41QuantRowAddress32)==8);
#endif
#if defined(FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32)
#if !defined(FLASHMLA_V41_QUANT_ADDRESS32)
#error "Offset/delta state is only valid in a bounded32 TU"
#endif
template<typename T>
using V41PrecomputedKAddress=std::conditional_t<T::kQuantCache && !T::kFP4Cache,V41QuantRowAddress32,typename T::InputT*>;
using V41QuantLoadRowAddress=V41QuantRowAddress32;
#elif defined(FLASHMLA_V41_QUANT_DECODE_SCOPE_ROW32)
template<typename T>
using V41PrecomputedKAddress=std::conditional_t<T::kDecodeFP8ScopeRow32,V41QuantRowAddress32,typename T::InputT*>;
#endif
#if defined(FLASHMLA_V41_QUANT_DECODE_PAGE_BASE64)
struct V41QuantRowAddress64 {
    const uint8_t *payload_row;
    int64_t scale_delta;
};
static_assert(sizeof(V41QuantRowAddress64)==16);
using V41QuantLoadRowAddress=V41QuantRowAddress64;
#endif

// Preserve globally bounded Prefill TUs while allowing only proven Decode cache
// scopes to narrow their leaf KV/scale arithmetic. Base pointers remain 64-bit.
template<typename T>
inline constexpr bool v41_cache_offset32_enabled =
#if defined(FLASHMLA_V41_QUANT_ADDRESS32)
    true;
#else
    T::kCacheOffset32;
#endif

// Cache policy only: decoded math/layout/state are inherited from the parent.
struct QuantDecodeFixedExtraFP4CacheAdapter;
template<typename T>
struct QuantDecodeNativeFP4SteadyTraits : T {
    static_assert(T::kDecode && T::kMixedFP4 && T::kHeadDim==512);
    using CacheAdapter=QuantDecodeFixedExtraFP4CacheAdapter;
    // This child is instantiated only after every future index key is extra.
    static constexpr bool kDecodeFixedExtraIndex=true;
    static constexpr bool kFP4Cache=true;
    static constexpr bool kMixedFP4=false;
};

// UInt32 payload/scale carry and decoded math remain the parent FP8 types.
struct QuantDecodeFixedExtraFP8CacheAdapter;
template<typename T>
struct QuantDecodeFixedExtraFP8SteadyTraits : T {
    static_assert(T::kDecode && !T::kMixedFP4 && !T::kFP4Cache);
    static_assert(T::kDecodeFP8ScopeRow32 && !T::kDecodePageBase64);
    using CacheAdapter=QuantDecodeFixedExtraFP8CacheAdapter;
    static constexpr bool kDecodeFixedExtraIndex=true;
};

static constexpr bool kFP4AsyncEarly = true;
static constexpr int V41_FP4_ASYNC_STAGE_BASE=248960;
static constexpr int V41_FP4_ASYNC_STAGE_STRIDE=24;
template<typename T>
__forceinline__ __device__ uint8_t *v41_prefill_fp4_async_slot() {
    static_assert(T::kQuantCache && T::kFP4Cache && T::kHeadDim==512);
    static_assert(sizeof(typename T::SharedMemoryPlan)<=V41_FP4_ASYNC_STAGE_BASE);
    static_assert(V41_FP4_ASYNC_STAGE_BASE+T::NUM_THREADS*V41_FP4_ASYNC_STAGE_STRIDE
                  <=sizeof(typename T::SharedMemoryOutPut));
    static_assert(sizeof(typename T::SharedMemoryOutPut)==262144);
    static_assert(V41_FP4_ASYNC_STAGE_BASE%128==0 && V41_FP4_ASYNC_STAGE_STRIDE%8==0);
    extern __shared__ char wksp_buf[];
    return reinterpret_cast<uint8_t*>(wksp_buf)+V41_FP4_ASYNC_STAGE_BASE+
        threadIdx.x*V41_FP4_ASYNC_STAGE_STRIDE;
}
template<int S,int E,typename T>
__forceinline__ __device__ void v41_prefill_fp4_async_issue(
    const typename T::Params &params,int tidx,int token_idx,bool valid) {
    static_assert(E-S==4 && (S==0 || S==4));
    const int old_lane=tidx&7;
    const int page=valid?token_idx>>__builtin_ctz(params.quant_page_size):0;
    const int off=valid?token_idx&(params.quant_page_size-1):0;
    const uint8_t *block;
    const uint8_t *data;
    const uint8_t *scales;
    if constexpr(v41_cache_offset32_enabled<T>) {
        block=reinterpret_cast<const uint8_t*>(params.kv)+
            uint32_t(page)*uint32_t(params.quant_block_stride);
        data=block+uint32_t(off)*256+old_lane*8;
        scales=block+uint32_t(params.quant_page_size)*256+uint32_t(off)*32;
    } else {
        block=reinterpret_cast<const uint8_t*>(params.kv)+
            int64_t(page)*params.quant_block_stride;
        data=block+int64_t(off)*256+old_lane*8;
        scales=block+int64_t(params.quant_page_size)*256+int64_t(off)*32;
    }
    auto *slot=v41_prefill_fp4_async_slot<T>();
    // This issue touches only its own private tail, never a live BF16 bank.
    #pragma unroll
    for(int group=0;group<2;++group) {
        const int tile_pair=S+2*group;
        __ppu_pipeline_memcpy_async_zfill(slot+group*8,
            data+tile_pair*32,8,valid?0:8,1);
        const int scale_idx=tile_pair*4+old_lane;
        __ppu_pipeline_memcpy_async_zfill(slot+16+group*4,
            scales+(scale_idx&~3),4,valid?0:4,1);
    }
    __pipeline_commit();
}
template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
__forceinline__ __device__ void v41_prefill_fp4_async_finish(RawK &raw_sK,int tidx) {
    static_assert(E-S==4);
    const int row=tidx/8,old_lane=tidx&7;
    const int paired_base=(S/2)*4096+(row>>4)*2048+(row&15)*64
        +(old_lane>>2)*1024+(((old_lane&3)^((row>>1)&3))*16);
    const int first_chunk=paired_base+(row&1)*8;
    const int second_chunk=paired_base+((row&1)^1)*8;
    auto *dst=cute::raw_pointer_cast(raw_sK.data());
    // Retain the exact baseline reader drain at its original finish point.
    if constexpr(FRONT_DRAIN) {
        cutlass::arch::NamedBarrier::sync(256,
            static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
    }
    __pipeline_wait_prior(0);
    const auto *slot=v41_prefill_fp4_async_slot<T>();
    #pragma unroll
    for(int group=0;group<2;++group) {
        const uint2 packed=*reinterpret_cast<const uint2*>(slot+group*8);
        {
            const uint32_t scale_word=*reinterpret_cast<const uint32_t*>(slot+16+group*4);
            const int scale_idx=(S+2*group)*4+old_lane;
            const uint8_t raw=uint8_t(scale_word>>((scale_idx&3)*8));
            const float scale=float(*reinterpret_cast<const __nv_fp8_e4m3*>(&raw));
            const __nv_bfloat162 scale2=__float2bfloat162_rn(scale);
            const uint32_t words[2]={packed.x,packed.y};
            #pragma unroll
            for(int half=0;half<2;++half) {
                const uint4 word_bits=v41_prefill_fp4_word_shared_sign(words[half]);
                const uint32_t decoded[4]={word_bits.x,word_bits.y,word_bits.z,word_bits.w};
                uint32_t out_bits[4]{};
                #pragma unroll
                for(int pair=0;pair<4;++pair) {
                    const uint32_t bits=decoded[pair];
                    const __nv_bfloat162 out=__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&bits),scale2);
                    out_bits[pair]=*reinterpret_cast<const uint32_t*>(&out);
                }
                const int pos=(half==0?first_chunk:second_chunk)+group*4096;
                *reinterpret_cast<uint4*>(dst+pos)=uint4{out_bits[0],out_bits[1],out_bits[2],out_bits[3]};
            }
        }
    }
#if defined(FLASHMLA_V41_QUANT_DECODED_STORE_RELEASE)
    __threadfence_block();
    asm volatile("" ::: "memory");
#else
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
#endif
}

// UE8 raw0 is 2^-127; raw255 is canonical NaN. Invalid tokens still use bits0.
__forceinline__ __device__ uint32_t v41_prefill_e8m0_scale2_bits(uint8_t raw) {
    uint32_t bits=uint32_t(raw)*0x00800080u;
    bits=raw==0?0x00400040u:bits;
    bits=raw==255?0x7fff7fffu:bits;
    return bits;
}

template<int S,int E,typename T,bool FRONT_DRAIN=true,typename RawK>
__forceinline__ __device__ void v41_prefill_load_fp8(
    const typename T::Params &params,RawK &raw_sK,int tidx,int token_idx,bool valid) {
#if defined(FLASHMLA_V40_PREFILL_MODEL1)
    if constexpr(T::kV40Cache) {
        v40_prefill_load_fp8<S,E,T,FRONT_DRAIN>(params,raw_sK,tidx,token_idx,valid);
        return;
    }
#endif
    static_assert(!T::kFP4Cache && T::kBlockM==128 && T::kBlockN==32 && T::kHeadDim==512 && T::kUsePairedCubes);
    const int row=tidx/8,old_lane=tidx&7;
    // Flatten the paired64-dim and16-row geometry once perproducer.
    const int paired_base=(S/2)*4096+(row>>4)*2048+(row&15)*64
        +(old_lane>>2)*1024+(((old_lane&3)^((row>>1)&3))*16);
    const int first_chunk=paired_base+(row&1)*8;
    const int second_chunk=paired_base+((row&1)^1)*8;
    static_assert(E-S==4);
    auto *dst=cute::raw_pointer_cast(raw_sK.data());
    // The original8lanes/token metadata and all256 producer arrivals remain.
    if constexpr(FRONT_DRAIN) {
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
    }
    { // All256 threads remain active; adjacent4lane groups own pairedtiles.
        constexpr int payload=T::kFP4Cache?256:512;
        constexpr int scale_bytes=T::kFP4Cache?32:16;
        const int page=valid?token_idx>>__builtin_ctz(params.quant_page_size):0;
        const int off=valid?token_idx&(params.quant_page_size-1):0;
        const uint8_t *block;
        const uint8_t *data;
        const uint8_t *scales;
        if constexpr(v41_cache_offset32_enabled<T>) {
            block=reinterpret_cast<const uint8_t*>(params.kv)+uint32_t(page)*uint32_t(params.quant_block_stride);
            data=block+uint32_t(off)*payload+old_lane*(T::kFP4Cache?8:16);
            scales=block+uint32_t(params.quant_page_size)*payload+uint32_t(off)*scale_bytes
                +(T::kFP4Cache?old_lane:(old_lane>>1));
        } else {
            block=reinterpret_cast<const uint8_t*>(params.kv)+int64_t(page)*params.quant_block_stride;
            data=block+int64_t(off)*payload+old_lane*(T::kFP4Cache?8:16);
            scales=block+int64_t(params.quant_page_size)*payload+int64_t(off)*scale_bytes
                +(T::kFP4Cache?old_lane:(old_lane>>1));
        }
        #pragma unroll
        for(int group=0;group<(E-S)/2;++group) {
            const int tile_pair=S+2*group;
            uint32_t scale_bits2=0;
            {
                uint4 packed{}; // one16B load for16FP8 values
                if(valid) {
                    packed=__ldg(reinterpret_cast<const uint4*>(data+tile_pair*64));
                    const uint8_t raw=__ldg(scales+tile_pair*2);
                    scale_bits2=v41_prefill_e8m0_scale2_bits(raw);
                }
                const __nv_bfloat162 scale2=*reinterpret_cast<const __nv_bfloat162*>(&scale_bits2);
                const uint32_t words[4]={packed.x,packed.y,packed.z,packed.w};
                #pragma unroll
                for(int half=0;half<2;++half) {
                    uint32_t out_bits[4]{};
                    #pragma unroll
                    for(int word=0;word<2;++word) {
                        const __nv_fp8x4_e4m3 vals4=*reinterpret_cast<const __nv_fp8x4_e4m3*>(&words[half*2+word]);
                        const float4 vals=(float4)vals4;
                        const __nv_bfloat162 lo=__hmul2(__float22bfloat162_rn({vals.x,vals.y}),scale2);
                        const __nv_bfloat162 hi=__hmul2(__float22bfloat162_rn({vals.z,vals.w}),scale2);
                        out_bits[word*2]=*reinterpret_cast<const uint32_t*>(&lo);
                        out_bits[word*2+1]=*reinterpret_cast<const uint32_t*>(&hi);
                    }
                    const int pos=(half==0?first_chunk:second_chunk)+group*4096;
                    *reinterpret_cast<uint4*>(dst+pos)=uint4{out_bits[0],out_bits[1],out_bits[2],out_bits[3]};
                }
            }
        }
    }
    // Everythread publishes its paired-tile32 decodedBF16 values.
#if defined(FLASHMLA_V41_QUANT_DECODED_STORE_RELEASE)
    __threadfence_block();
    asm volatile("" ::: "memory");
#else
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
#endif
}

#if defined(FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32) || defined(FLASHMLA_V41_QUANT_DECODE_PAGE_BASE64)
template<int S,int E,typename T,bool FRONT_DRAIN=true,typename RawK>
__forceinline__ __device__ void v41_prefill_load_fp8(
    const typename T::Params &params,RawK &raw_sK,int tidx,int token_idx,bool valid,
    const V41QuantLoadRowAddress &row_addr) {
    static_assert(T::kQuantCache && !T::kFP4Cache);
    static_assert(!T::kFP4Cache && T::kBlockM==128 && T::kBlockN==32 && T::kHeadDim==512 && T::kUsePairedCubes);
    const int row=tidx/8,old_lane=tidx&7;
    // Flatten the paired64-dim and16-row geometry once perproducer.
    const int paired_base=(S/2)*4096+(row>>4)*2048+(row&15)*64
        +(old_lane>>2)*1024+(((old_lane&3)^((row>>1)&3))*16);
    const int first_chunk=paired_base+(row&1)*8;
    const int second_chunk=paired_base+((row&1)^1)*8;
    static_assert(E-S==4);
    auto *dst=cute::raw_pointer_cast(raw_sK.data());
    // The original8lanes/token metadata and all256 producer arrivals remain.
    if constexpr(FRONT_DRAIN) {
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
    }
    { // All256 threads remain active; adjacent4lane groups own pairedtiles.
        constexpr int payload=T::kFP4Cache?256:512;
        constexpr int scale_bytes=T::kFP4Cache?32:16;
#if defined(FLASHMLA_V41_QUANT_DECODE_PAGE_BASE64)
        const uint8_t *block=row_addr.payload_row;
#else
        const uint8_t *block=reinterpret_cast<const uint8_t*>(params.kv)+row_addr.row_payload_offset;
#endif
        #pragma unroll
        for(int group=0;group<(E-S)/2;++group) {
            const int tile_pair=S+2*group;
            uint32_t scale_bits2=0;
            {
                uint4 packed{}; // one16B load for16FP8 values
                if(valid) {
                    const uint8_t *data=block+old_lane*(T::kFP4Cache?8:16);
                    const uint8_t *scales=block+row_addr.scale_delta+(T::kFP4Cache?old_lane:(old_lane>>1));
                    packed=__ldg(reinterpret_cast<const uint4*>(data+tile_pair*64));
                    const uint8_t raw=__ldg(scales+tile_pair*2);
                    scale_bits2=v41_prefill_e8m0_scale2_bits(raw);
                }
                const __nv_bfloat162 scale2=*reinterpret_cast<const __nv_bfloat162*>(&scale_bits2);
                const uint32_t words[4]={packed.x,packed.y,packed.z,packed.w};
                #pragma unroll
                for(int half=0;half<2;++half) {
                    uint32_t out_bits[4]{};
                    #pragma unroll
                    for(int word=0;word<2;++word) {
                        const __nv_fp8x4_e4m3 vals4=*reinterpret_cast<const __nv_fp8x4_e4m3*>(&words[half*2+word]);
                        const float4 vals=(float4)vals4;
                        const __nv_bfloat162 lo=__hmul2(__float22bfloat162_rn({vals.x,vals.y}),scale2);
                        const __nv_bfloat162 hi=__hmul2(__float22bfloat162_rn({vals.z,vals.w}),scale2);
                        out_bits[word*2]=*reinterpret_cast<const uint32_t*>(&lo);
                        out_bits[word*2+1]=*reinterpret_cast<const uint32_t*>(&hi);
                    }
                    const int pos=(half==0?first_chunk:second_chunk)+group*4096;
                    *reinterpret_cast<uint4*>(dst+pos)=uint4{out_bits[0],out_bits[1],out_bits[2],out_bits[3]};
                }
            }
        }
    }
    // Everythread publishes its paired-tile32 decodedBF16 values.
#if defined(FLASHMLA_V41_QUANT_DECODED_STORE_RELEASE)
    __threadfence_block();
    asm volatile("" ::: "memory");
#else
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
#endif
}
#endif

// Validated register packets: both formats carry WG0 first across QK and second across PV.
struct V41Fp4RegisterPacket {
    uint2 packed0,packed1;
    uint32_t scale_word0,scale_word1;
};
static_assert(sizeof(V41Fp4RegisterPacket)==24);
struct V41Fp8RegisterPacket {
    uint32_t x0,y0,z0,w0,x1,y1,z1,w1;
    uint8_t scale_raw0,scale_raw1;
};
static_assert(sizeof(V41Fp8RegisterPacket)==36);
// A carried packet retains its own format across a main/extra boundary.
struct V41MixedRegisterPacket {
    union {
        V41Fp8RegisterPacket fp8;
        V41Fp4RegisterPacket fp4;
    };
    bool fp4_format;
};
struct SparseWiNoRegisterPacket {};
template<typename T>
#if defined(FLASHMLA_V40_PREFILL_MODEL1)
using V41QuantRegisterPacket=std::conditional_t<T::kQuantCache,std::conditional_t<T::kMixedFP4,V41MixedRegisterPacket,std::conditional_t<T::kV40Cache,V40Fp8RegisterPacket,std::conditional_t<T::kFP4Cache,V41Fp4RegisterPacket,V41Fp8RegisterPacket>>>,SparseWiNoRegisterPacket>;
#else
using V41QuantRegisterPacket=std::conditional_t<T::kQuantCache,std::conditional_t<T::kMixedFP4,V41MixedRegisterPacket,std::conditional_t<T::kFP4Cache,V41Fp4RegisterPacket,V41Fp8RegisterPacket>>,SparseWiNoRegisterPacket>;
#endif
template<int S,int E,typename T>
__forceinline__ __device__ V41QuantRegisterPacket<T> v41_prefill_register_load(
    const typename T::Params &params,int tidx,int token_idx,bool valid,int block_idx=-1) {
    if constexpr(T::kDecode) {
        return T::CacheAdapter::template register_load<S,E,T>(params,tidx,token_idx,valid,block_idx);
    } else {

#if defined(FLASHMLA_V40_PREFILL_MODEL1)
    if constexpr(T::kV40Cache) return v40_prefill_register_load<S,E,T>(params,tidx,token_idx,valid);
#endif
    static_assert(E-S==4 && (S==0 || S==4));
#if defined(FLASHMLA_V41_QUANT_FP8_PROVEN_PACKET_VALID)
    if constexpr(!T::kFP4Cache && T::kAllValidNoLengths) valid = true;
#endif
    const int old_lane=tidx&7;
    const int page=valid?token_idx>>__builtin_ctz(params.quant_page_size):0;
    const int off=valid?token_idx&(params.quant_page_size-1):0;
    const uint8_t *base;
    if constexpr(v41_cache_offset32_enabled<T>) {
        base=reinterpret_cast<const uint8_t*>(params.kv)+uint32_t(page)*uint32_t(params.quant_block_stride);
    } else {
        base=reinterpret_cast<const uint8_t*>(params.kv)+int64_t(page)*params.quant_block_stride;
    }
    V41QuantRegisterPacket<T> packet{};
    if(valid) {
        if constexpr(T::kFP4Cache) {
            const uint8_t *data;
            const uint8_t *scales;
            if constexpr(v41_cache_offset32_enabled<T>) {
                data=base+uint32_t(off)*256+old_lane*8;
                scales=base+uint32_t(params.quant_page_size)*256+uint32_t(off)*32;
            } else {
                data=base+int64_t(off)*256+old_lane*8;
                scales=base+int64_t(params.quant_page_size)*256+int64_t(off)*32;
            }
            packet.packed0=__ldg(reinterpret_cast<const uint2*>(data+S*32));
            packet.scale_word0=__ldg(reinterpret_cast<const uint32_t*>(scales+((S*4+old_lane)&~3)));
            packet.packed1=__ldg(reinterpret_cast<const uint2*>(data+(S+2)*32));
            packet.scale_word1=__ldg(reinterpret_cast<const uint32_t*>(scales+(((S+2)*4+old_lane)&~3)));
        } else {
            const uint8_t *data;
            const uint8_t *scales;
            if constexpr(v41_cache_offset32_enabled<T>) {
                data=base+uint32_t(off)*512+old_lane*16;
                scales=base+uint32_t(params.quant_page_size)*512+uint32_t(off)*16+(old_lane>>1);
            } else {
                data=base+int64_t(off)*512+old_lane*16;
                scales=base+int64_t(params.quant_page_size)*512+int64_t(off)*16+(old_lane>>1);
            }
            const uint4 g0=__ldg(reinterpret_cast<const uint4*>(data+S*64));
            const uint8_t r0=__ldg(scales+S*2);
            const uint4 g1=__ldg(reinterpret_cast<const uint4*>(data+(S+2)*64));
            const uint8_t r1=__ldg(scales+(S+2)*2);
            packet.x0=g0.x;packet.y0=g0.y;packet.z0=g0.z;packet.w0=g0.w;
            packet.x1=g1.x;packet.y1=g1.y;packet.z1=g1.z;packet.w1=g1.w;
            packet.scale_raw0=r0;packet.scale_raw1=r1;
        }
    }
    return packet;
    }
}

#if defined(FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32) || defined(FLASHMLA_V41_QUANT_DECODE_PAGE_BASE64)
template<int S,int E,typename T>
__forceinline__ __device__ V41QuantRegisterPacket<T> v41_prefill_register_load(
    const typename T::Params &params,int tidx,int token_idx,bool valid,
    const V41QuantLoadRowAddress &row_addr) {
    static_assert(T::kQuantCache && !T::kFP4Cache);
    static_assert(E-S==4 && (S==0 || S==4));
#if defined(FLASHMLA_V41_QUANT_FP8_PROVEN_PACKET_VALID)
    if constexpr(!T::kFP4Cache && T::kAllValidNoLengths) valid = true;
#endif
    const int old_lane=tidx&7;
#if defined(FLASHMLA_V41_QUANT_DECODE_PAGE_BASE64)
    const uint8_t *base=row_addr.payload_row;
#else
    const uint8_t *base=reinterpret_cast<const uint8_t*>(params.kv)+row_addr.row_payload_offset;
#endif
    V41QuantRegisterPacket<T> packet{};
    if(valid) {
        if constexpr(T::kFP4Cache) {
            const uint8_t *data=base+old_lane*8;
            const uint8_t *scales=base+row_addr.scale_delta;
            packet.packed0=__ldg(reinterpret_cast<const uint2*>(data+S*32));
            packet.scale_word0=__ldg(reinterpret_cast<const uint32_t*>(scales+((S*4+old_lane)&~3)));
            packet.packed1=__ldg(reinterpret_cast<const uint2*>(data+(S+2)*32));
            packet.scale_word1=__ldg(reinterpret_cast<const uint32_t*>(scales+(((S+2)*4+old_lane)&~3)));
        } else {
            const uint8_t *data=base+old_lane*16;
            const uint8_t *scales=base+row_addr.scale_delta+(old_lane>>1);
            const uint4 g0=__ldg(reinterpret_cast<const uint4*>(data+S*64));
            const uint8_t r0=__ldg(scales+S*2);
            const uint4 g1=__ldg(reinterpret_cast<const uint4*>(data+(S+2)*64));
            const uint8_t r1=__ldg(scales+(S+2)*2);
            packet.x0=g0.x;packet.y0=g0.y;packet.z0=g0.z;packet.w0=g0.w;
            packet.x1=g1.x;packet.y1=g1.y;packet.z1=g1.z;packet.w1=g1.w;
            packet.scale_raw0=r0;packet.scale_raw1=r1;
        }
    }
    return packet;
}
#endif
template<int S,int E,typename T,typename RowAddress>
__forceinline__ __device__ V41QuantRegisterPacket<T> v41_prefill_register_load_prepared(
    const typename T::Params &params,int tidx,int token_idx,bool valid,
    int loaded_block,const RowAddress &row) {
#if defined(FLASHMLA_V41_QUANT_DECODE_SCOPE_ROW32)
    if constexpr(T::kDecode && T::kDecodeFP8ScopeRow32) {
        return T::CacheAdapter::template register_load_scope_row32<S,E,T>(
            params,tidx,token_idx,valid,loaded_block,row);
    } else
#endif
#if defined(FLASHMLA_V41_QUANT_DECODE_PAGE_BASE64)
    if constexpr(T::kDecode && T::kDecodePageBase64) {
        return T::CacheAdapter::template register_load_page64<S,E,T>(
            params,tidx,token_idx,valid,loaded_block,row);
    } else
#endif
#if defined(FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32)
    if constexpr(!T::kFP4Cache) {
        return v41_prefill_register_load<S,E,T>(params,tidx,token_idx,valid,row);
    } else
#endif
    {
        return v41_prefill_register_load<S,E,T>(params,tidx,token_idx,valid,loaded_block);
    }
}

template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
__forceinline__ __device__ void v41_prefill_fp4_register_finish(RawK &raw_sK,int tidx,const V41Fp4RegisterPacket &packet) {
    static_assert(E-S==4);
    const int row=tidx/8,old_lane=tidx&7;
    const int paired_base=(S/2)*4096+(row>>4)*2048+(row&15)*64
        +(old_lane>>2)*1024+(((old_lane&3)^((row>>1)&3))*16);
    const int first_chunk=paired_base+(row&1)*8;
    const int second_chunk=paired_base+((row&1)^1)*8;
    auto *dst=cute::raw_pointer_cast(raw_sK.data());
    // Retain the exact baseline reader drain at its original finish point.
    if constexpr(FRONT_DRAIN) {
        cutlass::arch::NamedBarrier::sync(256,
            static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
    }
    #pragma unroll
    for(int group=0;group<2;++group) {
        const uint2 packed=group==0?packet.packed0:packet.packed1;
        {
            const uint32_t scale_word=group==0?packet.scale_word0:packet.scale_word1;
            const int scale_idx=(S+2*group)*4+old_lane;
            const uint8_t raw=uint8_t(scale_word>>((scale_idx&3)*8));
            const float scale=float(*reinterpret_cast<const __nv_fp8_e4m3*>(&raw));
            const __nv_bfloat162 scale2=__float2bfloat162_rn(scale);
            const uint32_t words[2]={packed.x,packed.y};
            #pragma unroll
            for(int half=0;half<2;++half) {
                const uint4 word_bits=v41_prefill_fp4_word_shared_sign(words[half]);
                const uint32_t decoded[4]={word_bits.x,word_bits.y,word_bits.z,word_bits.w};
                uint32_t out_bits[4]{};
                #pragma unroll
                for(int pair=0;pair<4;++pair) {
                    const uint32_t bits=decoded[pair];
                    const __nv_bfloat162 out=__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&bits),scale2);
                    out_bits[pair]=*reinterpret_cast<const uint32_t*>(&out);
                }
                const int pos=(half==0?first_chunk:second_chunk)+group*4096;
                *reinterpret_cast<uint4*>(dst+pos)=uint4{out_bits[0],out_bits[1],out_bits[2],out_bits[3]};
            }
        }
    }
#if defined(FLASHMLA_V41_QUANT_DECODED_STORE_RELEASE)
    __threadfence_block();
    asm volatile("" ::: "memory");
#else
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
#endif
}

template<int S,int E,typename T,bool FRONT_DRAIN=true,typename RawK>
__forceinline__ __device__ void v41_prefill_fp8_register_finish(
    const typename T::Params &params,RawK &raw_sK,int tidx,int token_idx,bool valid,
#if defined(FLASHMLA_V40_PREFILL_MODEL1)
    const V41QuantRegisterPacket<T> &packet) {
    if constexpr(T::kV40Cache) {
        v40_prefill_register_finish<S,E,T,FRONT_DRAIN>(raw_sK,tidx,valid,packet);
        return;
    }
#else
    const V41Fp8RegisterPacket &packet) {
#endif
    static_assert(!T::kFP4Cache && T::kBlockM==128 && T::kBlockN==32 && T::kHeadDim==512 && T::kUsePairedCubes);
    const int row=tidx/8,old_lane=tidx&7;
    // Flatten the paired64-dim and16-row geometry once perproducer.
    const int paired_base=(S/2)*4096+(row>>4)*2048+(row&15)*64
        +(old_lane>>2)*1024+(((old_lane&3)^((row>>1)&3))*16);
    const int first_chunk=paired_base+(row&1)*8;
    const int second_chunk=paired_base+((row&1)^1)*8;
    static_assert(E-S==4);
    auto *dst=cute::raw_pointer_cast(raw_sK.data());
    // The original8lanes/token metadata and all256 producer arrivals remain.
    if constexpr(FRONT_DRAIN) {
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
    }
    { // All256 threads remain active; adjacent4lane groups own pairedtiles.
        // All input bytes and scales are carried in packet; no cache address is formed here.
        #pragma unroll
        for(int group=0;group<(E-S)/2;++group) {
            const int tile_pair=S+2*group;
            uint32_t scale_bits2=0;
            {
                uint4 packed{}; // one16B load for16FP8 values
                if(valid) {
                    packed=group==0?uint4{packet.x0,packet.y0,packet.z0,packet.w0}:
                                    uint4{packet.x1,packet.y1,packet.z1,packet.w1};
                    const uint8_t raw=uint8_t(group==0?packet.scale_raw0:packet.scale_raw1);
                    scale_bits2=v41_prefill_e8m0_scale2_bits(raw);
                }
                const __nv_bfloat162 scale2=*reinterpret_cast<const __nv_bfloat162*>(&scale_bits2);
                const uint32_t words[4]={packed.x,packed.y,packed.z,packed.w};
                #pragma unroll
                for(int half=0;half<2;++half) {
                    uint32_t out_bits[4]{};
                    #pragma unroll
                    for(int word=0;word<2;++word) {
                        const __nv_fp8x4_e4m3 vals4=*reinterpret_cast<const __nv_fp8x4_e4m3*>(&words[half*2+word]);
                        const float4 vals=(float4)vals4;
                        const __nv_bfloat162 lo=__hmul2(__float22bfloat162_rn({vals.x,vals.y}),scale2);
                        const __nv_bfloat162 hi=__hmul2(__float22bfloat162_rn({vals.z,vals.w}),scale2);
                        out_bits[word*2]=*reinterpret_cast<const uint32_t*>(&lo);
                        out_bits[word*2+1]=*reinterpret_cast<const uint32_t*>(&hi);
                    }
                    const int pos=(half==0?first_chunk:second_chunk)+group*4096;
                    *reinterpret_cast<uint4*>(dst+pos)=uint4{out_bits[0],out_bits[1],out_bits[2],out_bits[3]};
                }
            }
        }
    }
    // Everythread publishes its paired-tile32 decodedBF16 values.
#if defined(FLASHMLA_V41_QUANT_DECODED_STORE_RELEASE)
    __threadfence_block();
    asm volatile("" ::: "memory");
#else
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
#endif
}


template <
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void launch_kv_tiles_dsa_wg_copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV)       // (BLOCK_N, HEAD_DIM_K), swizzled
{
    Tensor cur_gKV = gKV(_, _, Int<START_HEAD_DIM_TILE_IDX>{});
    Tensor cur_sKV = sKV(_, _, Int<START_HEAD_DIM_TILE_IDX>{});
    cute::copy(tiled_copy, cur_gKV, cur_sKV);

    if constexpr (START_HEAD_DIM_TILE_IDX + 1 < END_HEAD_DIM_TILE_IDX)
    {
        launch_kv_tiles_dsa_wg_copy<START_HEAD_DIM_TILE_IDX + 1, END_HEAD_DIM_TILE_IDX>(tiled_copy, gKV, sKV);
    }
}

template<typename T,int S,int E,bool FRONT_DRAIN=true,bool ASYNC_PREISSUED=false,typename TiledCopy,
    typename EG,typename LG,typename ES,typename LS,typename RawK>
__forceinline__ __device__ void launch_kv_tiles_dsa_wg(
    TiledCopy tiled_copy,Tensor<EG,LG> const &gKV,Tensor<ES,LS> &sKV,
    __mbarrier_t *barriers_K,RawK &raw_sK,int tidx,
    const typename T::Params &params,int token_idx) {
    if constexpr(!T::kQuantCache) {
        launch_kv_tiles_dsa_wg_copy<S,E>(tiled_copy,gKV,sKV);
    } else if constexpr(T::kFP4Cache) {
        if constexpr(!ASYNC_PREISSUED) {
            v41_prefill_fp4_async_issue<S,E,T>(params,tidx,token_idx,tiled_copy.pred);
        }
        v41_prefill_fp4_async_finish<S,E,T,FRONT_DRAIN>(raw_sK,tidx);
    } else {
        v41_prefill_load_fp8<S,E,T,FRONT_DRAIN>(params,raw_sK,tidx,token_idx,tiled_copy.pred);
    }
    cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
}

#if defined(FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32)
template<typename T,int S,int E,bool FRONT_DRAIN=true,bool ASYNC_PREISSUED=false,typename TiledCopy,
    typename EG,typename LG,typename ES,typename LS,typename RawK>
__forceinline__ __device__ void launch_kv_tiles_dsa_wg(
    TiledCopy tiled_copy,Tensor<EG,LG> const &gKV,Tensor<ES,LS> &sKV,
    __mbarrier_t *barriers_K,RawK &raw_sK,int tidx,
    const typename T::Params &params,int token_idx,
    const V41QuantLoadRowAddress &row_addr) {
    static_assert(T::kQuantCache && !T::kFP4Cache);
    v41_prefill_load_fp8<S,E,T,FRONT_DRAIN>(params,raw_sK,tidx,token_idx,tiled_copy.pred,row_addr);
    cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
}
#endif


template<typename T>
__forceinline__ __device__ int quant_wi_token_index(
    const typename T::Params &params, const int *prefill_indices,
    int block_idx, int real_end_block_idx, int tidx) {
    if constexpr(T::kDecode) {
        const int column=tidx/8;
        const int absolute=block_idx*T::kBlockN+column;
        const bool extra=T::kDecodeFixedExtraIndex || absolute>=params.main_padded_length;
        const int position=extra?absolute-params.main_padded_length:absolute;
        if constexpr(T::kAllValidNoLengths) {
            // Prefetch can extend past a task's end, but never past physical storage.
            if(block_idx<0 || block_idx>=real_end_block_idx) return -1;
        } else {
            const int storage=extra?params.extra_storage_topk:params.main_storage_topk;
            const int length=extra?params.extra_length:params.main_length;
            if(position<0 || position>=storage || position>=length) return -1;
        }
        const int *indices=extra?params.extra_indices:params.indices;
        return __ldg(indices+position);
    } else {
        return block_idx<real_end_block_idx?__ldg(prefill_indices+block_idx*T::kBlockN):-1;
    }
}

// mirrors decode sparse_decode_wg.cuh:1533-1591 (compute_K_addr_bf16) -- Phase 1
// of the K-address two-phase split (Opt-C): pure address computation + valid
// mask write. Does NOT issue cp.async, so it can be placed in TC idle windows.
// Key adaptation vs decode: prefill addressing is flat --
//   gK_base + token_idx * stride_kv_s_kv + (idx_in_warpgroup % 8) * 8
// with NO page-table two-level decomposition and no USE_EXTRA branch; only the
// two-phase STRUCTURE is ported, not decode's paged address math.
// WRITE_VI: whether this call site owns the block's flag slot (WG0 owns even
// blocks -> slots 0/1, WG1 owns odd blocks -> slots 2/3). The cross-WG call
// (same block, other half of the tiles) passes false and skips the flag store.
template<typename T, bool WRITE_VI, typename TensorVI>
__forceinline__ __device__ void dsa_compute_K_addr(
    const typename T::Params &params,
    typename T::InputT *gK_base,
    int token_idx,                 // prefetched token index (this thread's column)
    int block_idx,                 // block being computed (border fold-in base)
    int seqlen_k,
    int idx_in_warpgroup,
    TensorVI &smem_valid_indices,
    int vi_buf,
#if defined(FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32) || defined(FLASHMLA_V41_QUANT_DECODE_SCOPE_ROW32)
    V41PrecomputedKAddress<T> &precomp_ptr,
#else
    typename T::InputT *&precomp_ptr,
#endif
    bool &precomp_valid)
{
    using InputT = typename T::InputT;
    if constexpr(T::kDecode) {
        if constexpr(T::kAllValidNoLengths) {
            precomp_valid=true;
        } else {
            const int absolute=block_idx*T::kBlockN+idx_in_warpgroup/8;
            const bool extra=T::kDecodeFixedExtraIndex || absolute>=params.main_padded_length;
            const int position=extra?absolute-params.main_padded_length:absolute;
            const int length=extra?params.extra_length:params.main_length;
            const int storage=extra?params.extra_storage_topk:params.main_storage_topk;
            const int capacity=extra?params.extra_s_kv:params.s_kv;
            precomp_valid=token_idx>=0 && token_idx<capacity && position>=0 && position<length && position<storage;
        }
#if defined(FLASHMLA_V41_QUANT_DECODE_SCOPE_ROW32)
        if constexpr(T::kDecodeFP8ScopeRow32) {
            precomp_ptr=T::CacheAdapter::template prepare_scope_row32<T>(
                params,token_idx,precomp_valid,block_idx);
        } else
#endif
#if defined(FLASHMLA_V41_QUANT_DECODE_PAGE_BASE64)
        if constexpr(T::kDecodePageBase64) {
            precomp_ptr=T::CacheAdapter::template prepare_page_base64<T>(
                params,token_idx,precomp_valid,block_idx);
        } else
#endif
        {
            precomp_ptr=gK_base;
        }
        if constexpr(WRITE_VI && !T::kAllValidNoLengths) {
            if(idx_in_warpgroup%8==0) smem_valid_indices(vi_buf,idx_in_warpgroup/8)=precomp_valid;
        }
    } else {
#if defined(FLASHMLA_V41_QUANT_ALLVALID_NO_LENGTHS)
    bool is_token_valid = T::kAllValidNoLengths || (token_idx >= 0 && token_idx < params.s_kv);
#else
    bool is_token_valid = token_idx >= 0 && token_idx < params.s_kv;
#endif
    if constexpr(T::kQuantCache) {
#if defined(FLASHMLA_V40_PREFILL_MODEL1)
        constexpr int payload=T::kV40Cache?576:(T::kFP4Cache?256:512);
#else
        constexpr int payload=T::kFP4Cache?256:512;
#endif
        const int page=is_token_valid?token_idx>>__builtin_ctz(params.quant_page_size):0;
        const int off=is_token_valid?token_idx&(params.quant_page_size-1):0;
#if defined(FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32)
        if constexpr(!T::kFP4Cache) {
            constexpr int scale_bytes=16;
            const uint32_t page_offset=uint32_t(page)*uint32_t(params.quant_block_stride);
            precomp_ptr.row_payload_offset=page_offset+uint32_t(off)*payload;
            precomp_ptr.scale_delta=uint32_t(params.quant_page_size)*payload-uint32_t(off)*(payload-scale_bytes);
        } else {
#if defined(FLASHMLA_V41_QUANT_ADDRESS32)
        precomp_ptr=reinterpret_cast<typename T::InputT*>(
            reinterpret_cast<uint8_t*>(params.kv)+uint32_t(page)*uint32_t(params.quant_block_stride)+
            uint32_t(off)*payload+(idx_in_warpgroup%8)*(T::kFP4Cache?4:8));
#else
        precomp_ptr=reinterpret_cast<typename T::InputT*>(
            reinterpret_cast<uint8_t*>(params.kv)+int64_t(page)*params.quant_block_stride+
            int64_t(off)*payload+(idx_in_warpgroup%8)*(T::kFP4Cache?4:8));
#endif
        }
#else
#if defined(FLASHMLA_V41_QUANT_ADDRESS32)
        precomp_ptr=reinterpret_cast<typename T::InputT*>(
            reinterpret_cast<uint8_t*>(params.kv)+uint32_t(page)*uint32_t(params.quant_block_stride)+
            uint32_t(off)*payload+(idx_in_warpgroup%8)*(T::kFP4Cache?4:8));
#else
        precomp_ptr=reinterpret_cast<typename T::InputT*>(
            reinterpret_cast<uint8_t*>(params.kv)+int64_t(page)*params.quant_block_stride+
            int64_t(off)*payload+(idx_in_warpgroup%8)*(T::kFP4Cache?4:8));
#endif
#endif
    } else {
    precomp_ptr = gK_base + token_idx * (int64_t)params.stride_kv_s_kv + (idx_in_warpgroup % 8) * 8;
    }
    precomp_valid = is_token_valid;
    // Same flag value as the legacy inline write points: per-token validity with
    // the topk_length right border folded in (absolute topk position
    // block_idx*kBlockN + col < seqlen_k). The store guard (idx%8==0) and the
    // value formula are identical to the pre-split code -- only the TIMING moves
    // one iteration earlier (vi slot invariant (b) is preserved by the matching
    // runtime guards at the call sites; see traits.h L193-202).
#if defined(FLASHMLA_V41_QUANT_ALLVALID_NO_LENGTHS)
    if constexpr(!T::kAllValidNoLengths) {
#endif
    if constexpr (WRITE_VI) {
        if (idx_in_warpgroup % 8 == 0) {
            smem_valid_indices(vi_buf, idx_in_warpgroup / 8) =
                is_token_valid && (block_idx * T::kBlockN + idx_in_warpgroup / 8 < seqlen_k);
        }
    }
#if defined(FLASHMLA_V41_QUANT_ALLVALID_NO_LENGTHS)
    }
#endif
    }
}

// mirrors decode sparse_decode_wg.cuh:1620-1674 (issue_K_load_bf16) -- Phase 2
// of the split: issue cp.async with the precomputed pointer, fused with the
// __ldg prefetch of the NEXT block's token index. No address math, no vi store.
// DO_PREFETCH=false is for the prolog (which does its own guarded prefetch of
// the next two block indices).
template<int S, int E, typename T, bool DO_PREFETCH = true, bool FRONT_DRAIN = true, bool ASYNC_PREISSUED = false,
         bool REGISTER_PREISSUED = false, typename TiledCopy, typename Engine0, typename Layout0,
         typename Engine1, typename Layout1, typename RawK>
__forceinline__ __device__ void dsa_issue_K_load(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> &tKgK,   // partitioned gmem src (data ptr overwritten)
    Tensor<Engine1, Layout1> &tKsK,   // partitioned smem dst
    __mbarrier_t *barriers_K,
#if defined(FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32) || defined(FLASHMLA_V41_QUANT_DECODE_SCOPE_ROW32)
    V41PrecomputedKAddress<T> precomp_ptr,
#else
    typename T::InputT *precomp_ptr,
#endif
    bool precomp_valid,
    int *gIndices_ptr,                // per-thread prefetch base (incl. idx/8 offset)
    int prefetch_block,               // block whose token index is prefetched next
    int real_end_block_idx,           // [Even-align] REAL topk block count -- the only
                                      // safe bound for gIndices reads (see kernel prolog)
    int &nxt_token_idx, RawK &raw_sK, int tidx,
    const typename T::Params &params, int quant_token_idx,
    const V41QuantRegisterPacket<T> *register_packet=nullptr, int load_block_idx=-1)
{
#if defined(FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32)
    if constexpr(T::kFP4Cache || !T::kQuantCache) tKgK.data() = precomp_ptr;
#elif defined(FLASHMLA_V41_QUANT_DECODE_SCOPE_ROW32)
    if constexpr(!T::kDecodeFP8ScopeRow32) tKgK.data() = precomp_ptr;
#else
    tKgK.data() = precomp_ptr;
#endif
    tiled_copy.pred = precomp_valid;
    if constexpr(T::kQuantCache && REGISTER_PREISSUED) {
        if constexpr(T::kDecode) {
            T::CacheAdapter::template register_finish<S,E,T,FRONT_DRAIN>(
                params,raw_sK,tidx,quant_token_idx,precomp_valid,*register_packet);
        } else if constexpr(T::kFP4Cache) {
            v41_prefill_fp4_register_finish<S,E,T,FRONT_DRAIN>(raw_sK,tidx,*register_packet);
        } else {
            v41_prefill_fp8_register_finish<S,E,T,FRONT_DRAIN>(params,raw_sK,tidx,quant_token_idx,precomp_valid,*register_packet);
        }
        cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
    } else {
        if constexpr(T::kDecode) {
#if defined(FLASHMLA_V41_QUANT_DECODE_SCOPE_ROW32)
            if constexpr(T::kDecodeFP8ScopeRow32) {
                T::CacheAdapter::template load_scope_row32<S,E,T,FRONT_DRAIN>(
                    params,raw_sK,tidx,quant_token_idx,precomp_valid,load_block_idx,precomp_ptr);
            } else
#endif
#if defined(FLASHMLA_V41_QUANT_DECODE_PAGE_BASE64)
            if constexpr(T::kDecodePageBase64) {
                T::CacheAdapter::template load_page64<S,E,T,FRONT_DRAIN>(
                    params,raw_sK,tidx,quant_token_idx,precomp_valid,load_block_idx,precomp_ptr);
            } else
#endif
            {
                T::CacheAdapter::template load<S,E,T,FRONT_DRAIN>(
                    params,raw_sK,tidx,quant_token_idx,precomp_valid,load_block_idx,ASYNC_PREISSUED);
            }
            cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
        } else {
#if defined(FLASHMLA_V41_QUANT_FP8_OFFSET_DELTA32)
        if constexpr(T::kFP4Cache || !T::kQuantCache) {
        launch_kv_tiles_dsa_wg<T,S,E,FRONT_DRAIN,ASYNC_PREISSUED>(tiled_copy,tKgK,tKsK,barriers_K,raw_sK,tidx,params,quant_token_idx);
        } else {
        launch_kv_tiles_dsa_wg<T,S,E,FRONT_DRAIN,ASYNC_PREISSUED>(tiled_copy,tKgK,tKsK,barriers_K,raw_sK,tidx,params,quant_token_idx,precomp_ptr);
        }
#else
        launch_kv_tiles_dsa_wg<T,S,E,FRONT_DRAIN,ASYNC_PREISSUED>(tiled_copy,tKgK,tKsK,barriers_K,raw_sK,tidx,params,quant_token_idx);
#endif
        }
    }
    if constexpr (DO_PREFETCH) {
        // [Even-align] guard by the REAL block count: a padding block (beyond
        // real_end_block_idx) has no backing gIndices memory, so fabricate
        // token_idx = -1 -> invalid -> pred=false cp.async + vi-flag=false.
        if (prefetch_block < real_end_block_idx) {
            nxt_token_idx = quant_wi_token_index<T>(params,gIndices_ptr,prefetch_block,real_end_block_idx,tidx);
        } else {
            nxt_token_idx = -1;
        }
    }
}

#include "wg_tile_driver.cuh"

// Compatibility entry for existing quant Decode schedulers. No tile loop here.
template<typename T, bool Is_causal>
__forceinline__ __device__ void quant_wi_tile_body(const typename T::Params &params, int m_block_idx) {
    sparse_wi_tile_driver<T,Is_causal>(params,m_block_idx);
}

template<typename T, bool Is_causal>
__forceinline__ __device__ void quant_prefill_wi_device_body(const typename T::Params &params) {
    quant_wi_tile_body<T,Is_causal>(params,static_cast<int>(blockIdx.x));
}

template<typename T, bool Is_causal>
__global__ void __launch_bounds__(T::NUM_THREADS,1,1)
sparse_quant_wi_kernel(__grid_constant__ const typename T::KernelParams params) {
    if constexpr(T::kDecode) {
        using Scheduler=typename T::Scheduler;
        Scheduler::template run<T>(params);
    } else {
        quant_wi_tile_body<T,Is_causal>(params,static_cast<int>(blockIdx.x));
    }
}

#undef QKT_GEMM_ONE_TILE
#undef DSA_LAUNCH_WG0_SUBROUTINE
#undef DSA_LAUNCH_WG1_SUBROUTINE
} // namespace flashmla::v41prefill
