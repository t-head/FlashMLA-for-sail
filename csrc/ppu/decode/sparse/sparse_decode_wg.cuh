// =============================================================================
// sparse_decode_wg.cuh
//
// Sparse FP8 decode warp-interleave (T2) variant -- forked verbatim from
// csrc/ppu/decode/dense/splitkv_mla_kernel.cuh. The double-warpgroup ping-pong skeleton
// (wg0_subroutine / wg1_subroutine / softmax / O store / mbarrier handshake)
// is reused as-is. The only sparse-specific divergences live in:
//   * Q / K-load layer  (replaces launch_kv_tiles_copy AIU LOAD with cp.async
//                        + indices indirection + per-warpgroup local FP8→BF16
//                        dequant)
//   * sparse mask via GMEM re-read of indices (applied to rP0 / rP1 before softmax)
// extra-K and attn_sink are supported (extra-K load path and attn_sink
// epilogue below); only the causal path is removed -- not supported, gated at
// host dispatch.
// =============================================================================

#pragma once

#include "kerutils/host/host.h"

#include <cute/tensor.hpp>
#include <climits>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cute/util/debug.hpp>

#include "decode/sparse/sparse_decode_wg.h"
// M128 traits inherit the dense splitkv layout; M64 traits live alongside them
// in the sparse WG traits header.
#include "decode/sparse/sparse_decode_wg_traits.h"

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cute/util/debug.hpp>
#include <type_traits>
#include "params.h"
#include "decode/sparse/sparse_decode_wg_traits.h"


#include "acc_vreg_fraga.h"
#include "kernel_traits.h"
#include "utils.h"
#include "kerutils/device/ppu/softmax.cuh"
#include "kerutils/device/ppu/mask.cuh"
#include "kerutils/device/ppu/dequant.cuh"

namespace flashmla::dsa::v41 {
using namespace cute;
using cutlass::arch::NamedBarrier;

static constexpr float MAX_INIT_VAL_SM = -1e30f;

// Adapt packed register words to the shared SplitKV converter.
__forceinline__ __device__ uint4 fp8_words_to_scaled_bf16x8_bits(
        uint32_t first_word, uint32_t second_word, float scale) {
    alignas(16) uint32_t words[2] = {first_word, second_word};
    alignas(16) uint32_t out_bits[4]{};
    auto src = make_tensor(make_rmem_ptr(words), Layout<Shape<_2>>{});
    auto dst = make_tensor(make_rmem_ptr(out_bits), Layout<Shape<_4>>{});
    flash::fp8_to_bf16(src, dst, scale);
    return uint4{out_bits[0], out_bits[1], out_bits[2], out_bits[3]};
}

struct V41RowAddress32 {
    uint32_t row_payload_offset;
    uint32_t scale_delta;
};
static_assert(sizeof(V41RowAddress32)==8);
template<typename T>
using V41PrecomputedKAddress=std::conditional_t<T::kDecodeFP8ScopeRow32,V41RowAddress32,typename T::InputT*>;
struct V41RowAddress64 {
    const uint8_t *payload_row;
    int64_t scale_delta;
};
static_assert(sizeof(V41RowAddress64)==16);
template<typename T>
inline constexpr bool v41_cache_offset32_enabled =
    T::kCacheOffset32;
struct SparseDecodeV41FixedExtraFP4CacheAdapter;
template<typename T>
struct SparseDecodeV41NativeFP4SteadyTraits : T {
    static_assert(T::kDecode && T::kMixedFP4 && T::kHeadDim==512);
    using CacheAdapter=SparseDecodeV41FixedExtraFP4CacheAdapter;
    static constexpr bool kDecodeFixedExtraIndex=true;
    static constexpr bool kFP4Cache=true;
    static constexpr bool kMixedFP4=false;
};
struct SparseDecodeV41FixedExtraFP8CacheAdapter;
template<typename T>
struct SparseDecodeV41FixedExtraFP8SteadyTraits : T {
    static_assert(T::kDecode && !T::kMixedFP4 && !T::kFP4Cache);
    static_assert(T::kDecodeFP8ScopeRow32 && !T::kDecodePageBase64);
    using CacheAdapter=SparseDecodeV41FixedExtraFP8CacheAdapter;
    static constexpr bool kDecodeFixedExtraIndex=true;
};

static constexpr bool kFP4AsyncEarly = true;
static constexpr int V41_FP4_ASYNC_STAGE_BASE=248960;
static constexpr int V41_FP4_ASYNC_STAGE_STRIDE=24;
template<typename T>
__forceinline__ __device__ uint8_t *quant_cache_fp4_async_slot() {
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
__forceinline__ __device__ void quant_cache_fp4_async_issue(
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
        data=block+uint32_t(off)*288+old_lane*8;
        scales=block+uint32_t(off)*288+256;
    } else {
        block=reinterpret_cast<const uint8_t*>(params.kv)+
            int64_t(page)*params.quant_block_stride;
        data=block+int64_t(off)*288+old_lane*8;
        scales=block+int64_t(off)*288+256;
    }
    auto *slot=quant_cache_fp4_async_slot<T>();
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
__forceinline__ __device__ void quant_cache_fp4_async_finish(RawK &raw_sK,int tidx) {
    static_assert(E-S==4);
    const int row=tidx/8,old_lane=tidx&7;
    const int paired_base=(S/2)*4096+(row>>4)*2048+(row&15)*64
        +(old_lane>>2)*1024+(((old_lane&3)^((row>>1)&3))*16);
    const int first_chunk=paired_base+(row&1)*8;
    const int second_chunk=paired_base+((row&1)^1)*8;
    auto *dst=cute::raw_pointer_cast(raw_sK.data());
    if constexpr(FRONT_DRAIN) {
        cutlass::arch::NamedBarrier::sync(256,
            static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
    }
    __pipeline_wait_prior(0);
    const auto *slot=quant_cache_fp4_async_slot<T>();
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
                const uint4 word_bits=flash::fp4_word_to_bf16x2_bits(words[half]);
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
    __threadfence_block();
    asm volatile("" ::: "memory");
}
__forceinline__ __device__ uint32_t quant_cache_e8m0_scale2_bits(uint8_t raw) {
    uint32_t bits=uint32_t(raw)*0x00800080u;
    bits=raw==0?0x00400040u:bits;
    bits=raw==255?0x7fff7fffu:bits;
    return bits;
}

template<int S,int E,typename T,bool FRONT_DRAIN=true,typename RawK>
__forceinline__ __device__ void quant_cache_load_fp8(
    const typename T::Params &params,RawK &raw_sK,int tidx,int token_idx,bool valid) {
    static_assert(!T::kFP4Cache && T::kBlockM==128 && T::kBlockN==32 && T::kHeadDim==512 && T::kUsePairedCubes);
    const int row=tidx/8,old_lane=tidx&7;
    const int paired_base=(S/2)*4096+(row>>4)*2048+(row&15)*64
        +(old_lane>>2)*1024+(((old_lane&3)^((row>>1)&3))*16);
    const int first_chunk=paired_base+(row&1)*8;
    const int second_chunk=paired_base+((row&1)^1)*8;
    static_assert(E-S==4);
    auto *dst=cute::raw_pointer_cast(raw_sK.data());
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
            data=block+uint32_t(off)*(payload+scale_bytes)+old_lane*(T::kFP4Cache?8:16);
            scales=block+uint32_t(off)*(payload+scale_bytes)+payload
                +(T::kFP4Cache?old_lane:(old_lane>>1));
        } else {
            block=reinterpret_cast<const uint8_t*>(params.kv)+int64_t(page)*params.quant_block_stride;
            data=block+int64_t(off)*(payload+scale_bytes)+old_lane*(T::kFP4Cache?8:16);
            scales=block+int64_t(off)*(payload+scale_bytes)+payload
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
                    scale_bits2=quant_cache_e8m0_scale2_bits(raw);
                }
                const __nv_bfloat162 scale2=*reinterpret_cast<const __nv_bfloat162*>(&scale_bits2);
                const uint32_t words[4]={packed.x,packed.y,packed.z,packed.w};
                #pragma unroll
                for(int half=0;half<2;++half) {
                    const uint4 out_bits = fp8_words_to_scaled_bf16x8_bits(
                        words[half * 2], words[half * 2 + 1],
                        __bfloat162float(__low2bfloat16(scale2)));
                    const int pos=(half==0?first_chunk:second_chunk)+group*4096;
                    *reinterpret_cast<uint4*>(dst+pos)=out_bits;
                }
            }
        }
    }
    __threadfence_block();
    asm volatile("" ::: "memory");
}

template<int S,int E,typename T,bool FRONT_DRAIN=true,typename RawK>
__forceinline__ __device__ void quant_cache_load_fp8(
    RawK &raw_sK,int tidx,bool valid,const V41RowAddress64 &row_addr) {
    static_assert(T::kQuantCache && !T::kFP4Cache);
    static_assert(!T::kFP4Cache && T::kBlockM==128 && T::kBlockN==32 && T::kHeadDim==512 && T::kUsePairedCubes);
    const int row=tidx/8,old_lane=tidx&7;
    const int paired_base=(S/2)*4096+(row>>4)*2048+(row&15)*64
        +(old_lane>>2)*1024+(((old_lane&3)^((row>>1)&3))*16);
    const int first_chunk=paired_base+(row&1)*8;
    const int second_chunk=paired_base+((row&1)^1)*8;
    static_assert(E-S==4);
    auto *dst=cute::raw_pointer_cast(raw_sK.data());
    if constexpr(FRONT_DRAIN) {
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
    }
    { // All256 threads remain active; adjacent4lane groups own pairedtiles.
        const uint8_t *block=row_addr.payload_row;
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
                    scale_bits2=quant_cache_e8m0_scale2_bits(raw);
                }
                const __nv_bfloat162 scale2=*reinterpret_cast<const __nv_bfloat162*>(&scale_bits2);
                const uint32_t words[4]={packed.x,packed.y,packed.z,packed.w};
                #pragma unroll
                for(int half=0;half<2;++half) {
                    const uint4 out_bits = fp8_words_to_scaled_bf16x8_bits(
                        words[half * 2], words[half * 2 + 1],
                        __bfloat162float(__low2bfloat16(scale2)));
                    const int pos=(half==0?first_chunk:second_chunk)+group*4096;
                    *reinterpret_cast<uint4*>(dst+pos)=out_bits;
                }
            }
        }
    }
    __threadfence_block();
    asm volatile("" ::: "memory");
}
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
struct V41MixedRegisterPacket {
    union {
        V41Fp8RegisterPacket fp8;
        V41Fp4RegisterPacket fp4;
    };
    bool fp4_format;
};
struct SparseWiNoRegisterPacket {};
template<typename T>
using V41QuantRegisterPacket=std::conditional_t<T::kQuantCache,std::conditional_t<T::kMixedFP4,V41MixedRegisterPacket,std::conditional_t<T::kFP4Cache,V41Fp4RegisterPacket,V41Fp8RegisterPacket>>,SparseWiNoRegisterPacket>;
template<int S,int E,typename T>
__forceinline__ __device__ V41QuantRegisterPacket<T> quant_cache_register_load(
    const typename T::Params &params,int tidx,int token_idx,bool valid,int block_idx=-1) {
    if constexpr(T::kDecode) {
        return T::CacheAdapter::template register_load<S,E,T>(params,tidx,token_idx,valid,block_idx);
    } else {

    static_assert(E-S==4 && (S==0 || S==4));
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
                data=base+uint32_t(off)*288+old_lane*8;
                scales=base+uint32_t(off)*288+256;
            } else {
                data=base+int64_t(off)*288+old_lane*8;
                scales=base+int64_t(off)*288+256;
            }
            packet.packed0=__ldg(reinterpret_cast<const uint2*>(data+S*32));
            packet.scale_word0=__ldg(reinterpret_cast<const uint32_t*>(scales+((S*4+old_lane)&~3)));
            packet.packed1=__ldg(reinterpret_cast<const uint2*>(data+(S+2)*32));
            packet.scale_word1=__ldg(reinterpret_cast<const uint32_t*>(scales+(((S+2)*4+old_lane)&~3)));
        } else {
            const uint8_t *data;
            const uint8_t *scales;
            if constexpr(v41_cache_offset32_enabled<T>) {
                data=base+uint32_t(off)*528+old_lane*16;
                scales=base+uint32_t(off)*528+512+(old_lane>>1);
            } else {
                data=base+int64_t(off)*528+old_lane*16;
                scales=base+int64_t(off)*528+512+(old_lane>>1);
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

template<int S,int E,typename T>
__forceinline__ __device__ V41QuantRegisterPacket<T> quant_cache_register_load(
    int tidx,bool valid,const V41RowAddress64 &row_addr) {
    static_assert(T::kQuantCache && !T::kFP4Cache);
    static_assert(E-S==4 && (S==0 || S==4));
    const int old_lane=tidx&7;
    const uint8_t *base=row_addr.payload_row;
    V41QuantRegisterPacket<T> packet{};
    if(valid) {
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
    return packet;
}
template<int S,int E,typename T,typename RowAddress>
__forceinline__ __device__ V41QuantRegisterPacket<T> quant_cache_register_load_prepared(
    const typename T::Params &params,int tidx,int token_idx,bool valid,
    int loaded_block,const RowAddress &row) {
    if constexpr(T::kDecode && T::kDecodeFP8ScopeRow32) {
        return T::CacheAdapter::template register_load_scope_row32<S,E,T>(
            params,tidx,token_idx,valid,loaded_block,row);
    } else
    if constexpr(T::kDecode && T::kDecodePageBase64) {
        return T::CacheAdapter::template register_load_page64<S,E,T>(
            params,tidx,token_idx,valid,loaded_block,row);
    } else
    {
        return quant_cache_register_load<S,E,T>(params,tidx,token_idx,valid,loaded_block);
    }
}

template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
__forceinline__ __device__ void quant_cache_fp4_register_finish(RawK &raw_sK,int tidx,const V41Fp4RegisterPacket &packet) {
    static_assert(E-S==4);
    const int row=tidx/8,old_lane=tidx&7;
    const int paired_base=(S/2)*4096+(row>>4)*2048+(row&15)*64
        +(old_lane>>2)*1024+(((old_lane&3)^((row>>1)&3))*16);
    const int first_chunk=paired_base+(row&1)*8;
    const int second_chunk=paired_base+((row&1)^1)*8;
    auto *dst=cute::raw_pointer_cast(raw_sK.data());
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
                const uint4 word_bits=flash::fp4_word_to_bf16x2_bits(words[half]);
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
    __threadfence_block();
    asm volatile("" ::: "memory");
}

template<int S,int E,typename T,bool FRONT_DRAIN=true,typename RawK>
__forceinline__ __device__ void quant_cache_fp8_register_finish(
    RawK &raw_sK,int tidx,bool valid,const V41Fp8RegisterPacket &packet) {
    static_assert(!T::kFP4Cache && T::kBlockM==128 && T::kBlockN==32 && T::kHeadDim==512 && T::kUsePairedCubes);
    const int row=tidx/8,old_lane=tidx&7;
    const int paired_base=(S/2)*4096+(row>>4)*2048+(row&15)*64
        +(old_lane>>2)*1024+(((old_lane&3)^((row>>1)&3))*16);
    const int first_chunk=paired_base+(row&1)*8;
    const int second_chunk=paired_base+((row&1)^1)*8;
    static_assert(E-S==4);
    auto *dst=cute::raw_pointer_cast(raw_sK.data());
    if constexpr(FRONT_DRAIN) {
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
    }
    { // All256 threads remain active; adjacent4lane groups own pairedtiles.
        #pragma unroll
        for(int group=0;group<(E-S)/2;++group) {
            uint32_t scale_bits2=0;
            {
                uint4 packed{}; // one16B load for16FP8 values
                if(valid) {
                    packed=group==0?uint4{packet.x0,packet.y0,packet.z0,packet.w0}:
                                    uint4{packet.x1,packet.y1,packet.z1,packet.w1};
                    const uint8_t raw=uint8_t(group==0?packet.scale_raw0:packet.scale_raw1);
                    scale_bits2=quant_cache_e8m0_scale2_bits(raw);
                }
                const __nv_bfloat162 scale2=*reinterpret_cast<const __nv_bfloat162*>(&scale_bits2);
                const uint32_t words[4]={packed.x,packed.y,packed.z,packed.w};
                #pragma unroll
                for(int half=0;half<2;++half) {
                    const uint4 out_bits = fp8_words_to_scaled_bf16x8_bits(
                        words[half * 2], words[half * 2 + 1],
                        __bfloat162float(__low2bfloat16(scale2)));
                    const int pos=(half==0?first_chunk:second_chunk)+group*4096;
                    *reinterpret_cast<uint4*>(dst+pos)=out_bits;
                }
            }
        }
    }
    __threadfence_block();
    asm volatile("" ::: "memory");
}

template<typename T>
__forceinline__ __device__ int quant_wi_token_index(
    const typename T::Params &params, int block_idx, int real_end_block_idx, int tidx) {
    const int column=tidx/8;
    const int absolute=block_idx*T::kBlockN+column;
    const bool extra=T::kDecodeFixedExtraIndex || absolute>=params.main_padded_length;
    const int position=extra?absolute-params.main_padded_length:absolute;
    if constexpr(T::kAllValidNoLengths) {
        if(block_idx<0 || block_idx>=real_end_block_idx) return -1;
    } else {
        const int storage=extra?params.extra_storage_topk:params.main_storage_topk;
        const int length=extra?params.extra_length:params.main_length;
        if(position<0 || position>=storage || position>=length) return -1;
    }
    const int *indices=extra?params.extra_indices:params.indices;
    return __ldg(indices+position);
}
template<typename T, bool WRITE_VI, typename TensorVI>
__forceinline__ __device__ void dsa_compute_K_addr(
    const typename T::Params &params,
    typename T::InputT *gK_base,
    int token_idx,                 // prefetched token index (this thread's column)
    int block_idx,                 // block being computed (border fold-in base)
    int idx_in_warpgroup,
    TensorVI &smem_valid_indices,
    int vi_buf,
    V41PrecomputedKAddress<T> &precomp_ptr,
    bool &precomp_valid)
{
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
    if constexpr(T::kDecodeFP8ScopeRow32) {
        precomp_ptr=T::CacheAdapter::template prepare_scope_row32<T>(
            params,token_idx,precomp_valid,block_idx);
    } else
    if constexpr(T::kDecodePageBase64) {
        precomp_ptr=T::CacheAdapter::template prepare_page_base64<T>(
            params,token_idx,precomp_valid,block_idx);
    } else
    {
        precomp_ptr=gK_base;
    }
    if constexpr(WRITE_VI && !T::kAllValidNoLengths) {
        if(idx_in_warpgroup%8==0) smem_valid_indices(vi_buf,idx_in_warpgroup/8)=precomp_valid;
    }
}
template<int S, int E, typename T, bool DO_PREFETCH = true, bool FRONT_DRAIN = true, bool ASYNC_PREISSUED = false,
         bool REGISTER_PREISSUED = false, typename RawK>
__forceinline__ __device__ void dsa_issue_K_load(
    __mbarrier_t *barriers_K,
    V41PrecomputedKAddress<T> precomp_ptr,
    bool precomp_valid,
    int prefetch_block,               // block whose token index is prefetched next
    int real_end_block_idx,           // [Even-align] REAL topk block count -- the only
    int &nxt_token_idx, RawK &raw_sK, int tidx,
    const typename T::Params &params, int quant_token_idx,
    const V41QuantRegisterPacket<T> *register_packet=nullptr, int load_block_idx=-1)
{
    static_assert(T::kDecode);
    if constexpr(T::kQuantCache && REGISTER_PREISSUED) {
        T::CacheAdapter::template register_finish<S,E,T,FRONT_DRAIN>(
            params,raw_sK,tidx,quant_token_idx,precomp_valid,*register_packet);
        cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
    } else {
        if constexpr(T::kDecodeFP8ScopeRow32) {
            T::CacheAdapter::template load_scope_row32<S,E,T,FRONT_DRAIN>(
                params,raw_sK,tidx,quant_token_idx,precomp_valid,load_block_idx,precomp_ptr);
        } else
        if constexpr(T::kDecodePageBase64) {
            T::CacheAdapter::template load_page64<S,E,T,FRONT_DRAIN>(
                params,raw_sK,tidx,quant_token_idx,precomp_valid,load_block_idx,precomp_ptr);
        } else
        {
            T::CacheAdapter::template load<S,E,T,FRONT_DRAIN>(
                params,raw_sK,tidx,quant_token_idx,precomp_valid,load_block_idx,ASYNC_PREISSUED);
        }
        cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
    }
    if constexpr (DO_PREFETCH) {
        if (prefetch_block < real_end_block_idx) {
            nxt_token_idx = quant_wi_token_index<T>(params,prefetch_block,real_end_block_idx,tidx);
        } else {
            nxt_token_idx = -1;
        }
    }
}

}
#include "acc_vreg_fraga.h"
#include "ppuxx/decode/combine/combine.h"

#include "kerutils/common/common.h"
#include "kerutils/host/hardware_info.h"
#include "kerutils/device/ppu/dequant.cuh"

#include <hggc_ad.h>

using namespace cute;
using cutlass::arch::NamedBarrier;

template<typename T, typename Engine, typename Layout>
__forceinline__ __device__ void wg_reduce_softmax_sums(
    float (&rL)[2], Tensor<Engine, Layout> &sL_reduction_wksp,
    int my_row, int warpgroup_idx, int idx_in_warpgroup) {
    if (idx_in_warpgroup % 4 == 0 &&
            (T::kV41PairedPipeline || idx_in_warpgroup < T::kMmaThreads)) {
        sL_reduction_wksp[my_row + warpgroup_idx * 128] = rL[0];
        sL_reduction_wksp[my_row + 8 + warpgroup_idx * 128] = rL[1];
    }
    __syncthreads();

    if constexpr (T::kBlockM == 64) {
        // BlockM=64: Merge WG0 + WG1 partial rL sums.
        // Only primary warps (idx < kMmaThreads) do the +=;
        // after __syncthreads(), ALL threads re-read the merged rL.
        if (warpgroup_idx == 1
                && idx_in_warpgroup % 4 == 0
                && idx_in_warpgroup < T::kMmaThreads) {
            sL_reduction_wksp[my_row] += rL[0];
            sL_reduction_wksp[my_row + 8] += rL[1];
        }
        __syncthreads();
        rL[0] = sL_reduction_wksp[my_row];
        rL[1] = sL_reduction_wksp[my_row + 8];
    } else {
        // BlockM=128: optimized rL cross-WG reduction.
        // WG0 directly reads WG1's partial from register; WG1 uses __syncwarp.
        if (warpgroup_idx == 0) {
            rL[0] += sL_reduction_wksp[my_row + 128];
            rL[1] += sL_reduction_wksp[my_row + 8 + 128];
        } else {
            if (idx_in_warpgroup % 4 == 0) {
                sL_reduction_wksp[my_row] += rL[0];
                sL_reduction_wksp[my_row + 8] += rL[1];
            }
            __syncwarp();
            rL[0] = sL_reduction_wksp[my_row];
            rL[1] = sL_reduction_wksp[my_row + 8];
        }
    }

    // Prune out when rL is 0.0f or NaN
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i)
        rL[i] = (rL[i] == 0.0f || rL[i] != rL[i]) ? 1.0f : rL[i];
}

// e8m0 scale dtype alias (matches kerutils dequant.cuh)
using fp8_e8m0 = __hg_fp8_e8m0;

// ---------------------------------------------------------------------------
// Localized from the patch's flash_splitkv/splitkv_dsa.h: async K tile copy
// helpers for WG kernels, verbatim except for the rename. They are placed in
// namespace flash and drop the dsa_ prefix to avoid ODR clashes with the
// global launch_kv_tiles_dsa_wg* templates in
// prefill/sparse/sparse_prefill_wg.cuh.
// Async K tile copy from GMEM to SMEM for WG kernels (prefill & decode).
// Iterates over head_dim tiles [START, END), issues cp.async per tile,
// then arrives the mbarrier.
// ---------------------------------------------------------------------------
namespace flash {

// Match dense MLA's paired 16x64 K cubes while retaining sparse gather/zfill.
template<int START, int END, typename T, typename TiledCopy,
         typename EG, typename LG, typename ES, typename LS>
__forceinline__ __device__ void launch_kv_pair_copy(
    TiledCopy tiled_copy, Tensor<EG, LG> const &gKV,
    Tensor<ES, LS> &sK, int tidx) {
    auto src = recast<cute::uint128_t>(gKV(_, _, Int<START>{}));
    CUTE_STATIC_ASSERT_V(size(src) == Int<1>{});
    const int row = tidx / 8;
    const int col = ((tidx % 8) ^ (row % 8)) * 8;
    const int offset = START == 8
        ? START * T::kBlockN * 64 + row * 64 + col
        : (START & ~1) * T::kBlockN * 64 + (row / 16) * 2048
            + (START & 1) * 1024 + (row % 16) * 64 + col;
    auto *dst = reinterpret_cast<cute::uint128_t*>(sK.data().get() + offset);
    PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(src[0], *dst, tiled_copy.pred);
    if constexpr (START + 1 < END) {
        launch_kv_pair_copy<START + 1, END, T>(tiled_copy, gKV, sK, tidx);
    }
}

template <
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void launch_kv_tiles_wg_copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV, // (BLOCK_N, HEAD_DIM_K), partitioned
    Tensor<Engine1, Layout1> &sKV)       // (BLOCK_N, HEAD_DIM_K), swizzled, partitioned
{
    Tensor cur_gKV = gKV(_, _, Int<START_HEAD_DIM_TILE_IDX>{});
    Tensor cur_sKV = sKV(_, _, Int<START_HEAD_DIM_TILE_IDX>{});
    cute::copy(tiled_copy, cur_gKV, cur_sKV);

    if constexpr (START_HEAD_DIM_TILE_IDX + 1 < END_HEAD_DIM_TILE_IDX)
    {
        launch_kv_tiles_wg_copy<START_HEAD_DIM_TILE_IDX + 1, END_HEAD_DIM_TILE_IDX>(tiled_copy, gKV, sKV);
    }
}

template <
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void launch_kv_tiles_wg(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV,
    Tensor<Engine1, Layout1> &sKV,
    __mbarrier_t *barriers_K)
{
    launch_kv_tiles_wg_copy<START_HEAD_DIM_TILE_IDX, END_HEAD_DIM_TILE_IDX>(tiled_copy, gKV, sKV);
    cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
}

} // namespace flash

// Here we use MAX_INIT_VAL_SM to initialize sM, and MAX_INIT_VAL for masking
// The reason is that, we need to calculate new_max = max(sM(row_idx), cur_max*scale_softmax_log2)
// so we must guarantee that MAX_INIT_VAL*scale_softmax_log2 < MAX_INIT_VAL_SM
static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;

template<typename Layout>
__forceinline__ __device__ auto pv_rowcol_layout(Layout acc_layout) {
    auto atom_div = logical_divide(acc_layout, Shape<_4>{});
    auto row_div = logical_divide(atom_div, Shape<Shape<_2>>{});
    return make_layout(
        make_layout(get<0, 0, 1>(row_div), get<1>(row_div)),
        make_layout(get<0, 0, 0>(row_div),
                    make_layout(get<0, 1>(row_div), get<2>(row_div))));
}

template<typename T>
__forceinline__ __device__ int pv_row_idx(int mi, int tid) {
    return ((tid / 32) % T::kPvAtomLayoutM) * 16 + (tid % 32) / 4
           + (mi & 1) * 8 + (mi / 2) * (16 * T::kPvAtomLayoutM);
}

template<typename T, typename ER, typename LR, typename ES, typename LS>
__forceinline__ __device__ void pv_scale_one(
    Tensor<ER, LR> &rO, Tensor<ES, LS> const &scale, int tid) {
    auto rc = make_tensor(rO.data(), pv_rowcol_layout(rO.layout()));
    CUTE_STATIC_ASSERT_V(size<0>(rc) == _4{});
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < 4; ++mi) {
        float factor = scale(pv_row_idx<T>(mi, tid));
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(rc); ++ni) rc(mi, ni) *= factor;
    }
}

template <int AtomLayoutM = 8>
__forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup)
{
    // In the layout of fragment A and fragment C during WGMMA, data each thread holds resides in two particular rows. This function converts the local_row_idx (0~2) to the actual row_idx
    // You may refer to this link for the detailed layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
    int row_idx = ((idx_in_warpgroup / 32) % AtomLayoutM) * 16 + local_row_idx * 8 + (idx_in_warpgroup % 32 / 4);
    return row_idx;
}

template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_acc(Tensor<Engine, Layout> const &tensor)
{
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    NumericArrayConverterPPU<To_type, From_type, numel> convert_op;
    // convert_op returns frag from reinterpret_cast of tensor data
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// Launch TMA copy for a range of KV tile
// A tile has a shape of BlockN (32) x 64
template <
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void launch_kv_tiles_copy_aiu(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV,       // (BLOCK_N, HEAD_DIM_K), swizzled
    const Flash_fwd_mla_params &params,
    __mbarrier_t *barriers_K,
    int warp_idx)
{
    Tensor cur_gKV = gKV(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
    Tensor cur_sKV = sKV(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
    cute::copy(tiled_copy, cur_gKV, cur_sKV);

    if constexpr (START_HEAD_DIM_TILE_IDX + 1 < END_HEAD_DIM_TILE_IDX) {
        launch_kv_tiles_copy_aiu<START_HEAD_DIM_TILE_IDX + 1, END_HEAD_DIM_TILE_IDX>(tiled_copy, gKV, sKV, params, barriers_K, warp_idx);
    }
}

// Launch TMA copy for a range of KV tile
// A tile has a shape of BLOCK_N (32) x 64
template <
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void launch_kv_tiles_copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &gKV, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV,       // (BLOCK_N, HEAD_DIM_K), swizzled
    const Flash_fwd_mla_params &params,
    __mbarrier_t *barriers_K,
    int warp_idx)
{
    launch_kv_tiles_copy_aiu<START_HEAD_DIM_TILE_IDX, END_HEAD_DIM_TILE_IDX>(tiled_copy, gKV, sKV, params, barriers_K, warp_idx);
    cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PPU: shared memory not support init by zero, need clear if not align.
template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = true, bool Clear_OOB_K = true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                     Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                     Tensor<Engine3, Layout3> const &predicate_K, const int max_MN = 0)
{

    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});

    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D)); // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D)); // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D)); // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

// Prefetch some KV tiles
// Currently this is not used because it leads to performance degradation
// template<
//     int START_HEAD_DIM_TILE_IDX,
//     int END_HEAD_DIM_TILE_IDX,
//     typename TMA_K_OneTile,
//     typename Engine0, typename Layout0
// >
// __forceinline__ __device__ void prefetch_kv_tiles(
//     Tensor<Engine0, Layout0> const &gKV,	// (BLOCK_N, HEAD_DIM_K)
//     TMA_K_OneTile &tma_K,
//     int idx_in_warpgroup
// ) {
//     if (idx_in_warpgroup == 0) {
//         auto thr_tma = tma_K.get_slice(_0{});
//         Tensor cur_gKV = thr_tma.partition_S(gKV)(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
//         cute::prefetch(tma_K, cur_gKV);
//         if constexpr (START_HEAD_DIM_TILE_IDX+1 < END_HEAD_DIM_TILE_IDX) {
//             prefetch_kv_tiles<START_HEAD_DIM_TILE_IDX+1, END_HEAD_DIM_TILE_IDX>(gKV, tma_K, idx_in_warpgroup);
//         }
//     }
// }

// Adapted from https://github.com/Dao-AILab/flash-attention/blob/cdaf2de6e95cb05400959b5ab984f66e4c7df317/hopper/utils.h
// * Copyright (c) 2024, Tri Dao.
template <
    typename Tensor0, typename Tensor1,
    typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm(Tensor2 &tCrC, Tensor0 const &tCrA, Tensor1 const &tCrB, TiledMma &tiled_mma)
{
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(tCrC)); // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), tCrC);
    }
}

__forceinline__ __device__ void kernel_sleep_ns()
{
    __nanosleep(1);
}

// Wait for one KV-tile to be ready, and then calculate P += Q K^T for one Q-tile (BLOCK_SIZE_Mx64) and one KV-tile (PAGE_BLOCK_SIZEx64)
// The Q-tile should be in shared memory
template <
    bool InterleaveLoads,
    typename TiledMMA,
    typename TiledCopyA,
    typename TiledCopyB,
    typename ThrCopyA,
    typename ThrCopyB,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4>
__forceinline__ __device__ void qkt_gemm_one_tile_sQ(
    TiledMMA &tiled_mma,
    TiledCopyA &smem_tiled_copy_Q,
    TiledCopyB &smem_tiled_copy_K,
    ThrCopyA &smem_thr_copy_Q,
    ThrCopyB &smem_thr_copy_K,
    Tensor<Engine0, Layout0> const &sQ_tiled,         // (MMA, 1, 4)
    Tensor<Engine1, Layout1> const &thr_mma_sQ_tile,  // (MMA, 1, 4)
    Tensor<Engine2, Layout2> const &sKV_tiled,        // (MMA, 1, 4)
    Tensor<Engine3, Layout3> const &thr_mma_sKV_tile, // (MMA, 1, 4)
    Tensor<Engine4, Layout4> &rP,                     // ((2, 2, 8), 1, 1)
    int idx_in_warpgroup)
{
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor rQ = thr_mma.partition_fragment_A(sQ_tiled);
    Tensor rK = thr_mma.partition_fragment_B(sKV_tiled);

    Tensor rQ_copy_view = smem_thr_copy_Q.retile_D(rQ);
    CUTE_STATIC_ASSERT_V(size<1>(thr_mma_sQ_tile) == size<1>(rQ_copy_view)); // M

    Tensor rK_copy_view = smem_thr_copy_K.retile_D(rK);
    CUTE_STATIC_ASSERT_V(size<1>(thr_mma_sKV_tile) == size<1>(rK_copy_view)); // M

#if ACOMPUTE_VERSION >= 10500
    if constexpr (InterleaveLoads) {
        // Keep independent slice slots while overlapping Q/K supply and MMA.
        CUTE_STATIC_ASSERT_V(size<2>(rQ_copy_view) == _4{});
        CUTE_STATIC_ASSERT_V(size<2>(rK_copy_view) == _4{});
        auto load_slice = [&](auto k) {
            cute::copy(smem_tiled_copy_Q, thr_mma_sQ_tile(_, _, k), rQ_copy_view(_, _, k));
            cute::copy(smem_tiled_copy_K, thr_mma_sKV_tile(_, _, k), rK_copy_view(_, _, k));
        };
        load_slice(_0{});
        load_slice(_1{});
        cute::gemm(tiled_mma, rQ_copy_view(_, _, _0{}), rK_copy_view(_, _, _0{}), rP);
        load_slice(_2{});
        cute::gemm(tiled_mma, rQ_copy_view(_, _, _1{}), rK_copy_view(_, _, _1{}), rP);
        load_slice(_3{});
        cute::gemm(tiled_mma, rQ_copy_view(_, _, _2{}), rK_copy_view(_, _, _2{}), rP);
        cute::gemm(tiled_mma, rQ_copy_view(_, _, _3{}), rK_copy_view(_, _, _3{}), rP);
        return;
    }
#endif
    cute::copy(smem_tiled_copy_Q, thr_mma_sQ_tile, rQ_copy_view);

    cute::copy(smem_tiled_copy_K, thr_mma_sKV_tile, rK_copy_view);

    cute::gemm(tiled_mma, rQ_copy_view(_, _, _0{}), rK_copy_view(_, _, _0{}), rP);
    cute::gemm(tiled_mma, rQ_copy_view(_, _, _1{}), rK_copy_view(_, _, _1{}), rP);
    cute::gemm(tiled_mma, rQ_copy_view(_, _, _2{}), rK_copy_view(_, _, _2{}), rP);
    cute::gemm(tiled_mma, rQ_copy_view(_, _, _3{}), rK_copy_view(_, _, _3{}), rP);
}

// Stream a ready four-tile QK interval through four independent slice slots.
template <int FirstTile, typename T, typename TiledMMA,
          typename CopyQ, typename CopyK, typename ThrQ, typename ThrK,
          typename SQ, typename SK, typename SourceQ, typename SourceK, typename RP>
__forceinline__ __device__ void qkt_gemm_four_tiles_sQ(
    TiledMMA &mma, CopyQ &copy_q, CopyK &copy_k, ThrQ &thr_q, ThrK &thr_k,
    SQ const &sQ, SK const &sK, SourceQ const &src_q, SourceK const &src_k,
    RP &rP, int cute_idx)
{
    auto thr_mma = mma.get_slice(cute_idx);
    Tensor rQ = thr_mma.partition_fragment_A(sQ(_, _, Int<FirstTile>{}));
    Tensor rK = thr_mma.partition_fragment_B(sK(_, _, Int<FirstTile>{}));
    Tensor cq = thr_q.retile_D(rQ);
    Tensor ck = thr_k.retile_D(rK);
    CUTE_STATIC_ASSERT_V(size<2>(cq) == _4{});
    CUTE_STATIC_ASSERT_V(size<2>(ck) == _4{});
    auto load = [&](auto k) {
        constexpr int slice = decltype(k)::value % 4;
        constexpr int tile = FirstTile + decltype(k)::value / 4;
        cute::copy(copy_q, src_q(_, _, Int<slice>{}, Int<tile>{}), cq(_, _, Int<slice>{}));
        cute::copy(copy_k, src_k(_, _, Int<slice>{}, Int<tile>{}), ck(_, _, Int<slice>{}));
    };
    for_each(make_int_sequence<4>{}, load);
    for_each(make_int_sequence<16>{}, [&](auto k) {
        constexpr int slice = decltype(k)::value % 4;
        cute::gemm(mma, cq(_, _, Int<slice>{}), ck(_, _, Int<slice>{}), rP);
        if constexpr (decltype(k)::value + 4 < 16) {
            load(Int<decltype(k)::value + 4>{});
        }
    });
}

template <
    typename TiledMMA,
    typename TiledCopy,
    typename ThrCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3>
__forceinline__ __device__ void qkt_gemm_one_tile_rQ(
    TiledMMA &tiled_mma,
    TiledCopy &smem_tiled_copy_K,
    ThrCopy &smem_thr_copy_K,
    Tensor<Engine0, Layout0> const &thr_mma_rQ_tile,  // (MMA, 1, 4)
    Tensor<Engine1, Layout1> const &sKV_tiled,        // (MMA, 1, 4)
    Tensor<Engine2, Layout2> const &thr_mma_sKV_tile, // (MMA, 1, 4)
    Tensor<Engine3, Layout3> &rP,                     // ((2, 2, 8), 1, 1)
    int idx_in_warpgroup)
{
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor rK = thr_mma.partition_fragment_B(sKV_tiled);
    Tensor rK_copy_view = smem_thr_copy_K.retile_D(rK);
    CUTE_STATIC_ASSERT_V(size<1>(thr_mma_sKV_tile) == size<1>(rK_copy_view)); // M

    cute::copy(smem_tiled_copy_K, thr_mma_sKV_tile, rK_copy_view);

    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _0{}), rK_copy_view(_, _, _0{}), rP);
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _1{}), rK_copy_view(_, _, _1{}), rP);
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _2{}), rK_copy_view(_, _, _2{}), rP);
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _3{}), rK_copy_view(_, _, _3{}), rP);
}

// Pipelined TMA wait and Q K^T gemm
// In order to overlap memory copy (G->S copy for K) and computation, we divide both Q and K into tiles of shape (BLOCK_SIZE_M, 64), and (BLOCK_N, 64) respectively, and then do the computation as follows:
// - Wait for the 0-th tile to be ready using `barrier.wait()`
// - Compute Q K^T for the 0-th tile
// - Wait for the 1-st tile to be ready
// - Compute Q K^T for the 1-st tile
// ...
// This gives latter tiles more time to be ready, and thus can overlap the memory copy and computation
template <
    typename T,    // Traits
    int PHASE_IDX, // See comments in the code
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3>
__forceinline__ __device__ void warpgroup_cooperative_qkt_gemm(
    Tensor<Engine0, Layout0> &sQ,   // (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV0, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV1, // (BLOCK_N, HEAD_DIM_K)
    Tensor<Engine2, Layout2> &rP,   // ((2, 2, 8), 1, 1)
    Tensor<Engine3, Layout3> &rQ8,  // The 8-th tile of Q. We store it separately to leave some room for storing sP1
    __mbarrier_t *barriers,
    bool &cur_phase,
    int idx_in_warpgroup,
    const int warp_idx)
{
    typename T::TiledMma tiled_mma;
    // Wrap thread index for BlockM=64 (4-warp TiledMMA, 8-warp WG)
    const int cute_idx = T::kV41PairedPipeline
        ? idx_in_warpgroup : idx_in_warpgroup % T::kMmaThreads;
    const int cute_warp = T::kV41PairedPipeline
        ? warp_idx * 32 : (warp_idx % T::kAtomLayoutM) * 32;
    constexpr bool interleave_loads = [] {
        if constexpr (T::kV41PairedPipeline) return false;
        else return T::CvtGemmSwzlLd;
    }();
    ThrMMA thr_mma = tiled_mma.get_slice(cute_idx);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename T::SmemCopyAtomK{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(cute_warp);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename T::SmemCopyAtomQ{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(cute_warp);

    Tensor sQ_tiled = flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, _64>{})(_, _, _0{}, _); // (BLOCK_SIZE_M, 64, 9)
    Tensor sKV0_tiled = flat_divide(sKV0, Shape<Int<T::kBlockN>, _64>{})(_, _, _0{}, _);  // (BLOCK_N, 64, 9)
    Tensor sKV1_tiled = flat_divide(sKV1, Shape<Int<T::kBlockN>, _64>{})(_, _, _0{}, _);  // (BLOCK_N, 64, 9)
    Tensor thr_mma_sQ_tiled = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ_tiled));
    Tensor thr_mma_sKV0_tiled = smem_thr_copy_K.partition_S(make_mix_tensor_like(sKV0_tiled));
    Tensor thr_mma_sKV1_tiled = smem_thr_copy_K.partition_S(make_mix_tensor_like(sKV1_tiled));

    #define QKT_GEMM_ONE_TILE(TILE_IDX) \
        if constexpr(TILE_IDX == 8) { \
            qkt_gemm_one_tile_rQ(tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, \
                    rQ8, sKV1_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV1_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, cute_idx); \
        } else if constexpr(TILE_IDX < 4) { \
            qkt_gemm_one_tile_sQ<interleave_loads>(tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, \
                    smem_thr_copy_Q, smem_thr_copy_K, \
                    sQ_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sQ_tiled(_, _, _, Int<TILE_IDX>{}), \
                    sKV0_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV0_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, cute_idx); \
        } else  { \
            qkt_gemm_one_tile_sQ<interleave_loads>(tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, \
                    smem_thr_copy_Q, smem_thr_copy_K, \
                    sQ_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sQ_tiled(_, _, _, Int<TILE_IDX>{}), \
                    sKV1_tiled(_, _, Int<TILE_IDX>{}), thr_mma_sKV1_tiled(_, _, _, Int<TILE_IDX>{}), \
                    rP, cute_idx); \
        }

    auto gemm_four_tiles = [&](auto first) {
        constexpr int start = decltype(first)::value;
#if ACOMPUTE_VERSION >= 10500
        if constexpr (interleave_loads && T::kHeadDim == 576) {
            auto &sK = start == 0 ? sKV0_tiled : sKV1_tiled;
            auto &src_k = start == 0 ? thr_mma_sKV0_tiled : thr_mma_sKV1_tiled;
            qkt_gemm_four_tiles_sQ<start, T>(tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K, sQ_tiled, sK, thr_mma_sQ_tiled, src_k,
                rP, cute_idx);
        } else
#endif
        {
            QKT_GEMM_ONE_TILE(start);
            QKT_GEMM_ONE_TILE(start + 1);
            QKT_GEMM_ONE_TILE(start + 2);
            QKT_GEMM_ONE_TILE(start + 3);
        }
    };

    if constexpr (PHASE_IDX == 0) {
        // In PHASE-0, warpgroup 0 calculates Q K^T for the first 4 tiles
        while (!cutlass::arch::test_wait(&barriers[0], cur_phase, 1)) {
            kernel_sleep_ns();
        };

        gemm_four_tiles(_0{});
    } else if constexpr (PHASE_IDX == 1) {
        // In PHASE-1, warpgroup 1 calculates Q K^T for all the 9 tiles
        while (!cutlass::arch::test_wait(&barriers[1], cur_phase, 1)) {
            kernel_sleep_ns();
        };

        gemm_four_tiles(_4{});
        if constexpr (T::kHasExtraRopeTile) { QKT_GEMM_ONE_TILE(8); }

        while (!cutlass::arch::test_wait(&barriers[0], cur_phase, 1)) {
            kernel_sleep_ns();
        };
        gemm_four_tiles(_0{});
        cur_phase = (cur_phase + 1) & 1;
    } else {
        // In PHASE-2, warpgroup 0 calculates Q K^T for the last 5 tiles
        static_assert(PHASE_IDX == 2);

        while (!cutlass::arch::test_wait(&barriers[1], cur_phase, 1)) {
            kernel_sleep_ns();
        };

        gemm_four_tiles(_4{});
        if constexpr (T::kHasExtraRopeTile) { QKT_GEMM_ONE_TILE(8); }
        cur_phase = (cur_phase + 1) & 1;
    }
}

template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2>
__forceinline__ __device__ void warpgroup_cooperative_qkt_gemm_no_pipeline(
    Tensor<Engine0, Layout0> &rQ,  // (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV, // (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine2, Layout2> &rP,  // ((2, 2, 8), 1, 1)
    int idx_in_warpgroup)
{
    typename T::TiledMma tiled_mma;
    const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
    ThrMMA thr_mma = tiled_mma.get_slice(cute_idx);

    Tensor rK = thr_mma.partition_fragment_B(sKV); // (MMA, 1, 576/16=36)
    auto smem_tiled_copy_K = make_tiled_copy_B(typename T::SmemCopyAtomK{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(cute_idx);
    Tensor rK_copy_view = smem_thr_copy_K.retile_D(rK);
    auto tSsK = smem_thr_copy_K.partition_S(make_mix_tensor_like(sKV));
    CUTE_STATIC_ASSERT_V(size<1>(tSsK) == size<1>(rK_copy_view)); // M
    cute::copy(smem_tiled_copy_K, tSsK, rK_copy_view);

    cute::clear(rP);
    gemm(rP, rQ, rK, tiled_mma);
}

// Compute O += PV, where P resides in register
template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2>
__forceinline__ __device__ void warpgroup_cooperative_pv_gemm_localP(
    Tensor<Engine0, Layout0> &rP,       // ((2, 2, 8), 1, 1), fragment A layout
    Tensor<Engine1, Layout1> &sKV_half, // (HEAD_DIM_V/2, BLOCK_N)
    Tensor<Engine2, Layout2> &rO,       // ((2, 2, 32), 1, 1)
    int idx_in_warpgroup,
    int warp_idx)
{
    // const int warp_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);
    typename T::TiledMma tiled_mma;
    const int cute_idx = T::kV41PairedPipeline ? idx_in_warpgroup : idx_in_warpgroup % T::kMmaThreads;
    const int cute_warp = T::kV41PairedPipeline ? warp_idx * 32 : (warp_idx % T::kAtomLayoutM) * 32;
    ThrMMA thr_mma = tiled_mma.get_slice(cute_idx);

    auto smem_tiled_copy_Vt = make_tiled_copy_B(typename T::SmemCopyAtomVt{}, tiled_mma);
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(cute_warp);
    Tensor rVt = thr_mma.partition_fragment_B(sKV_half);
    Tensor rVt_copy_view = smem_thr_copy_Vt.retile_D(rVt);
    auto tSsVt = smem_thr_copy_Vt.partition_S(make_mix_tensor_like(sKV_half));

    CUTE_STATIC_ASSERT_V(size<1>(tSsVt) == size<1>(rVt_copy_view)); // M
    cute::copy(smem_tiled_copy_Vt, tSsVt, rVt_copy_view);

    gemm(rO, rP, rVt, tiled_mma);
}

// Compute O += PV, where P resides in shared memory
template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2>
__forceinline__ __device__ void warpgroup_cooperative_pv_gemm_remoteP(
    Tensor<Engine0, Layout0> &sP,
    Tensor<Engine1, Layout1> &sKV_half, // (HEAD_DIM_V/2, BLOCK_N)
    Tensor<Engine2, Layout2> &rO,       // ((2, 2, 32), 1, 1)
    int idx_in_warpgroup,
    int warp_idx)
{
    typename T::TiledMmaPV tiled_mma;
    const int cute_idx = T::kV41PairedPipeline ? idx_in_warpgroup : idx_in_warpgroup % T::kMmaThreads;
    const int cute_warp = T::kV41PairedPipeline ? warp_idx * 32 : (warp_idx % T::kAtomLayoutM) * 32;
    auto smem_tiled_copy_P = make_tiled_copy_A(typename T::SmemCopyAtomPV{}, tiled_mma);
    const int p_copy_idx = T::kUsePv4x2
        ? (__builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32) % T::kPvAtomLayoutM) * 32
        : cute_idx;
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(p_copy_idx);
    auto smem_tiled_copy_Vt = make_tiled_copy_B(typename T::SmemCopyAtomVt{}, tiled_mma);
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(cute_warp);

    auto tSsP = [&]() {
        if constexpr (T::kUsePv4x2) {
            return smem_thr_copy_P.partition_S(make_mix_tensor_like(sP));
        } else {
            return smem_thr_copy_P.partition_S(sP);
        }
    }();
    auto tSsVt = smem_thr_copy_Vt.partition_S(make_mix_tensor_like(sKV_half));

    ThrMMA thr_mma = tiled_mma.get_slice(cute_idx);
    Tensor thr_mma_sP = thr_mma.partition_fragment_A(sP);
    Tensor thr_mma_sKV_half = thr_mma.partition_fragment_B(sKV_half); // (MMA, 1, 64/16=4)

    Tensor rP_copy_view = smem_thr_copy_P.retile_D(thr_mma_sP);
    Tensor rVt_copy_view = smem_thr_copy_Vt.retile_D(thr_mma_sKV_half);

    cute::copy(smem_tiled_copy_P, tSsP, rP_copy_view);
    cute::copy(smem_tiled_copy_Vt, tSsVt, rVt_copy_view);
    gemm(rO, rP_copy_view, rVt_copy_view, tiled_mma);
}

#if ACOMPUTE_VERSION == 10000
template<
    typename T,
    bool DO_OOB_FILLING,
    // typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ auto wg0_bunch_0(
    // Tensor<Engine0, Layout0> &rPb,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &rP0,	// ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rO0,	// ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sScale0,	// (BLOCK_SIZE_M)
    Tensor<Engine4, Layout4> &sM,	// (BLOCK_SIZE_M)
    float rL[2],
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf)
{
    // Preload smem_valid_indices into registers to hide SMEM latency
    // from the __shfl_xor_sync critical path.
    // ACOMPUTE_VERSION==10000: each thread needs 8 values (2 groups of 4)
    int r_valid[8];
    {
        int lane4 = idx_in_warpgroup % 4;
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            int base = (k * 16 + lane4) % T::kBlockN;
            r_valid[k*4]   = smem_valid_indices(valid_indices_buf, base);
            r_valid[k*4+1] = smem_valid_indices(valid_indices_buf, (base + 4) % T::kBlockN);
            r_valid[k*4+2] = smem_valid_indices(valid_indices_buf, (base + 8) % T::kBlockN);
            r_valid[k*4+3] = smem_valid_indices(valid_indices_buf, (base + 12) % T::kBlockN);
        }
    }

    // This piece of code is tightly coupled [Accumulate's layout](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png)
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 4 : 0; i < size(rP0); i += 8) {
            int k_base = ((i/8) % 2) * 4;
            rP0(i)   = r_valid[k_base]     ? rP0(i)   : MAX_INIT_VAL;
            rP0(i+1) = r_valid[k_base + 1] ? rP0(i+1) : MAX_INIT_VAL;
            rP0(i+2) = r_valid[k_base + 2] ? rP0(i+2) : MAX_INIT_VAL;
            rP0(i+3) = r_valid[k_base + 3] ? rP0(i+3) : MAX_INIT_VAL;
            cur_max = max(cur_max, max(max(rP0(i), rP0(i+1)), max(rP0(i+2), rP0(i+3))));
        }

        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

        // Update sM and sL
        cur_max *= scale_softmax_log2;
        // TOCTOU fix: read sM once to avoid race with concurrent warpgroup.
        float old_max = sM(row_idx);
        float new_max = max(old_max, cur_max);
        float scale_for_old = exp2f(old_max - new_max);

        __syncwarp(); // Make sure all reads have finished before updating sM

        // For kAtomLayoutM < 8, warps kAtomLayoutM..7 alias the same SMEM rows
        // as warps 0..(kAtomLayoutM-1). Only primary warps write sM/sScale to
        // avoid intra-WG write-write races.
        if (idx_in_warpgroup % 4 == 0 && idx_in_warpgroup < T::kMmaThreads) {
            sScale0(row_idx) = scale_for_old;
            sM(row_idx) = new_max;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 4 : 0; i < size(rP0); i += 8) {
            rP0(i) = exp2f(rP0(i)*scale_softmax_log2 - new_max);
            rP0(i+1) = exp2f(rP0(i+1)*scale_softmax_log2 - new_max);
            rP0(i+2) = exp2f(rP0(i+2)*scale_softmax_log2 - new_max);
            rP0(i+3) = exp2f(rP0(i+3)*scale_softmax_log2 - new_max);
            cur_sum += (rP0(i) + rP0(i+1) + rP0(i+2) + rP0(i+3));
        }

        rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
    }

    auto rPb_ret = convert_acc<typename T::InputT>(rP0);
    return rPb_ret;
}

template<
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool IS_BLK2_LAST,
    // typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename EngineVI, typename LayoutVI>
__forceinline__ __device__ auto wg1_bunch_0(
    // Tensor<Engine0, Layout0> &rP1b,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &sScale1, // (BLOCK_SIZE_M)
    Tensor<Engine2, Layout2> &rO1,     // ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sM,      // (BLOCK_SIZE_M)
    float rL[2],
    Tensor<Engine4, Layout4> const &sScale0, // (BLOCK_SIZE_M)
    Tensor<Engine5, Layout5> &rP1,           // ((2, 2, 8), 1, 1)
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf,
    float r_cur_max_in[2] = nullptr)  // if non-null, skip mask+max (pre-computed)
{
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);

        float cur_max;
        if (r_cur_max_in) {
            cur_max = r_cur_max_in[local_row_idx];
        } else {
            // Preload smem_valid_indices into registers
            int r_valid[8];
            {
                int lane4 = idx_in_warpgroup % 4;
                #pragma unroll
                for (int k = 0; k < 2; k++) {
                    int base = (k * 16 + lane4) % T::kBlockN;
                    r_valid[k*4]   = smem_valid_indices(valid_indices_buf, base);
                    r_valid[k*4+1] = smem_valid_indices(valid_indices_buf, (base + 4) % T::kBlockN);
                    r_valid[k*4+2] = smem_valid_indices(valid_indices_buf, (base + 8) % T::kBlockN);
                    r_valid[k*4+3] = smem_valid_indices(valid_indices_buf, (base + 12) % T::kBlockN);
                }
            }
            // Mask, and get row-wise max
            cur_max = MAX_INIT_VAL;
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 4 : 0; i < size(rP1); i += 8) {
                if constexpr (IS_BLK0_LAST) {
                    // wg0's block was the last; wg1 has no real block.
                    rP1(i) = rP1(i + 1) = rP1(i + 2) = rP1(i + 3) = MAX_INIT_VAL;
                } else {
                    int k_base = ((i / 8) % 2) * 4;
                    rP1(i)     = r_valid[k_base]     ? rP1(i)     : MAX_INIT_VAL;
                    rP1(i + 1) = r_valid[k_base + 1] ? rP1(i + 1) : MAX_INIT_VAL;
                    rP1(i + 2) = r_valid[k_base + 2] ? rP1(i + 2) : MAX_INIT_VAL;
                    rP1(i + 3) = r_valid[k_base + 3] ? rP1(i + 3) : MAX_INIT_VAL;
                }
                cur_max = max(cur_max, max(max(rP1(i), rP1(i + 1)), max(rP1(i + 2), rP1(i + 3))));
            }
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

            cur_max *= scale_softmax_log2;
        }

        float old_max = sM(row_idx);
        float new_max = max(old_max, cur_max);
        float scale_for_old = exp2f(old_max - new_max);

        __syncwarp();
        // Only primary warps write sM/sScale (prevents aliased warp double-write).
        if (idx_in_warpgroup % 4 == 0 && idx_in_warpgroup < T::kMmaThreads) {
            sM(row_idx) = new_max;
            sScale1(row_idx) = scale_for_old;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        if constexpr (!IS_BLK0_LAST) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 4 : 0; i < size(rP1); i += 8) {
                rP1(i) = exp2f(rP1(i) * scale_softmax_log2 - new_max);
                rP1(i + 1) = exp2f(rP1(i + 1) * scale_softmax_log2 - new_max);
                rP1(i + 2) = exp2f(rP1(i + 2) * scale_softmax_log2 - new_max);
                rP1(i + 3) = exp2f(rP1(i + 3) * scale_softmax_log2 - new_max);
                cur_sum += (rP1(i) + rP1(i + 1) + rP1(i + 2) + rP1(i + 3));
            }
        }

        float cur_scale_for_o1 = scale_for_old * sScale0(row_idx);

        // Update rL
        rL[local_row_idx] = rL[local_row_idx]*cur_scale_for_o1 + cur_sum;
    }

    auto rP1b_ret = convert_acc<typename T::InputT>(rP1);
    return rP1b_ret;
}
#endif

// The output-parameter overload also serves the paired SM89 pipeline.
template <
    typename T,
    bool DO_OOB_FILLING,
    bool query_all_valid = false,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename EngineVI, typename LayoutVI>
__forceinline__ __device__ void wg0_bunch_0(
    Tensor<Engine0, Layout0> &rPb,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &rP0,     // ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rO0,     // ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sScale0, // (BLOCK_SIZE_M)
    Tensor<Engine4, Layout4> &sM,      // (BLOCK_SIZE_M)
    float rL[2],
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf)
{
    // Preload smem_valid_indices into registers to hide SMEM latency
    // from the __shfl_xor_sync critical path.
    // #else path: each thread needs 8 values (4 groups of 2)
    int r_valid[8];
    if constexpr (!query_all_valid) {
        int lane4 = idx_in_warpgroup % 4;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int base = (k * 8 + lane4 * 2) % T::kBlockN;
            r_valid[k*2]   = smem_valid_indices(valid_indices_buf, base);
            r_valid[k*2+1] = smem_valid_indices(valid_indices_buf, (base + 1) % T::kBlockN);
        }
    }

     // This piece of code is tightly coupled [Accumulate's layout](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png)
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);

        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            if constexpr (!query_all_valid) {
                int k_base = ((i/4) % 4) * 2;
                rP0(i)   = r_valid[k_base]     ? rP0(i)   : MAX_INIT_VAL;
                rP0(i+1) = r_valid[k_base + 1] ? rP0(i+1) : MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(rP0(i), rP0(i+1)));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

        // Update sM and sL
        cur_max *= scale_softmax_log2;
        float new_max = max(sM(row_idx), cur_max);
        float scale_for_old = exp2f(sM(row_idx) - new_max);
        __syncwarp();   // Make sure all reads have finished before updating sM
        if (idx_in_warpgroup%4 == 0 && idx_in_warpgroup < T::kMmaThreads) {
            sScale0(row_idx) = scale_for_old;
            sM(row_idx) = new_max;
        }
        
        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            rP0(i) = exp2f(rP0(i)*scale_softmax_log2 - new_max);
            rP0(i+1) = exp2f(rP0(i+1)*scale_softmax_log2 - new_max);
            rPb(i) = (typename T::InputT)rP0(i);
            rPb(i+1) = (typename T::InputT)rP0(i+1);
            cur_sum += rP0(i) + rP0(i+1);
        }
        rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
    }
}

template <
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool IS_BLK2_LAST,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename EngineVI, typename LayoutVI>
__forceinline__ __device__ auto wg1_bunch_0(
    Tensor<Engine0, Layout0> &rP1b,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &sScale1, // (BLOCK_SIZE_M)
    Tensor<Engine2, Layout2> &rO1,     // ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sM,      // (BLOCK_SIZE_M)
    float rL[2],
    Tensor<Engine4, Layout4> const &sScale0, // (BLOCK_SIZE_M)
    Tensor<Engine5, Layout5> &rP1,           // ((2, 2, 8), 1, 1)
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf,
    float r_cur_max_in[2] = nullptr)  // if non-null, skip mask+max (pre-computed)
{
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);

        float cur_max;
        if (r_cur_max_in) {
            cur_max = r_cur_max_in[local_row_idx];
        } else {
            // Preload smem_valid_indices into registers
            int r_valid[8];
            {
                int lane4 = idx_in_warpgroup % 4;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    int base = (k * 8 + lane4 * 2) % T::kBlockN;
                    r_valid[k*2]   = smem_valid_indices(valid_indices_buf, base);
                    r_valid[k*2+1] = smem_valid_indices(valid_indices_buf, (base + 1) % T::kBlockN);
                }
            }
            // Mask, and get row-wise max
            cur_max = MAX_INIT_VAL;
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
                if constexpr (IS_BLK0_LAST) {
                    rP1(i) = rP1(i+1) = MAX_INIT_VAL;
                } else {
                    int k_base = ((i/4) % 4) * 2;
                    rP1(i)   = r_valid[k_base]     ? rP1(i)   : MAX_INIT_VAL;
                    rP1(i+1) = r_valid[k_base + 1] ? rP1(i+1) : MAX_INIT_VAL;
                }
                cur_max = max(cur_max, max(rP1(i), rP1(i+1)));
            }

            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
            cur_max *= scale_softmax_log2;
        }

        float old_max = sM(row_idx);
        float new_max = max(old_max, cur_max);
        float scale_for_old = exp2f(old_max - new_max);
        __syncwarp();
        // Only primary warps write sM/sScale (prevents aliased warp double-write).
        if (idx_in_warpgroup%4 == 0 && idx_in_warpgroup < T::kMmaThreads) {
            sM(row_idx) = new_max;
            sScale1(row_idx) = scale_for_old;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        if constexpr (!IS_BLK0_LAST) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
                rP1(i) = exp2f(rP1(i)*scale_softmax_log2 - new_max);
                rP1(i+1) = exp2f(rP1(i+1)*scale_softmax_log2 - new_max);
                rP1b(i) = (typename T::InputT)rP1(i);
                rP1b(i+1) = (typename T::InputT)rP1(i+1);
                cur_sum += rP1(i) + rP1(i+1);
            }
        }

        float cur_scale_for_o1 = scale_for_old * sScale0(row_idx);

        // Update rL
        rL[local_row_idx] = rL[local_row_idx]*cur_scale_for_o1 + cur_sum;
    }
}

// wg1_bunch_0_pre: compute cur_max before barrier (doesn't depend on sM)
// Masks rP1 and computes per-row cur_max via warp shuffle reduce.
template<
    typename T,
    bool IS_BLK0_LAST,
    bool query_all_valid = false,
    typename Engine5, typename Layout5,
    typename EngineVI, typename LayoutVI>
__forceinline__ __device__ void wg1_bunch_0_pre(
    float r_cur_max[2],                // output: per-row cur_max * scale
    Tensor<Engine5, Layout5> &rP1,           // ((2, 2, 8), 1, 1)
    float scale_softmax_log2,
    int idx_in_warpgroup,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    int valid_indices_buf)
{
    int r_valid[8];
    int lane4 = idx_in_warpgroup % 4;
#if ACOMPUTE_VERSION == 10000
    if constexpr (!T::kV41PairedPipeline) {
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        int base = (k * 16 + lane4) % T::kBlockN;
        r_valid[k*4]   = smem_valid_indices(valid_indices_buf, base);
        r_valid[k*4+1] = smem_valid_indices(valid_indices_buf, (base + 4) % T::kBlockN);
        r_valid[k*4+2] = smem_valid_indices(valid_indices_buf, (base + 8) % T::kBlockN);
        r_valid[k*4+3] = smem_valid_indices(valid_indices_buf, (base + 12) % T::kBlockN);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 4 : 0; i < size(rP1); i += 8) {
            if constexpr (IS_BLK0_LAST) {
                rP1(i) = rP1(i + 1) = rP1(i + 2) = rP1(i + 3) = MAX_INIT_VAL;
            } else {
                int k_base = ((i / 8) % 2) * 4;
                rP1(i)     = r_valid[k_base]     ? rP1(i)     : MAX_INIT_VAL;
                rP1(i + 1) = r_valid[k_base + 1] ? rP1(i + 1) : MAX_INIT_VAL;
                rP1(i + 2) = r_valid[k_base + 2] ? rP1(i + 2) : MAX_INIT_VAL;
                rP1(i + 3) = r_valid[k_base + 3] ? rP1(i + 3) : MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(max(rP1(i), rP1(i + 1)), max(rP1(i + 2), rP1(i + 3))));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
        r_cur_max[local_row_idx] = cur_max * scale_softmax_log2;
    }
    } else
#endif
    {
    if constexpr (!T::kV41PairedPipeline || (!IS_BLK0_LAST && !query_all_valid)) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int base = (k * 8 + lane4 * 2) % T::kBlockN;
        r_valid[k*2]   = smem_valid_indices(valid_indices_buf, base);
        r_valid[k*2+1] = smem_valid_indices(valid_indices_buf, (base + 1) % T::kBlockN);
    }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
            if constexpr (IS_BLK0_LAST) {
                rP1(i) = rP1(i+1) = MAX_INIT_VAL;
            } else if constexpr (!query_all_valid) {
                int k_base = ((i/4) % 4) * 2;
                rP1(i)   = r_valid[k_base]     ? rP1(i)   : MAX_INIT_VAL;
                rP1(i+1) = r_valid[k_base + 1] ? rP1(i+1) : MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(rP1(i), rP1(i+1)));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
        r_cur_max[local_row_idx] = cur_max * scale_softmax_log2;
    }
    }
}

// Save rPb (64x64, bfloat16/half) to sP using the stmatrix instruction
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void save_rP1_to_sP(
    Tensor<Engine0, Layout0> &rPb,
    Tensor<Engine1, Layout1> &sP,
    int idx_in_warpgroup
) {
    typename T::TiledMma tiled_mma;
    const int cute_idx = T::kV41PairedPipeline ? idx_in_warpgroup : idx_in_warpgroup % T::kMmaThreads;
    auto r2s_copy = make_tiled_copy_A(typename T::SmemCopyAtomS{}, tiled_mma);
    ThrCopy thr_copy = r2s_copy.get_slice(cute_idx);
    Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
    Tensor thr_copy_sP = thr_copy.partition_D(sP);

    cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
}

// Save rPb (64x64, bfloat16/half) to sP using the stmatrix instruction
template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void save_rP0_to_sP(
    Tensor<Engine0, Layout0> &rPb,
    Tensor<Engine1, Layout1> &sP,
    int idx_in_warpgroup)
{
    typename T::TiledMma tiled_mma;
    const int cute_idx = T::kV41PairedPipeline ? idx_in_warpgroup : idx_in_warpgroup % T::kMmaThreads;
    auto r2s_copy = make_tiled_copy_C(typename T::SmemCopyAtomS{}, tiled_mma);
    ThrCopy thr_copy = r2s_copy.get_slice(cute_idx);
    Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
    Tensor thr_copy_sP = thr_copy.partition_D(sP);

    cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
}

// Retrieve rPb (64x64, bfloat16/half) from sP using the ldmatrix instruction
template <
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1>
__forceinline__ __device__ void retrieve_rP_from_sP(
    Tensor<Engine0, Layout0> &rPb,
    Tensor<Engine1, Layout1> const &sP,
    int idx_in_warpgroup)
{
    typename T::TiledMma tiled_mma;
    const int cute_idx = T::kV41PairedPipeline ? idx_in_warpgroup : idx_in_warpgroup % T::kMmaThreads;
    const int warp_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);
    const int cute_warp = (warp_idx % T::kAtomLayoutM) * 32;

    auto thr_mma = tiled_mma.get_thread_slice(cute_idx);
    // This helper receives Q tile 8 as a standalone view. It is unpaired,
    // so retain the original descriptor instead of treating its local tile 0
    // coordinate as the first half of a paired cube.
    using Q8CopyAtom = Copy_Atom<PPU_TSM_LD_SWZL<typename T::InputT,
        T::kBlockM, T::kBlockKSmem, false, false, 1>, typename T::InputT>;
    auto smem_tiled_copy_Q = make_tiled_copy_A(Q8CopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(cute_warp);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sP));
    // Tensor tSrQ  = thr_mma.partition_fragment_A(sP);
    // Tensor rQ8 = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(rPb));
    cute::copy(smem_tiled_copy_Q, tSsQ, rPb);
}


// Rescale rP0 and save the result to rPb
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
>
__forceinline__ __device__ void wg0_scale_rP0(
    Tensor<Engine0, Layout0> const &sScale1,	// (BLOCK_M)
    Tensor<Engine1, Layout1> const &rP0,		// ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rPb,		// ((2, 2, 8), 1, 1)
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale1(row_idx);
#if ACOMPUTE_VERSION == 10000
        if constexpr (!T::kV41PairedPipeline) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 4 : 0; i < size(rP0); i += 8) {
            rPb(i) = (typename T::InputT)(rP0(i)*scale_factor);
            rPb(i+1) = (typename T::InputT)(rP0(i+1)*scale_factor);
            rPb(i+2) = (typename T::InputT)(rP0(i+2)*scale_factor);
            rPb(i+3) = (typename T::InputT)(rP0(i+3)*scale_factor);
        }
        } else
#endif
        {
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            rPb(i) = (typename T::InputT)(rP0(i)*scale_factor);
            rPb(i+1) = (typename T::InputT)(rP0(i+1)*scale_factor);
        }
        }
    }
}


// Rescale rO0 according to sScale1
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void wg0_rescale_rO0(
    Tensor<Engine0, Layout0> &rO0,
    Tensor<Engine1, Layout1> &sScale1,
    float rL[2],
    int idx_in_warpgroup
) {
    if constexpr (T::kUsePv4x2) {
        pv_scale_one<T>(rO0, sScale1, idx_in_warpgroup);
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < 2; ++mi) {
            rL[mi] *= sScale1(get_AorC_row_idx<T::kAtomLayoutM>(mi, idx_in_warpgroup));
        }
        return;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale1(row_idx);
#if ACOMPUTE_VERSION == 10000
        if constexpr (!T::kV41PairedPipeline) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 4 : 0; i < size(rO0); i += 8) {
            rO0(i) *= scale_factor;
            rO0(i+1) *= scale_factor;
            rO0(i+2) *= scale_factor;
            rO0(i+3) *= scale_factor;
        }
        } else
#endif
        {
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
            rO0(i) = rO0(i)*scale_factor;
            rO0(i+1) = rO0(i+1)*scale_factor;
        }
        }
        rL[local_row_idx] *= scale_factor;
    }

}

// Rescale rO1 according to sScale0
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
>
__forceinline__ __device__ void wg1_scale0_rO1(
    Tensor<Engine0, Layout0> &rO1,
    Tensor<Engine1, Layout1> &sScale0,
    Tensor<Engine2, Layout2> &sScale1,
    int idx_in_warpgroup
) {
    if constexpr (T::kUsePv4x2) {
        auto rc = make_tensor(rO1.data(), pv_rowcol_layout(rO1.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < 4; ++mi) {
            int row = pv_row_idx<T>(mi, idx_in_warpgroup);
            float factor = sScale0(row) * sScale1(row);
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(rc); ++ni) rc(mi, ni) *= factor;
        }
        return;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale0(row_idx) * sScale1(row_idx);
#if ACOMPUTE_VERSION == 10000
        if constexpr (!T::kV41PairedPipeline) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 4 : 0; i < size(rO1); i += 8) {
            rO1(i) *= scale_factor;
            rO1(i+1) *= scale_factor;
            rO1(i+2) *= scale_factor;
            rO1(i+3) *= scale_factor;
        }
        } else
#endif
        {
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rO1); i += 4) {
            rO1(i) = (rO1(i)*scale_factor);
            rO1(i+1) = (rO1(i+1)*scale_factor);
        }
        }
    }
}

// Rescale rO0 according to local scale
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void wg0_scale0_rO0(
    Tensor<Engine0, Layout0> &rO0,
    Tensor<Engine1, Layout1> &sScale0,
    int idx_in_warpgroup
) {
    if constexpr (T::kUsePv4x2) {
        pv_scale_one<T>(rO0, sScale0, idx_in_warpgroup);
        return;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx<T::kAtomLayoutM>(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale0[row_idx];
#if ACOMPUTE_VERSION == 10000
        if constexpr (!T::kV41PairedPipeline) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 4 : 0; i < size(rO0); i += 8) {
            rO0(i) *= scale_factor;
            rO0(i+1) *= scale_factor;
            rO0(i+2) *= scale_factor;
            rO0(i+3) *= scale_factor;
        }
        } else
#endif
        {
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
            rO0(i) *= scale_factor;
            rO0(i+1) *= scale_factor;
        }
        }
    }
}

// The dense paired-cube PV load exchanges the two 64-column fragments.
// Restore their logical column positions without another memory round trip.
template<typename T, typename Copy, typename ER, typename LR, typename ES, typename LS>
__forceinline__ __device__ void store_pv_fragment(
    Copy copy_op, Tensor<ER, LR> const &src, Tensor<ES, LS> &dst) {
    constexpr bool reorder_columns = [] {
        if constexpr (T::kV41PairedPipeline) return true;
        else return T::CvtGemmSwzlLd;
    }();
    if constexpr (reorder_columns) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<2>(dst); ++j) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size<0>(dst); ++i) {
                const int real_i = T::kUsePv4x2
                    ? (i & 3) | ((j & 2) << 1) : (i & 3) | (j & 4);
                const int real_j = T::kUsePv4x2
                    ? (j & ~2) | ((i & 4) >> 1) : i + j - real_i;
                cute::copy(copy_op, src(real_i, _, real_j), dst(i, _, j));
            }
        }
    } else {
        cute::copy(copy_op, src, dst);
    }
}

// Store O / OAccum
template<
    typename T,
    bool IS_NO_SPLIT,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void store_o(
    Tensor<Engine0, Layout0> &rO,	// ((2, 2, 32), 1, 1)
    Tensor<Engine1, Layout1> &gOorAccum,	// (BLOCK_SIZE_M, HEAD_DIM_V)
    float rL[2],
    char* sO_addr,
    int num_valid_seq_q,
    int warpgroup_idx,
    int idx_in_warpgroup
) {
    using InputT = typename T::InputT;
    using ElementO = std::conditional_t<IS_NO_SPLIT, typename T::InputT, typename T::ElementAccum>;

    using SmemTiledCopyO = std::conditional_t<
        IS_NO_SPLIT,
        typename T::SmemCopyAtomO,
        typename T::SmemCopyAtomOaccum
    >;

    // (SMEM_M,SMEM_N) // Sw<3,3,3> o _0 o (_32,(_64,_8)):(_64,(_1,_2048))
    Tensor rOb = make_tensor_like<ElementO>(rO);

    if constexpr (T::kUsePv4x2) {
        auto src = make_tensor(rO.data(), pv_rowcol_layout(rO.layout()));
        auto dst = make_tensor(rOb.data(), pv_rowcol_layout(rOb.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < 4; ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(src); ++ni) {
                dst(mi, ni) = (InputT)(src(mi, ni) / rL[mi]);
            }
        }
    } else {
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < size(rO); ++idx) {
        if constexpr (T::kV41PairedPipeline) {
            rOb(idx) = (ElementO)(rO(idx) / rL[idx%4 >= 2]);
        } else {
#if ACOMPUTE_VERSION == 10000
            rOb(idx) = (ElementO)(rO(idx) / rL[(idx / 4) % 2]);
#else
            rOb(idx) = (InputT)(rO(idx) / rL[idx%4 >= 2]);
#endif
        }
    }
    }

    if constexpr (!IS_NO_SPLIT && T::kBlockM >= 128 && !T::kV41PairedPipeline) {
        // Two-pass float32 output for BlockM>=128 to fit PPU M890P SMEM limit.
        using SmemLayoutO_Half = typename T::SmemLayoutO_Half;

        Tensor sHalfBuf = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(sO_addr)),
            SmemLayoutO_Half{});  // (kBlockM, kHeadDimV/2) = 128x256 floats

        typename T::TiledMmaPV tiled_mma;
        const int cute_idx = idx_in_warpgroup % T::kMmaThreads;
        auto r2s_tiled_copy = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
        ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(cute_idx);
        Tensor r2s_thr_copy_rOb = r2s_thr_copy.retile_S(rOb);
        Tensor r2s_thr_copy_sHalfBuf = r2s_thr_copy.partition_D(sHalfBuf);

        // GMEM copy infrastructure (partitioned over 128x256)
        using GmemTiledCopyOHalf = typename T::GmemTiledCopyOaccum;
        GmemTiledCopyOHalf gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(threadIdx.x);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sHalfBuf);

        // Identity tensor for M-boundary check (reused across passes)
        Tensor cO = make_identity_tensor(make_shape(size<0>(sHalfBuf), size<1>(sHalfBuf)));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);

        CUTLASS_PRAGMA_UNROLL
        for (int pass = 0; pass < 2; ++pass) {
            // r2s: only the WG whose data matches this pass writes
            if (warpgroup_idx == pass) {
                if constexpr (T::CvtGemmSwzlLd) {
                    store_pv_fragment<T>(r2s_tiled_copy, r2s_thr_copy_rOb, r2s_thr_copy_sHalfBuf);
                } else {
                    cute::copy(r2s_tiled_copy, r2s_thr_copy_rOb, r2s_thr_copy_sHalfBuf);
                }
            }
            __syncthreads();

            // s2g: all threads copy 128x256 half to correct GMEM columns
            // gOorAccum has stride (kHeadDimV, 1); offset by pass*(kHeadDimV/2)
            Tensor gO_half = make_tensor(
                make_gmem_ptr(reinterpret_cast<ElementO *>(gOorAccum.data().get()) + pass * (T::kHeadDimV / 2)),
                Layout<Shape<Int<T::kBlockM>, Int<T::kHeadDimV / 2>>,
                       Stride<Int<T::kHeadDimV>, _1>>{});
            Tensor tOgO = gmem_thr_copy_O.partition_D(gO_half);
            Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
            Tensor tOrO_local = make_tensor<ElementO>(shape(tOgO));

            // smem -> reg
            cute::copy(gmem_tiled_copy_O, tOsO, tOrO_local);
            // reg -> gmem (with M-boundary mask)
            copy<false, true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_O, tOrO_local, tOgO, tOcO, tOpO, num_valid_seq_q);

            // Sync before next pass's r2s overwrites the buffer
            if (pass == 0) __syncthreads();
        }
    } else {
        // ==================================================================
        // Original single-pass path (BF16 output or BlockM<128)
        // ==================================================================
        Tensor sOutputBuf = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(sO_addr)),
            typename T::SmemLayoutO{});

        Tensor sMyOutputBuf = local_tile(sOutputBuf, Shape<Int<T::kBlockM>, _256>{}, make_coord(_0{}, warpgroup_idx));

        typename T::TiledMmaPV tiled_mma;
        const int cute_idx = T::kV41PairedPipeline
            ? idx_in_warpgroup : idx_in_warpgroup % T::kMmaThreads;
        auto r2s_tiled_copy = make_tiled_copy_C(
            SmemTiledCopyO{}, tiled_mma);

        ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(cute_idx);
        Tensor r2s_thr_copy_rOb = r2s_thr_copy.retile_S(rOb);
        Tensor r2s_thr_copy_sMyOutputBuf = r2s_thr_copy.partition_D(sMyOutputBuf);
        store_pv_fragment<T>(r2s_tiled_copy, r2s_thr_copy_rOb, r2s_thr_copy_sMyOutputBuf);

        __syncthreads();

        // tsm->global

        using GmemTiledCopyO = std::conditional_t<
            IS_NO_SPLIT,
            typename T::GmemTiledCopyO,
            typename T::GmemTiledCopyOaccum>;
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(threadIdx.x);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sOutputBuf); // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgO = gmem_thr_copy_O.partition_D(gOorAccum);

        Tensor tOrO = make_tensor<ElementO>(shape(tOgO));

        /// tsm -> reg
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sOutputBuf), size<1>(sOutputBuf))); // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO); // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));

        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        copy<false, true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, num_valid_seq_q);
    }
}

// Internal address representation for aligned, main-only cache spans below 64GiB.
template<typename T>
struct CompactKAddrTraits : T { using Bf16KAddr = uint32_t; };
template<typename T, typename = void>
struct WgKAddrType { using type = typename T::InputT*; };
template<typename T>
struct WgKAddrType<T, std::void_t<typename T::Bf16KAddr>> {
    using type = typename T::Bf16KAddr;
};
template<typename T>
using WgKAddr = typename WgKAddrType<T>::type;

// Phase 1: Compute token_ptr from pre_token_idx + write smem_valid_indices.
// Does NOT issue cp.async, does NOT write to sK buffer.
// Can be placed anywhere (fills SIMT gaps during TC execution).
// Compile-time version: for mainloop (USE_EXTRA as template param)
template<int S, bool USE_EXTRA, typename T, typename TensorVI>
__forceinline__ __device__ WgKAddr<T> compute_K_addr_bf16(
    const Flash_fwd_mla_params &params,
    int batch_idx,
    int block_idx_kv,
    int seqlen_k,
    int tidx,
    int ori_block_max,
    TensorVI &smem_valid_indices,
    int vi_buf,
    int *pre_token_idx,
    int extra_seqlen_k,
    bool &is_valid
) {
    using Base   = typename T::Base;
    using InputT = typename T::InputT;
    constexpr int kBlockN        = Base::kBlockN;
    constexpr int kBytesPerToken = T::kBytesPerToken;

    // Use if constexpr to force compile-time branch elimination (PPU compiler doesn't optimize ternary with template params)
    int effective_block, page_block_size, valid_len;
    InputT* k_base_ptr;
    size_t k_stride_elems;
    if constexpr (USE_EXTRA) {
        effective_block = block_idx_kv - ori_block_max;
        page_block_size = params.extra_page_block_size;
        valid_len = extra_seqlen_k;
        k_base_ptr = reinterpret_cast<InputT*>(params.extra_k_ptr);
        k_stride_elems = static_cast<size_t>(params.extra_k_batch_stride);
    } else {
        effective_block = block_idx_kv;
        page_block_size = params.page_block_size;
        valid_len = seqlen_k;
        k_base_ptr = reinterpret_cast<InputT*>(params.k_ptr);
        k_stride_elems = static_cast<size_t>(params.k_batch_stride);
    }
    int tile_valid = valid_len - effective_block * kBlockN;

    int token_id = tidx / 8;
    int t_idx = (token_id < tile_valid) ? *pre_token_idx : -1;
    is_valid = (t_idx >= 0);
    if constexpr (S == 0) {
        if (tidx % 8 == 0) {
            smem_valid_indices(vi_buf, token_id) = is_valid;
        }
    }

    // k_stride_elems set above via if constexpr

    if constexpr (std::is_same_v<WgKAddr<T>, uint32_t>) {
        // Host guard proves all defined cache addresses fit uint32 units of 16B.
        // Compact kernels are main-only; default kernels keep full pointers.
        static_assert(sizeof(InputT) == 2 && kBytesPerToken % 16 == 0);
        const uint32_t lane = uint32_t(tidx % 8);
        if (!is_valid) return lane;
        const uint32_t page = uint32_t(t_idx >> __builtin_ctz(page_block_size));
        const uint32_t row = uint32_t(t_idx & ((page_block_size & (-page_block_size)) - 1));
        return page * uint32_t(k_stride_elems / 8) +
               row * uint32_t(kBytesPerToken / 16) + lane;
    } else {
        InputT* token_ptr = k_base_ptr + (tidx % 8) * 8;
        if (is_valid) {
            int page_idx = t_idx >> __builtin_ctz(page_block_size);
            int off_in_page = t_idx & ((page_block_size & (-page_block_size)) - 1);
            token_ptr = k_base_ptr + page_idx * k_stride_elems +
                off_in_page * (kBytesPerToken / sizeof(InputT)) + (tidx % 8) * 8;
        }
        return token_ptr;
    }
}

// Runtime version: for prolog (use_extra as runtime param)
template<int S, typename T, typename TensorVI>
__forceinline__ __device__ WgKAddr<T> compute_K_addr_bf16_dynamic(
    const Flash_fwd_mla_params &params,
    int batch_idx,
    int block_idx_kv,
    int seqlen_k,
    int tidx,
    int ori_block_max,
    TensorVI &smem_valid_indices,
    int vi_buf,
    int *pre_token_idx,
    int extra_seqlen_k,
    bool &is_valid,
    bool use_extra
) {
    if (use_extra) {
        return compute_K_addr_bf16<S, true, T>(params, batch_idx, block_idx_kv, seqlen_k, tidx, ori_block_max,
            smem_valid_indices, vi_buf, pre_token_idx, extra_seqlen_k, is_valid);
    } else {
        return compute_K_addr_bf16<S, false, T>(params, batch_idx, block_idx_kv, seqlen_k, tidx, ori_block_max,
            smem_valid_indices, vi_buf, pre_token_idx, extra_seqlen_k, is_valid);
    }
}

template<bool UseExtra, typename T>
__forceinline__ __device__ const int* wg_query_indices_bf16(
    const Flash_fwd_mla_params &params)
{
    // Dispatch requires complete head tiles; several M blocks may share a query.
    const int query_idx = static_cast<int>(blockIdx.x) / (params.ngroups / T::kBlockM);
    const int* indices = UseExtra ? params.extra_indices_ptr : params.indices_ptr;
    const auto row_stride = UseExtra
        ? params.extra_indices_row_stride : params.indices_row_stride;
    return indices + static_cast<int64_t>(query_idx) * row_stride;
}

template<bool UseExtra, typename T>
__forceinline__ __device__ const int* wg_query_indices(
    const Flash_fwd_mla_params &params)
{
    // Sq=1 can retain a partial final head tile; every tile uses query zero.
    // MTP dispatch requires ngroups % kBlockM == 0, so its positive tile
    // divisor and grid coordinate both fit int32. Only pointer offsets need int64.
    int query_idx = 0;
    if (params.seqlen_q > params.ngroups) {
        query_idx = static_cast<int>(blockIdx.x) / (params.ngroups / T::kBlockM);
    }
    const int* indices = UseExtra ? params.extra_indices_ptr : params.indices_ptr;
    const auto row_stride = UseExtra
        ? params.extra_indices_row_stride : params.indices_row_stride;
    return indices + static_cast<int64_t>(query_idx) * row_stride;
}

// Phase 2: Issue cp.async using precomputed token_ptr + prefetch next token index.
// LOAD_USE_EXTRA removed: k_base_ptr is dummy (overwritten by token_ptr), only PREFETCH_USE_EXTRA matters
template<int S, int E, bool PREFETCH_USE_EXTRA, typename T, bool DO_PREFETCH = true,
         bool DO_COPY = true, bool DO_COMMIT = true, typename TensorSK>
__forceinline__ __device__ void issue_K_load_bf16(
    const Flash_fwd_mla_params &params,
    TensorSK &sK_buf,
    int batch_idx,
    int block_idx_kv,
    int seqlen_k,
    __mbarrier_t *barriers_K,
    int tidx,
    int ori_block_max,
    WgKAddr<T> token_addr,
    bool is_valid,
    int *pre_token_idx,
    int next_block_idx,
    int end_block_idx,
    int extra_seqlen_k
) {
    using Base   = typename T::Base;
    using InputT = typename T::InputT;
    constexpr int kBlockN          = Base::kBlockN;
    constexpr int kThreadsPerWg    = Base::NUM_THREADS / 2;

    if constexpr (DO_COPY) {
        using KVCacheGmem = flash::KVCacheGmemBf16<InputT, kBlockN, kThreadsPerWg, T::kHeadDim>;
        using GmemTiledCopyKNoAiu = typename KVCacheGmem::GmemTiledCopy;
        GmemTiledCopyKNoAiu gmem_tiled_copy_K;

        // Dummy k_base_ptr — actual address comes from token_ptr (set below)
        InputT* k_base_ptr = reinterpret_cast<InputT*>(params.k_ptr);
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);
        Tensor tKsK = gmem_thr_copy_K.partition_D(sK_buf);
        Tensor gK_tok = make_tensor(make_gmem_ptr(k_base_ptr),
            Shape<Int<kBlockN>, Int<T::kHeadDim>>{},
            make_stride(params.k_row_stride, _1{}));
        Tensor tKgK_tok = gmem_thr_copy_K.partition_S(gK_tok);
        if constexpr (std::is_same_v<WgKAddr<T>, uint32_t>) {
            tKgK_tok.data() = k_base_ptr + uint64_t(token_addr) * 8;
        } else {
            tKgK_tok.data() = token_addr;
        }

        gmem_tiled_copy_K.pred = is_valid;
        if constexpr (T::CvtGemmSwzlLd) {
            flash::launch_kv_pair_copy<S, E, T>(gmem_tiled_copy_K, tKgK_tok, sK_buf, tidx);
            if constexpr (DO_COMMIT) {
                cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
            }
        } else {
            static_assert(DO_COMMIT, "Raw-copy specialization requires CvtGemmSwzlLd");
            flash::launch_kv_tiles_wg<S, E>(gmem_tiled_copy_K, tKgK_tok, tKsK, barriers_K);
        }
    } else if constexpr (DO_COMMIT) {
        cutlass::arch::cpasync_barrier_arrive_noinc(barriers_K);
    }

    if constexpr (DO_PREFETCH) {
        if (next_block_idx < end_block_idx) {
            int next_eff_block;
            const int *next_idx_base;
            if constexpr (PREFETCH_USE_EXTRA) {
                next_eff_block = next_block_idx - ori_block_max;
                next_idx_base = wg_query_indices_bf16<true, T>(params) + static_cast<int64_t>(batch_idx) * params.extra_indices_batch_stride + next_eff_block * kBlockN;
            } else {
                next_eff_block = next_block_idx;
                next_idx_base = wg_query_indices_bf16<false, T>(params) + static_cast<int64_t>(batch_idx) * params.indices_batch_stride + next_eff_block * kBlockN;
            }
            int token_id = tidx / 8;
            *pre_token_idx = __ldg(next_idx_base + token_id);
        }
    }
}

template<int S, int E, typename T, bool DO_PREFETCH = true, typename TensorSK, typename TensorVI>
__forceinline__ __device__ void load_K_tiles_bf16(
    const Flash_fwd_mla_params &params,
    TensorSK &sK_buf,
    int batch_idx,
    int block_idx_kv,
    int seqlen_k,
    __mbarrier_t *barriers_K,
    int tidx,
    int ori_block_max,
    TensorVI &smem_valid_indices,
    int vi_buf,
    int *pre_token_idx,
    int next_block_idx,
    int end_block_idx,
    int extra_seqlen_k
) {
    // Dynamic wrapper for prolog: compute addr + dispatch issue_K_load via template
    bool use_extra = (ori_block_max >= 0) && (block_idx_kv >= ori_block_max);
    bool prefetch_use_extra = (ori_block_max >= 0) && (next_block_idx >= ori_block_max);
    bool is_valid;
    WgKAddr<T> token_ptr = compute_K_addr_bf16_dynamic<S, T>(
        params, batch_idx, block_idx_kv, seqlen_k, tidx, ori_block_max,
        smem_valid_indices, vi_buf, pre_token_idx, extra_seqlen_k, is_valid, use_extra);
    
    if (prefetch_use_extra) {
        issue_K_load_bf16<S, E, true, T, DO_PREFETCH>(
            params, sK_buf, batch_idx, block_idx_kv, seqlen_k, barriers_K,
            tidx, ori_block_max, token_ptr, is_valid,
            pre_token_idx, next_block_idx, end_block_idx, extra_seqlen_k);
    } else {
        issue_K_load_bf16<S, E, false, T, DO_PREFETCH>(
            params, sK_buf, batch_idx, block_idx_kv, seqlen_k, barriers_K,
            tidx, ori_block_max, token_ptr, is_valid,
            pre_token_idx, next_block_idx, end_block_idx, extra_seqlen_k);
    }
}

// ============================================================================
// load_and_dequant_sparse_K_staged: staged K load with per-stage dim range.
//
// Splits the FP8 dequant into two stages:
//   Stage S=0, E=4: dims [0,   256)  -> nope only, barrier 0
//   Stage S=4, E=9: dims [256, 576)  -> nope [256,512) + rope [0,64), barrier 1
//
//   * Processes only [kDimStart, kDimEnd) instead of full [0, 576)
//   * smem_valid_indices written only when S == 0 (using vi_buf_idx)
//   * Arrives ONLY barriers_K[S/4] instead of both barriers
//   * No sV_direct parameter (V read handled elsewhere)
// ============================================================================
// Start the dependent indices -> KV-address chain before the softmax/PV work
// in each WI iteration. The staged loader still applies the per-request valid
// length, so it is safe to prefetch indices beyond a variable topk_len.
template<typename T>
__forceinline__ __device__ int prefetch_v41_token_index(
    const Flash_fwd_mla_params &params, int batch_idx, int block_idx_kv,
    int ori_block_max, int tidx) {
    using Base = typename T::Base;
    constexpr int kBlockN = Base::kBlockN;
    constexpr int kLanesPerToken = (Base::NUM_THREADS / 2) / kBlockN;
    static_assert(kLanesPerToken == 8);
    const bool use_extra = (ori_block_max >= 0) && (block_idx_kv >= ori_block_max);
    const int effective_block = use_extra ? block_idx_kv - ori_block_max : block_idx_kv;
    const int max_topk = use_extra ? params.extra_topk : params.topk;
    const int token_id = tidx / kLanesPerToken;
    // All eight lanes consume one token. Issue its dependent index load once
    // and distribute it before the conversion work instead of loading it eight times.
    int token_idx = -1;
    if (effective_block >= 0 && effective_block * kBlockN + token_id < max_topk
            && tidx % kLanesPerToken == 0) {
        const int *indices_base = use_extra
            ? wg_query_indices<true, T>(params) + static_cast<int64_t>(batch_idx) * params.extra_indices_batch_stride
                + effective_block * kBlockN
            : wg_query_indices<false, T>(params) + static_cast<int64_t>(batch_idx) * params.indices_batch_stride
                + effective_block * kBlockN;
        token_idx = __ldg(indices_base + token_id);
    }
    return __shfl_sync(0xffffffff, token_idx, 0, kLanesPerToken);
}

template<int S, int E, typename T, bool PreloadedIndex = false, typename TensorSK, typename TensorVI>
__forceinline__ __device__ void load_and_dequant_sparse_K_staged(
    const Flash_fwd_mla_params &params,
    TensorSK &sK,
    int batch_idx,
    int block_idx_kv,
    int seqlen_k,
    int buf_idx,
    __mbarrier_t *barriers_K,
    int tidx,
    int ori_block_max,
    TensorVI &smem_valid_indices,
    int vi_buf_idx,
    int preloaded_index = -1
) {
    using Base   = typename T::Base;
    using InputT = typename T::InputT;       // cutlass::bfloat16_t
    constexpr int kBlockN          = Base::kBlockN;                  // 32
    constexpr int kFp8NopeBytes    = T::kFp8NopeBytesPerToken;       // 512 (V3.2) or 448 (MODEL1)
    constexpr int kFp8ScaleBytes   = T::kFp8ScaleBytesPerToken;      // 16  (V3.2) or 8   (MODEL1)
    constexpr int kBytesPerToken   = T::kBytesPerToken;              // 656 (V3.2) or 576 (MODEL1)
    constexpr int kRopeOffset      = T::kRopeOffsetBytes;            // 528 (V3.2) or 448 (MODEL1)
    constexpr int kTileSize        = T::kScaleTileSize;              // 128 (V3.2) or 64  (MODEL1)
    constexpr int kRopeElems       = T::kRopeElems;                  // 64
    constexpr int kThreadsPerWg    = Base::NUM_THREADS / 2;          // 256
    (void)kRopeElems;  // unused in staged path; kRopeRange replaces it

    // ---- Stage dim range constants ----
    constexpr int kDimStart     = S * T::kBlockKSmem;      // 0 or 256
    constexpr int kDimEnd       = E * T::kBlockKSmem;      // 256 or 576
    constexpr int kNopeDimStart = kDimStart < kFp8NopeBytes ? kDimStart : kFp8NopeBytes;
    constexpr int kNopeDimEnd   = kDimEnd   < kFp8NopeBytes ? kDimEnd   : kFp8NopeBytes;
    constexpr int kRopeDimStart = kDimStart > kFp8NopeBytes ? (kDimStart - kFp8NopeBytes) : 0;
    constexpr int kRopeDimEnd   = kDimEnd   > kFp8NopeBytes ? (kDimEnd   - kFp8NopeBytes) : 0;
    constexpr int kNopeRange    = kNopeDimEnd - kNopeDimStart;
    constexpr int kRopeRange    = kRopeDimEnd - kRopeDimStart;

    // ---- Determine if this block is in extra cache ----
    const bool use_extra = (ori_block_max >= 0) && (block_idx_kv >= ori_block_max);
    const int effective_block = use_extra ? (block_idx_kv - ori_block_max) : block_idx_kv;

    const int *indices_base = use_extra
        ? (wg_query_indices<true, T>(params)
           + static_cast<int64_t>(batch_idx) * params.extra_indices_batch_stride
           + effective_block * kBlockN)
        : (wg_query_indices<false, T>(params)
           + static_cast<int64_t>(batch_idx) * params.indices_batch_stride
           + effective_block * kBlockN);
    const int page_block_size = use_extra ? params.extra_page_block_size : params.page_block_size;
    constexpr size_t kStrideElemSize = T::IsFP8 ? 1 : sizeof(InputT);
    const size_t k_batch_stride = (use_extra
        ? static_cast<size_t>(params.extra_k_batch_stride)
        : static_cast<size_t>(params.k_batch_stride)) * kStrideElemSize;
    const uint8_t *k_base = reinterpret_cast<const uint8_t*>(
        use_extra ? params.extra_k_ptr : params.k_ptr);

    // ---- Tile-local valid-token count ----
    int valid_len;
    if (use_extra) {
        valid_len = params.extra_topk_len_ptr
            ? __ldg(params.extra_topk_len_ptr + batch_idx)
            : params.extra_topk;
    } else {
        valid_len = params.topk_len_ptr
            ? __ldg(params.topk_len_ptr + batch_idx)
            : params.topk;
    }
    if (!use_extra && seqlen_k >= 0 && seqlen_k < valid_len) {
        valid_len = seqlen_k;
    }
    int tile_valid = valid_len - effective_block * kBlockN;
    if (tile_valid < 0) tile_valid = 0;
    if (tile_valid > kBlockN) tile_valid = kBlockN;

    // ---- 0. Valid indices (only for stage 0) ----
    if constexpr (S == 0 && !T::kV41FP8) {
        if (tidx < kBlockN) {
            int t_idx = (tidx < tile_valid) ? __ldg(indices_base + tidx) : -1;
            smem_valid_indices(vi_buf_idx, tidx) = (t_idx >= 0);
        }
    }

    // V4.1 keeps all 512 dimensions in the quantized data segment. Eight lanes
    // serve each token; each lane handles 32 values per stage. FP8 has one E8M0
    // scale per lane, while FP4 has two E4M3 scales (one per 16 values).
    if constexpr (T::kV41FP8) {
        static_assert(kNopeRange == 256 && kRopeRange == 0);
        constexpr int kLanesPerToken = kThreadsPerWg / kBlockN;
        static_assert(kLanesPerToken == 8);
        const int token_id = tidx / kLanesPerToken;
        const int lane = tidx % kLanesPerToken;
        const int dim_begin = kDimStart + lane * 32;
        const bool token_valid = token_id < tile_valid;
        int page_idx = 0;
        int off_in_page = 0;
        int t_idx = -1;
        if constexpr (PreloadedIndex) {
            t_idx = token_valid ? preloaded_index : -1;
        } else {
            if (token_valid && lane == 0) {
                t_idx = __ldg(indices_base + token_id);
            }
            t_idx = __shfl_sync(0xffffffff, t_idx, 0, kLanesPerToken);
        }
        if constexpr (S == 0) {
            // Publish validity from the same index used for KV addressing.
            if (lane == 0) smem_valid_indices(vi_buf_idx, token_id) = (t_idx >= 0);
        }
        if (token_valid) {
            if (t_idx >= 0) {
                // The V4.1 WI dispatch admits only power-of-two page sizes.
                const int page_shift = __ffs(page_block_size) - 1;
                page_idx = t_idx >> page_shift;
                off_in_page = t_idx & (page_block_size - 1);
            }
        }
        const bool valid = t_idx >= 0;
        if constexpr (T::kExtraFP4) {
            if (use_extra) {
                const uint8_t *block = k_base + page_idx * k_batch_stride;
                const uint8_t *token = block + int64_t(off_in_page) * 288;
                const uint8_t *data = token + dim_begin / 2;
                const uint8_t *scales = token + 256;
                float scales_fp4[2] = {0.0f, 0.0f};
                if (valid) {
                    #pragma unroll
                    for (int group = 0; group < 2; ++group) {
                        const uint8_t raw = __ldg(scales + dim_begin / 16 + group);
                        const __hg_fp8_e4m3 e4m3 = *reinterpret_cast<const __hg_fp8_e4m3*>(&raw);
                        scales_fp4[group] = float(e4m3);
                    }
                }
                // Each lane owns 16 aligned packed bytes: one 128-bit load.
                uint4 packed4{};
                if (valid) {
                    packed4 = __ldg(reinterpret_cast<const uint4*>(data));
                }
                const __ppu_bfloat162 scales_bf16x2[2] = {
                    __float2bfloat162_rn(scales_fp4[0]),
                    __float2bfloat162_rn(scales_fp4[1])
                };
                const uint32_t words[4] = {packed4.x, packed4.y, packed4.z, packed4.w};
                #pragma unroll
                for (int chunk = 0; chunk < 4; ++chunk) {
                    const uint4 decoded = flash::fp4_word_to_bf16x2_bits(words[chunk]);
                    const uint32_t pairs[4] = {decoded.x, decoded.y, decoded.z, decoded.w};
                    uint32_t out_bits[4]{};
                    #pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const uint32_t values = pairs[pair];
                        const __ppu_bfloat162 scaled = __hmul2(
                            *reinterpret_cast<const __ppu_bfloat162*>(&values),
                            scales_bf16x2[chunk / 2]);
                        out_bits[pair] = *reinterpret_cast<const uint32_t*>(&scaled);
                    }
                    // Swizzle<3,3,3> preserves the eight contiguous BF16 values
                    // beginning at a multiple of eight: one aligned 16-byte store.
                    *reinterpret_cast<uint4*>(&sK(token_id, dim_begin + chunk * 8, buf_idx)) =
                        uint4{out_bits[0], out_bits[1], out_bits[2], out_bits[3]};
                }
            } else {
                const uint8_t *block = k_base + page_idx * k_batch_stride;
                const uint8_t *token = block + int64_t(off_in_page) * 528;
                const uint8_t *data = token + dim_begin;
                const uint8_t *scales = token + 512;
                float scale = 0.0f;
                if (valid) {
                    const uint8_t raw = __ldg(scales + dim_begin / 32);
                    const fp8_e8m0 e8m0 = *reinterpret_cast<const fp8_e8m0*>(&raw);
                    scale = float(e8m0);
                }
                // Each lane owns 32 aligned FP8 bytes: two 128-bit loads.
                uint4 packed_lo{}, packed_hi{};
                if (valid) {
                    packed_lo = __ldg(reinterpret_cast<const uint4*>(data));
                    packed_hi = __ldg(reinterpret_cast<const uint4*>(data + 16));
                }
                const uint32_t words[8] = {
                    packed_lo.x, packed_lo.y, packed_lo.z, packed_lo.w,
                    packed_hi.x, packed_hi.y, packed_hi.z, packed_hi.w
                };
                #pragma unroll
                for (int vec = 0; vec < 4; ++vec) {
                    const uint4 out_bits = flashmla::dsa::v41::fp8_words_to_scaled_bf16x8_bits(
                        words[vec * 2], words[vec * 2 + 1], scale);
                    *reinterpret_cast<uint4*>(&sK(token_id, dim_begin + vec * 8, buf_idx)) = out_bits;
                }
            }
        } else {
            const uint8_t *block = k_base + page_idx * k_batch_stride;
            const uint8_t *token = block + int64_t(off_in_page) * 528;
            const uint8_t *data = token + dim_begin;
            const uint8_t *scales = token + 512;
            float scale = 0.0f;
            if (valid) {
                const uint8_t raw = __ldg(scales + dim_begin / 32);
                const fp8_e8m0 e8m0 = *reinterpret_cast<const fp8_e8m0*>(&raw);
                scale = float(e8m0);
            }
            uint4 packed_lo{}, packed_hi{};
            if (valid) {
                packed_lo = __ldg(reinterpret_cast<const uint4*>(data));
                packed_hi = __ldg(reinterpret_cast<const uint4*>(data + 16));
            }
            const uint32_t words[8] = {
                packed_lo.x, packed_lo.y, packed_lo.z, packed_lo.w,
                packed_hi.x, packed_hi.y, packed_hi.z, packed_hi.w
            };
            #pragma unroll
            for (int vec = 0; vec < 4; ++vec) {
                const uint4 out_bits = flashmla::dsa::v41::fp8_words_to_scaled_bf16x8_bits(
                    words[vec * 2], words[vec * 2 + 1], scale);
                *reinterpret_cast<uint4*>(&sK(token_id, dim_begin + vec * 8, buf_idx)) = out_bits;
            }
        }
    // ---- 1. Legacy FP8 dequant: [kNopeDimStart, kNopeDimEnd) ----
    } else if constexpr (kNopeRange > 0) {
        constexpr int kTotalNopeElems     = kBlockN * kNopeRange;
        constexpr int kNopeElemsPerThread = kTotalNopeElems / kThreadsPerWg;
        static_assert(kTotalNopeElems % kThreadsPerWg == 0,
                      "staged nope: total elems must be divisible by warpgroup threads");
        #pragma unroll
        for (int i = 0; i < kNopeElemsPerThread; ++i) {
            int elem_id       = tidx + i * kThreadsPerWg;
            int token_id      = elem_id / kNopeRange;
            int elem_in_range = elem_id - token_id * kNopeRange;
            int elem_in_tok   = elem_in_range + kNopeDimStart;

            InputT bf16_val = InputT(0.0f);
            if (token_id < tile_valid) {
                int t_idx = __ldg(indices_base + token_id);
                if (t_idx >= 0) {
                    int page_idx    = t_idx / page_block_size;
                    int off_in_page = t_idx - page_idx * page_block_size;
                    const uint8_t *gmem_tok = k_base
                        + page_idx * k_batch_stride
                        + off_in_page * kBytesPerToken;

                    // FP8 nope value
                    uint8_t fp8_raw = __ldg(gmem_tok + elem_in_tok);
                    __hg_fp8_e4m3 fp8_val = *reinterpret_cast<const __hg_fp8_e4m3*>(&fp8_raw);

                    float scale_val;
                    if constexpr (T::kModel1Layout) {
                        // MODEL1: e8m0 scales in block-level tail segment.
                        int tile_idx = elem_in_tok / kTileSize;
                        const uint8_t *scale_ptr = k_base
                            + page_idx * k_batch_stride
                            + page_block_size * kBytesPerToken
                            + off_in_page * kFp8ScaleBytes
                            + tile_idx;
                        uint8_t scale_raw = __ldg(scale_ptr);
                        fp8_e8m0 scale_e8m0 = *reinterpret_cast<const fp8_e8m0*>(&scale_raw);
                        scale_val = float(scale_e8m0);
                    } else {
                        // V3.2: per-token FP32 scales (one per 128 nope dims).
                        int tile_idx = elem_in_tok / kTileSize;
                        scale_val = *reinterpret_cast<const float*>(
                            gmem_tok + kFp8NopeBytes + tile_idx * sizeof(float));
                    }

                    float result = float(fp8_val) * scale_val;
                    bf16_val = InputT(result);
                }
            }
            sK(token_id, elem_in_tok, buf_idx) = bf16_val;
        }
    }

    // ---- 2. Rope copy: [kRopeDimStart, kRopeDimEnd) ----
    if constexpr (kRopeRange > 0 && T::kHasRope) {
        constexpr int kTotalRopeElems      = kBlockN * kRopeRange;
        constexpr int kRopeElemsPerThread = kTotalRopeElems / kThreadsPerWg;
        static_assert(kTotalRopeElems % kThreadsPerWg == 0,
                      "staged rope: total elems must be divisible by warpgroup threads");
        #pragma unroll
        for (int j = 0; j < kRopeElemsPerThread; ++j) {
            int elem_id  = tidx + j * kThreadsPerWg;
            int token_id = elem_id / kRopeRange;
            int rope_idx = elem_id - token_id * kRopeRange;

            InputT rope_val = InputT(0.0f);
            if (token_id < tile_valid) {
                int t_idx = __ldg(indices_base + token_id);
                if (t_idx >= 0) {
                    int page_idx    = t_idx / page_block_size;
                    int off_in_page = t_idx - page_idx * page_block_size;
                    const uint8_t *gmem_tok = k_base
                        + page_idx * k_batch_stride
                        + off_in_page * kBytesPerToken;

                    // BF16 rope value
                    rope_val = *reinterpret_cast<const InputT*>(
                        gmem_tok + kRopeOffset + rope_idx * sizeof(InputT));
                }
            }
            sK(token_id, kFp8NopeBytes + rope_idx, buf_idx) = rope_val;
        }
    }

    // ---- 3. Warpgroup-level sync ----
    {
        int _bar_id = 6 + (int)(threadIdx.x >> 8);
        NamedBarrier::sync(
            kThreadsPerWg, static_cast<cutlass::arch::ReservedNamedBarriers>(_bar_id));
    }
    __threadfence_block();

    // ---- 4. Arrive only one barrier (key difference from full function) ----
    constexpr int kBarrierIdx = S / 4;  // 0 or 1
    cutlass::arch::cpasync_barrier_arrive_noinc(&barriers_K[kBarrierIdx]);
}

template <
    typename T,
    typename Tensor0>
__forceinline__ __device__ void launch_q_copy(
    const Flash_fwd_mla_params &params,
    int batch_idx,
    int m_block_idx,
    int k_head_idx,
    Tensor0 &sQ,
    const int tidx,
    const int warp_idx,
    __mbarrier_t* barrier_Q
) {
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<typename T::InputT*>(params.q_ptr)
                                            + batch_idx * params.q_batch_stride),
                            make_shape(params.seqlen_q, params.h_h_k_ratio, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(make_mix_tensor_like(mQ(_, k_head_idx, _)), Shape<Int<T::kBlockM>, Int<T::kHeadDim>>{},
                            make_coord(m_block_idx, 0));  // (kBlockM, kHeadDim)
    typename T::GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);

#if ACOMPUTE_VERSION == 10000
    int aiu_offset_q = 0;
    gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, params.seqlen_q, params.q_row_stride, T::kBlockM, T::kBlockKSmem, aiu_offset_q};
    // const int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
#else
    gmem_tiled_copy_Q.desc_.init(nullptr, params.seqlen_q, params.d, params.q_row_stride);
#endif
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    if constexpr (T::CvtGemmSwzlLd) {
        if (warp_idx < T::kBlockM / T::kBlockMPerLoad) {
            CUTLASS_PRAGMA_UNROLL
            for (int tile = 0; tile < T::kHeadDim / T::kBlockKSmem; ++tile) {
                auto dst = tQsQ(_, warp_idx, tile);
                if (tile < T::TileNoCvt) {
                    const int offset = (tile & ~1) * T::kBlockM * T::kBlockKSmem
                        + ((warp_idx << 1) + (tile & 1)) * T::kBlockMPerLoad * T::kBlockKSmem;
                    dst.data() = tQsQ.data() + offset;
                }
                cute::copy(gmem_tiled_copy_Q, tQgQ(_, warp_idx, tile), dst);
            }
            cutlass::arch::cpasync_barrier_arrive_noinc(barrier_Q);
        }
    } else if (warp_idx == 0) {
        cute::copy(gmem_tiled_copy_Q, tQgQ, tQsQ);
        cutlass::arch::cpasync_barrier_arrive_noinc(barrier_Q);
    }
}

template <
    typename T,
    bool IS_R,
    typename Engine0, typename Layout0>
__forceinline__ __device__ auto get_half_V(
    Tensor<Engine0, Layout0> &sV_direct)
{
    if constexpr (T::kV41PairedPipeline) {
        Tensor sV = make_tensor(sV_direct.data(), typename T::SmemLayoutV{});
        return flat_divide(sV, Shape<Int<T::kHeadDimV / 2>, Int<T::kBlockN>>{})(_, _, Int<(int)IS_R>{}, _0{});
    } else {
    // Use SmemLayoutVtDirect (swizzled composition) to read V transposed
    // from the independent V buffer. This produces a ComposedLayout that
    // make_mix_tensor_like can handle (rank 2, rank0=1, rank1=1).
    Tensor sVt = make_tensor(sV_direct.data(), (typename T::SmemLayoutVtDirect){});
    return flat_divide(sVt, Shape<Int<T::kHeadDimV / 2>, Int<T::kBlockN>>{})(_, _, Int<(int)IS_R>{}, _0{});
    }
}

template <
    typename T,
    bool IS_R,
    typename Engine0, typename Layout0>
__forceinline__ __device__ auto get_half_V2(
    int block_idx,
    Tensor<Engine0, Layout0> sV_direct)
{
    // Same as get_half_V, for alternate calling convention.
    Tensor sVt = make_tensor(sV_direct.data(), (typename T::SmemLayoutVtDirect){});
    return flat_divide(sVt, Shape<Int<T::kHeadDimV / 2>, Int<T::kBlockN>>{})(_, _, Int<(int)IS_R>{}, _0{});
}

template <
    typename T>
__forceinline__ __device__ long get_block_index(
    int block_idx,
    const Flash_fwd_mla_params &params,
    int *block_table_ptr)
{
    // const int block_table_idx = block_idx * T::Page_In_BlockN;
    const int block_table_idx = block_idx * 1 / 2;
    const int block_table_offset = block_idx * T::kBlockN - block_table_idx * T::PAGE_BLOCK_SIZE;
    return long(__ldg(block_table_ptr + block_table_idx) * params.k_batch_stride + block_table_offset * params.k_row_stride);
}

template <
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool NEXT_EXTRA,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename Engine6, typename Layout6,
    typename Engine7, typename Layout7,
    typename Engine8, typename Layout8,
    typename Engine9, typename Layout9,
    typename Engine10, typename Layout10,
    typename Engine11, typename Layout11,
    typename Engine12, typename Layout12,
    typename Engine13, typename Layout13,
    typename EngineSV, typename LayoutSV,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ void wg0_subroutine(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> &tKgK,
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> sK,
    Tensor<EngineSV, LayoutSV> &sV_direct,
    Tensor<Engine3, Layout3> &cur_sK0,
    Tensor<Engine4, Layout4> &cur_sK1,
    Tensor<Engine5, Layout5> &nxt_sK0,
    Tensor<Engine6, Layout6> &sP0,
    Tensor<Engine7, Layout7> &sP1,
    Tensor<Engine8, Layout8> &sM,
    Tensor<Engine9, Layout9> &sScale0,
    Tensor<Engine10, Layout10> &sScale1,
    Tensor<Engine11, Layout11> &rQ8,
    Tensor<Engine12, Layout12> &rP0,
    Tensor<Engine13, Layout13> &rO0,
    float rL[2],
    __mbarrier_t barriers_K0[9],
    __mbarrier_t barriers_K1[9],
    bool &cur_phase_K0,
    const Flash_fwd_mla_params &params,
    int* block_table_ptr,
    int seqlen_k,
    int total_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup,
    int wg_idx,
    int &kv_idx,
    // [FP8-WI fix3] sparse FP8 path: only batch_idx is needed -- staging
    // SMEM is gone (register-direct dequant in load_and_dequant_sparse_K).
    int batch_idx,
    int ori_block_max,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    // Pre-fetch pipeline: token index for current block (in/out)
    int *pre_token_idx,
    int *pre_token_idx_b,
    int extra_seqlen_k,
    WgKAddr<T> &precomp_ptr0,
    WgKAddr<T> &precomp_ptr1,
    bool &precomp_valid0,
    bool &precomp_valid1
) {
    int start_token_idx = block_idx * T::kBlockN;
    int nxt_block0 = block_idx+2;
    int nxt_block1 = block_idx+3;

    int prefetched_idx0 = -1, prefetched_idx1 = -1;
    if constexpr (T::kV41FP8 && !IS_BLK0_LAST && !IS_BLK1_LAST) {
        prefetched_idx0 = prefetch_v41_token_index<T>(
            params, batch_idx, nxt_block0, ori_block_max, idx_in_warpgroup);
        prefetched_idx1 = prefetch_v41_token_index<T>(
            params, batch_idx, nxt_block1, ori_block_max, idx_in_warpgroup);
    }

    // [Task #68] Derive V directly from K buffers. cur_sK0 holds even block (local),
    // cur_sK1 holds odd block (remote). V = first 512 dims of each K buffer.
    // Cross-WG smem_valid_indices: 4 bufs, WG0 uses 0/1, WG1 uses 2/3 (alternating)
    int vi_softmax_wg0 = (block_idx / 2) % 2;       // 0 or 1
    auto sV_local = make_tensor(cur_sK0.data(), (typename T::SmemLayoutVDirect){});
    Tensor sV0L = get_half_V<T, 0>(sV_local);
    auto sV_remote = make_tensor(cur_sK1.data(), (typename T::SmemLayoutVDirect){});
    Tensor sV1L = get_half_V<T, 0>(sV_remote);

    auto nxt_sK1 = cur_sK0;
    // Calc P0 = softmax(P0) and signal sScale0Ready before K load
    // (WG1 is waiting for sScale0Ready — arriving earlier lets WG1 start sooner)
#if ACOMPUTE_VERSION == 10000
    Tensor rPb = wg0_bunch_0< T, IS_BLK0_LAST || IS_BLK1_LAST > (rP0, rO0, sScale0, sM, rL, params.scale_softmax_log2, start_token_idx, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg0);
#else
    Tensor rPb = make_tensor<typename T::InputT>(Shape<Shape<_2, _2, _2>, _1, _2>{});
    wg0_bunch_0< T, IS_BLK0_LAST || IS_BLK1_LAST > (rPb, rP0, rO0, sScale0, sM, rL, params.scale_softmax_log2, start_token_idx, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg0);
#endif

    if constexpr (T::kUsePv4x2) {
        // WG1 reaches bar1 after its previous remote-P0 reads. This existing
        // rendezvous makes the old P0 buffer available before local PV.
        NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale0Ready);
        save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);
    } else {
        NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale0Ready);
    }

    // [BlockM=64 only] CTA-wide barrier for sScale0/sM cross-WG visibility.
    // On PPU M890, named barriers do NOT guarantee cross-wg SMEM visibility.
    // wg0 wrote sScale0(row) and sM(row) -- wg1 reads them after its matching
    // arrive_and_wait(sScale0Ready).  Paired with wg1's __syncthreads after
    // its arrive_and_wait(sScale0Ready).
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    // Cross-WG: wg0 loads tiles 0-3 for nxt_block0 (async cp.async, doesn't block)
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if constexpr (!T::IsFP8) {
            issue_K_load_bf16<0, 4, NEXT_EXTRA, T>(params, nxt_sK0, batch_idx, nxt_block0, seqlen_k, &barriers_K0[0],
                idx_in_warpgroup, ori_block_max, precomp_ptr0, precomp_valid0,
                pre_token_idx, nxt_block0 + 2, end_block_idx, extra_seqlen_k);
        } else {
            int vi_preload_wg0 = 1 - vi_softmax_wg0;         // 1 or 0
            load_and_dequant_sparse_K_staged<0, 4, T, T::kV41FP8>(
                params, sK, batch_idx, nxt_block0, seqlen_k,
                /*buf_idx=*/(kv_idx + 2) % 3, &barriers_K0[0],
                idx_in_warpgroup, ori_block_max, smem_valid_indices, vi_preload_wg0,
                prefetched_idx0);
        }
    }

    // Issue rO0 += rPb @ sV0L
    wg0_scale0_rO0<T>(rO0, sScale0, idx_in_warpgroup);

    if constexpr (T::kUsePv4x2) {
        __ppu_barrier_sync(6, 256, 15u);
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0L, rO0, idx_in_warpgroup, wg_idx);
    } else if constexpr (T::kIsCrossCut) {
        // (4,2) layout: each N-warp only has 16/32 P columns, localP broken.
        // Save rPb to sP0 temporarily, per-WG barrier, then remoteP from SMEM.
        save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);
        const int barrier_id = 6 + static_cast<int>(threadIdx.x >> 8);
        NamedBarrier::sync(
            256, static_cast<cutlass::arch::ReservedNamedBarriers>(barrier_id));
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0L, rO0, idx_in_warpgroup, wg_idx);
    } else {
        warpgroup_cooperative_pv_gemm_localP<T>(rPb, sV0L, rO0, idx_in_warpgroup, wg_idx);
    }

    // sScale1Ready also signals sP1 is ready (WG1 saves sP1 before arriving)
    // With PV4x2 this also drains WG0's local P0 readers before P0 is rescaled.
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale1Ready);

    // [BlockM=64 only] CTA-wide barrier for sScale1/sM cross-WG visibility.
    // wg1 wrote sScale1(row) and sM(row) in wg1_bunch_0; wg0 reads sScale1
    // below in wg0_scale_rP0.  Paired with wg1's __syncthreads after its
    // arrive(sScale1Ready).
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    // Cross-WG: wg0 loads tiles 0-3 for nxt_block1
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if constexpr (!T::IsFP8) {
            issue_K_load_bf16<0, 4, NEXT_EXTRA, T>(params, nxt_sK1, batch_idx, nxt_block1, seqlen_k, &barriers_K1[0],
                idx_in_warpgroup, ori_block_max, precomp_ptr1, precomp_valid1,
                pre_token_idx_b, nxt_block1 + 2, end_block_idx, extra_seqlen_k);
        } else {
            int vi_preload_wg1 = 3 - vi_softmax_wg0;    // 3 or 2 (BF16 nxt_block1 preload; FP8 skips nxt_block1)
            load_and_dequant_sparse_K_staged<0, 4, T, T::kV41FP8>(
                params, sK, batch_idx, nxt_block1, seqlen_k,
                /*buf_idx=*/kv_idx, &barriers_K1[0],
                idx_in_warpgroup, ori_block_max, smem_valid_indices, vi_preload_wg1,
                prefetched_idx1);
        }
    }

    wg0_scale_rP0<T>(sScale1, rP0, rPb, idx_in_warpgroup);
    save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);

    // Sync with WG1: ensures sP0 not overwritten before WG1 reads it.
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sP0Ready);

    // [BlockM=64 only] CTA-wide barrier for cross-wg SMEM visibility.
    // On PPU, named barriers do NOT guarantee SMEM visibility across wgs.
    // Only barrier 0 (__syncthreads) reliably flushes SMEM writes for
    // cross-wg reads.  Ensures wg0 can see V data written by wg1 (remote PV)
    // and wg1 can see V data written by wg0.  Paired with wg1's matching
    // __syncthreads after its arrive_and_wait(sP0Ready).
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    // sP1 ready via sScale1Ready. remote PV can proceed after sP0Ready sync.
    if constexpr (!IS_BLK0_LAST) {
        wg0_rescale_rO0<T>(rO0, sScale1, rL, idx_in_warpgroup);
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1L, rO0, idx_in_warpgroup, wg_idx);
    }

    // WG0 QK computes scores for next_block0 (always exists when !BLK0 && !BLK1)
    // Split-buf GEMM reads nxt_sK0 (tiles 0-3) + nxt_sK1 (tiles 4-8).
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        cute::clear(rP0);
        warpgroup_cooperative_qkt_gemm<T, 0>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
    }

    // After QK: precompute addr for NEXT iteration's K loads
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if constexpr (!T::IsFP8) {
            int next_nxt_block0 = block_idx + 4;
            precomp_ptr0 = compute_K_addr_bf16<0, NEXT_EXTRA, T>(
                params, batch_idx, next_nxt_block0, seqlen_k,
                idx_in_warpgroup, ori_block_max, smem_valid_indices, vi_softmax_wg0,
                pre_token_idx, extra_seqlen_k, precomp_valid0);
            int next_nxt_block1 = block_idx + 5;
            int next_vi_preload_wg1 = vi_softmax_wg0 + 2;
            precomp_ptr1 = compute_K_addr_bf16<0, NEXT_EXTRA, T>(
                params, batch_idx, next_nxt_block1, seqlen_k,
                idx_in_warpgroup, ori_block_max, smem_valid_indices, next_vi_preload_wg1,
                pre_token_idx_b, extra_seqlen_k, precomp_valid1);
        }
    }

    // Issue P0 = Q @ K0^T (next_block0 always exists)
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        warpgroup_cooperative_qkt_gemm<T, 2>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
    }

    // [BlockM=64 only] CTA-wide barrier at K-block iteration boundary.
    // Without this sync, wg0's PV-remote in the NEXT iteration reads a buffer
    // loaded by wg1 in THIS iteration (sV1L = cur_sK1), but wg0 has no
    // per-buffer mbarrier wait on barriers_K1.  bar.sync 0 forces all writes
    // to be visible to the other warpgroup before the next iteration begins.
    // Paired with wg1's matching __syncthreads at the same logical location.
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        kv_idx = (kv_idx + 2) % 3;
        cur_sK0 = sK(_, _, kv_idx);
        cur_sK1 = sK(_, _, (kv_idx + 1) % 3);
        nxt_sK0 = sK(_, _, (kv_idx + 2) % 3);
    }
}

template <
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool NEXT_EXTRA,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename Engine6, typename Layout6,
    typename Engine7, typename Layout7,
    typename Engine8, typename Layout8,
    typename Engine9, typename Layout9,
    typename Engine10, typename Layout10,
    typename Engine11, typename Layout11,
    typename Engine12, typename Layout12,
    typename Engine13, typename Layout13,
    typename EngineSV, typename LayoutSV,
    typename EngineVI, typename LayoutVI
>
__forceinline__ __device__ void wg1_subroutine(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> &tKgK,
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> sK,
    Tensor<EngineSV, LayoutSV> &sV_direct,
    Tensor<Engine3, Layout3> &cur_sK1,
    Tensor<Engine4, Layout4> &cur_sK0,
    Tensor<Engine5, Layout5> &nxt_sK1,
    Tensor<Engine6, Layout6> &sP0,
    Tensor<Engine7, Layout7> &sP1,
    Tensor<Engine8, Layout8> &sM,
    Tensor<Engine9, Layout9> &sScale0,
    Tensor<Engine10, Layout10> &sScale1,
    Tensor<Engine11, Layout11> &rQ8,
    Tensor<Engine12, Layout12> &rP1,
    Tensor<Engine13, Layout13> &rO1,
    float rL[2],
    __mbarrier_t barriers_K0[9],
    __mbarrier_t barriers_K1[9],
    bool &cur_phase_K1,
    const Flash_fwd_mla_params &params,
    int* block_table_ptr,
    int seqlen_k,
    int total_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup,
    int wg_idx,
    int &kv_idx,
    // [FP8-WI fix3] sparse FP8 path: only batch_idx is needed -- staging
    // SMEM is gone (register-direct dequant in load_and_dequant_sparse_K).
    int batch_idx,
    int ori_block_max,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices,
    // Pre-fetch pipeline: token index for current block (in/out)
    int *pre_token_idx,
    int *pre_token_idx_b,
    int extra_seqlen_k,
    WgKAddr<T> &precomp_ptr0,
    WgKAddr<T> &precomp_ptr1,
    bool &precomp_valid0,
    bool &precomp_valid1
) {
    int start_token_idx = block_idx * T::kBlockN;
    int nxt_block0 = block_idx+2;
    int nxt_block1 = block_idx+3;

    int prefetched_idx0 = -1, prefetched_idx1 = -1;
    if constexpr (T::kV41FP8 && !IS_BLK0_LAST && !IS_BLK1_LAST) {
        prefetched_idx0 = prefetch_v41_token_index<T>(
            params, batch_idx, nxt_block0, ori_block_max, idx_in_warpgroup);
        prefetched_idx1 = prefetch_v41_token_index<T>(
            params, batch_idx, nxt_block1, ori_block_max, idx_in_warpgroup);
    }

    auto nxt_sK0 = cur_sK1;
    // Derive V directly from K buffers. In wg1_subroutine, due to
    // LAUNCH_WG1_SUBROUTINE arg-swap: cur_sK0 = odd block (local for wg1),
    // cur_sK1 = even block (remote from wg0). V = first 512 dims of each K buffer.
    // Cross-WG smem_valid_indices: 4 bufs, WG0 uses 0/1, WG1 uses 2/3 (alternating)
    int vi_softmax_wg1 = 2 + (block_idx / 2) % 2;    // 2 or 3
    auto sV_local = make_tensor(cur_sK0.data(), (typename T::SmemLayoutVDirect){});
    Tensor sV0R = get_half_V<T, 1>(sV_local);
    auto sV_remote = make_tensor(cur_sK1.data(), (typename T::SmemLayoutVDirect){});
    Tensor sV1R = get_half_V<T, 1>(sV_remote);

    // Non-target paths retain their original early odd-high copy.
    // This copy does not write the validity slot owned by WG0.
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        constexpr int kNumTiles = T::kHeadDim / T::kBlockKSmem;
        if constexpr (!T::IsFP8) {
            if constexpr (T::kBlockM == 128 && T::kUsePv4x2 &&
                          T::CvtGemmSwzlLd &&
                          std::is_same_v<typename T::InputT, cutlass::bfloat16_t>) {
                // The payload was copied before this pair's QK. Commit exactly
                // once here, with the unchanged next-index cache selection.
                issue_K_load_bf16<4, kNumTiles, NEXT_EXTRA, T, true, false, true>(params, nxt_sK1, batch_idx, nxt_block1, seqlen_k, &barriers_K1[1],
                    idx_in_warpgroup, ori_block_max, precomp_ptr0, precomp_valid0,
                    pre_token_idx, nxt_block1 + 2, end_block_idx, extra_seqlen_k);
            } else if constexpr (!(T::kBlockM == 128 &&
                            std::is_same_v<typename T::InputT, cutlass::bfloat16_t>)) {
                issue_K_load_bf16<4, kNumTiles, NEXT_EXTRA, T>(params, nxt_sK1, batch_idx, nxt_block1, seqlen_k, &barriers_K1[1],
                    idx_in_warpgroup, ori_block_max, precomp_ptr0, precomp_valid0,
                    pre_token_idx, nxt_block1 + 2, end_block_idx, extra_seqlen_k);
            }
        } else {
            load_and_dequant_sparse_K_staged<4, kNumTiles, T, T::kV41FP8>(
                params, sK, batch_idx, nxt_block1, seqlen_k,
                /*buf_idx=*/(kv_idx + 2) % 3, &barriers_K1[0],
                idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                prefetched_idx1);
        }
    }

    // Pre-compute cur_max before barrier (doesn't depend on sM or sScale0)
    float r_cur_max[2];
    wg1_bunch_0_pre<T, IS_BLK0_LAST>(r_cur_max, rP1, params.scale_softmax_log2, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg1);

    // Wait for sScale0 from WG0 (delayed: cur_max already computed)
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale0Ready);

    if constexpr (!T::IsFP8 && T::kBlockM == 128 &&
                  std::is_same_v<typename T::InputT, cutlass::bfloat16_t> &&
                  !(T::kUsePv4x2 && T::CvtGemmSwzlLd) &&
                  !IS_BLK0_LAST && !IS_BLK1_LAST) {
        // The next high bank held the previous pair's remote V. Reuse it
        // only after every WG1 reader reaches the existing CTA rendezvous.
        constexpr int kNumTiles = T::kHeadDim / T::kBlockKSmem;
        issue_K_load_bf16<4, kNumTiles, NEXT_EXTRA, T>(params, nxt_sK1, batch_idx, nxt_block1, seqlen_k, &barriers_K1[1],
            idx_in_warpgroup, ori_block_max, precomp_ptr0, precomp_valid0,
            pre_token_idx, nxt_block1 + 2, end_block_idx, extra_seqlen_k);
    }

    // [BlockM=64 only] CTA-wide barrier for sScale0/sM cross-WG visibility.
    // Pairs with wg0's __syncthreads after its arrive(sScale0Ready).
    // After this point wg1 can safely read sScale0(row) and sM(row) written by wg0.
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

#if ACOMPUTE_VERSION == 10000
    Tensor rP1b = wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST, false>(sScale1, rO1, sM, rL, sScale0, rP1, params.scale_softmax_log2, start_token_idx+T::kBlockN, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg1, r_cur_max);
#else
    Tensor rP1b = make_tensor<typename T::InputT>(Shape<Shape<_2, _2, _2>, _1, _2>{});
    wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST, false>(rP1b, sScale1, rO1, sM, rL, sScale0, rP1, params.scale_softmax_log2, start_token_idx+T::kBlockN, idx_in_warpgroup, smem_valid_indices, vi_softmax_wg1, r_cur_max);
#endif

    // Save sP1 early (before sScale1Ready) so WG0 can read it without extra barrier
    if constexpr (!IS_BLK0_LAST) {
        save_rP1_to_sP<T>(rP1b, sP1, idx_in_warpgroup);
    }
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale1Ready);

    // [BlockM=64 only] CTA-wide barrier for sScale1/sM cross-WG visibility.
    // Pairs with wg0's __syncthreads after its arrive_and_wait(sScale1Ready).
    // wg1 wrote sScale1(row) and sM(row) in wg1_bunch_0 above; this ensures
    // wg0 sees the updated values when it reads them.
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    // Issue rO1 += rP1b @ sV1R
    if constexpr (T::kUsePv4x2) {
        __ppu_barrier_sync(7, 256, 15u);
    }
    wg1_scale0_rO1<T>(rO1, sScale0, sScale1, idx_in_warpgroup);
    if constexpr (!IS_BLK0_LAST) {
        if constexpr (T::kUsePv4x2) {
            warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1R, rO1, idx_in_warpgroup, wg_idx);
        } else if constexpr (T::kIsCrossCut) {
            // (4,2) layout: each N-warp only has 16/32 P columns, localP broken.
            // sP1 already saved above, use per-WG barrier then remoteP from SMEM.
            const int barrier_id = 6 + static_cast<int>(threadIdx.x >> 8);
            NamedBarrier::sync(
                256, static_cast<cutlass::arch::ReservedNamedBarriers>(barrier_id));
            warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1R, rO1, idx_in_warpgroup, wg_idx);
        } else {
            warpgroup_cooperative_pv_gemm_localP<T>(rP1b, sV1R, rO1, idx_in_warpgroup, wg_idx);
        }
    }

    // Sync with WG0: ensures WG0 doesn't overwrite sP0 before we read it.
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sP0Ready);

    // [BlockM=64 only] CTA-wide barrier for cross-wg SMEM visibility.
    // Matches the __syncthreads() in wg0_subroutine after its arrive(sP0Ready).
    // Also implicitly publishes wg1's sP1 store to wg0 for remote PV GEMM.
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    // Cross-WG: wg1 loads tiles 4-8 for nxt_block0
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        constexpr int kNumTiles = T::kHeadDim / T::kBlockKSmem;
        if constexpr (!T::IsFP8) {
            issue_K_load_bf16<4, kNumTiles, NEXT_EXTRA, T>(params, nxt_sK0, batch_idx, nxt_block0, seqlen_k, &barriers_K0[1],
                idx_in_warpgroup, ori_block_max, precomp_ptr1, precomp_valid1,
                pre_token_idx_b, nxt_block0 + 2, end_block_idx, extra_seqlen_k);
        } else {
            load_and_dequant_sparse_K_staged<4, kNumTiles, T, T::kV41FP8>(
                params, sK, batch_idx, nxt_block0, seqlen_k,
                /*buf_idx=*/kv_idx, &barriers_K0[0],
                idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                prefetched_idx0);
        }
    }

    // After QK: precompute addr for NEXT iteration's K loads
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if constexpr (!T::IsFP8) {
            int next_nxt_block1 = block_idx + 5;
            precomp_ptr0 = compute_K_addr_bf16<4, NEXT_EXTRA, T>(
                params, batch_idx, next_nxt_block1, seqlen_k,
                idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                pre_token_idx, extra_seqlen_k, precomp_valid0);
            int next_nxt_block0 = block_idx + 4;
            precomp_ptr1 = compute_K_addr_bf16<4, NEXT_EXTRA, T>(
                params, batch_idx, next_nxt_block0, seqlen_k,
                idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                pre_token_idx_b, extra_seqlen_k, precomp_valid1);
        }
    }

    // Remote PV GEMM: reads sP0 (ready via sP0Ready) and sV0R
    warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0R, rO1, idx_in_warpgroup, wg_idx);

    if constexpr (!T::IsFP8 && T::kBlockM == 128 && T::kUsePv4x2 &&
                  T::CvtGemmSwzlLd &&
                  std::is_same_v<typename T::InputT, cutlass::bfloat16_t> &&
                  !IS_BLK0_LAST && !IS_BLK1_LAST) {
        if (block_idx + 4 < end_block_idx) {
            // Rendezvous after the old high-bank reader calls. The next payload
            // does not alias either bank used by the following QK.
            __ppu_barrier_sync(7, 256, 15u);
            constexpr int kNumTiles = T::kHeadDim / T::kBlockKSmem;
            issue_K_load_bf16<4, kNumTiles, NEXT_EXTRA, T, false, true, false>(params, cur_sK0, batch_idx, block_idx + 5, seqlen_k, &barriers_K1[1],
                idx_in_warpgroup, ori_block_max, precomp_ptr0, precomp_valid0,
                pre_token_idx, block_idx + 7, end_block_idx, extra_seqlen_k);
        }
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        cute::clear(rP1);
        warpgroup_cooperative_qkt_gemm<T, 1>(sQ, nxt_sK0, nxt_sK1, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup, wg_idx);
    }

    // [BlockM=64 only] CTA-wide barrier at K-block iteration boundary.
    // Mirrors the __syncthreads() at the end of wg0_subroutine.  Required so
    // wg1's PV-remote in the NEXT iteration sees the K data wg0 loaded in
    // THIS iteration (sV1R = cur_sK1 maps to wg0's previously loaded buf).
    if constexpr (T::kBlockM == 64) {
        __syncthreads();
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        kv_idx = (kv_idx + 2) % 3;
        cur_sK1 = sK(_, _, kv_idx);
        cur_sK0 = sK(_, _, (kv_idx + 1) % 3);
        nxt_sK1 = sK(_, _, (kv_idx + 2) % 3);
    }
}

// A helper function for determining the length of the causal mask for one q token
__forceinline__ __device__ int get_mask_len(const Flash_fwd_mla_params &params, int m_block_idx, int local_seq_q_idx) {
    int global_seq_q_idx = m_block_idx*Config::BLOCK_SIZE_M + local_seq_q_idx;
    if (global_seq_q_idx < params.seqlen_q) {
        int s_q_idx = global_seq_q_idx / params.ngroups;
        return params.q_orig - s_q_idx - 1;
    } else {
        // Out-of-bound request, regard as no masks
        return 0;
    }
}


template<int QArrivals>
__forceinline__ __device__ void wg_init_copy_barriers(
    __mbarrier_t *barrier_Q, __mbarrier_t *barriers_K0, __mbarrier_t *barriers_K1) {
    if (threadIdx.x == 0) {
        __mbarrier_init(barrier_Q, QArrivals);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
            __mbarrier_init(&barriers_K0[i], 256);
            __mbarrier_init(&barriers_K1[i], 256);
        }
    }
}

// Both cache policies launch the original WG kernel. The paired policy
// keeps its task scheduler and load pipeline, and shares the tile math above.
template<typename T>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
flash_sparse_decode_wg_kernel(__grid_constant__ const Flash_fwd_mla_params params);

namespace flashmla::dsa::v41 {
using namespace cute;
using cutlass::arch::NamedBarrier;
using ::get_AorC_row_idx;

// Adapted from https://github.com/Dao-AILab/flash-attention/blob/cdaf2de6e95cb05400959b5ab984f66e4c7df317/hopper/utils.h
// * Copyright (c) 2024, Tri Dao.
using ::gemm;

using ::kernel_sleep_ns;

#undef QKT_GEMM_ONE_TILE
using ::warpgroup_cooperative_qkt_gemm;


using ::warpgroup_cooperative_pv_gemm_localP;
using ::warpgroup_cooperative_pv_gemm_remoteP;


// Adapt the paired pipeline's register-fragment return convention to the
// shared WG softmax helpers. Both use the same SM89 accumulator mapping.
template<typename T, bool Last, bool AllValid, typename P, typename O,
         typename Scale, typename Max, typename Valid>
__forceinline__ __device__ auto paired_wg0_bunch_0(
    P &p, O &o, Scale &scale, Max &maxima, float sums[2], float sm_scale,
    int start, int tid, Valid &valid, int slot) {
    Tensor result = make_tensor<typename T::InputT>(Shape<Shape<_2, _2, _2>, _1, _2>{});
    ::wg0_bunch_0<T, Last, AllValid>(result, p, o, scale, maxima, sums,
                                    sm_scale, start, tid, valid, slot);
    return result;
}

template<typename T, bool Last0, bool Last1, typename Scale1, typename O,
         typename Max, typename Scale0, typename P, typename Valid>
__forceinline__ __device__ auto paired_wg1_bunch_0(
    Scale1 &scale1, O &o, Max &maxima, float sums[2], const Scale0 &scale0,
    P &p, float sm_scale, int start, int tid, Valid &valid, int slot,
    float cur_max[2]) {
    Tensor result = make_tensor<typename T::InputT>(Shape<Shape<_2, _2, _2>, _1, _2>{});
    ::wg1_bunch_0<T, Last0, Last1, false>(result, scale1, o, maxima, sums,
                                        scale0, p, sm_scale, start, tid,
                                        valid, slot, cur_max);
    return result;
}

using ::save_rP1_to_sP;

using ::save_rP0_to_sP;

using ::wg0_scale_rP0;

using ::wg0_rescale_rO0;

using ::wg1_scale0_rO1;

using ::wg0_scale0_rO0;

using ::store_o;

template <
    typename T,
    typename Tensor0>
__forceinline__ __device__ void launch_sparse_decode_v41_q(
    const typename T::Params &params,
    int m_block_idx,
    Tensor0 &sQ,
    const int tidx,
    const int warp_idx,
    __mbarrier_t* barrier_Q
) {
    using Element = typename T::InputT;
    const int m_block = m_block_idx % (params.h_q / T::kBlockM);
    const int64_t s_q_idx = m_block_idx / (params.h_q / T::kBlockM);
    const int64_t row_offset_q = s_q_idx * params.stride_q_s_q + int64_t(m_block) * T::kBlockM * params.stride_q_h_q;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q) + row_offset_q),
                            Shape<Int<T::kBlockM>, Int<T::kHeadDim>>{},
                            make_stride(params.stride_q_h_q, _1{}));
    typename T::GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(make_mix_tensor_like(gQ));

    gmem_tiled_copy_Q.desc_.init(nullptr, T::kBlockM, params.d_qk, params.stride_q_h_q);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    if (warp_idx < T::kBlockM / T::kQRowsPerLoad) {
        CUTLASS_PRAGMA_UNROLL
        for (int tile = 0; tile < T::NUM_TILES; ++tile) {
            auto dst = tQsQ(_, warp_idx, tile);
            const int offset = (tile & ~1) * T::kBlockM * T::kBlockKSmem
                + ((warp_idx << 1) + (tile & 1)) * T::kQRowsPerLoad * T::kBlockKSmem;
            dst.data() = tQsQ.data() + offset;
            cute::copy(gmem_tiled_copy_Q, tQgQ(_, warp_idx, tile), dst);
        }
        cutlass::arch::cpasync_barrier_arrive_noinc(barrier_Q);
    }
}

using ::get_half_V;

template <typename T,
          bool IS_BLK0_LAST, // "BLK0" means block_idx+0, "BLK1" means block_idx+1, ...
          bool IS_BLK1_LAST, typename Engine1,
          typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3,
          typename Engine4, typename Layout4, typename Engine5, typename Layout5, typename Engine6,
          typename Layout6, typename Engine7, typename Layout7, typename Engine8, typename Layout8,
          typename Engine9, typename Layout9, typename Engine10, typename Layout10, typename Engine11,
          typename Layout11, typename Engine12, typename Layout12, typename Engine13, typename Layout13,
          typename EngineVI, typename LayoutVI>
__forceinline__ __device__ void dsa_wg0_subroutine(
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> sK, Tensor<Engine3, Layout3> &cur_sK0, Tensor<Engine4, Layout4> &cur_sK1,
    Tensor<Engine5, Layout5> &nxt_sK0, Tensor<Engine6, Layout6> &sP0, Tensor<Engine7, Layout7> &sP1,
    Tensor<Engine8, Layout8> &sM, Tensor<Engine9, Layout9> &sScale0, Tensor<Engine10, Layout10> &sScale1,
    Tensor<Engine11, Layout11> &rQ8, Tensor<Engine12, Layout12> &rP0, Tensor<Engine13, Layout13> &rO0,
    float rL[2], __mbarrier_t barriers_K0[T::kHeadDim / 256], __mbarrier_t barriers_K1[T::kHeadDim / 256],
    bool &cur_phase_K0, const typename T::Params &params, int block_idx,
    int end_block_idx,      // [Even-align] ROUNDED-up (even) block count: loop/issue/compute guards
    int real_end_block_idx, // [Even-align] REAL topk block count: gIndices prefetch guards only
    int idx_in_warpgroup, int wg_idx, int &kv_idx, int &nxt_token_idx0, int &nxt_token_idx1,
    V41PrecomputedKAddress<T> &precomp_ptr0, V41PrecomputedKAddress<T> &precomp_ptr1, bool &precomp_valid0,
    bool &precomp_valid1, V41QuantRegisterPacket<T> &first_register_packet,
    Tensor<EngineVI, LayoutVI> &smem_valid_indices) {
    using InputT = typename T::InputT;
    int start_token_idx = block_idx * T::kBlockN;
    int nxt_block0 = block_idx + 4;
    int nxt_block1 = block_idx + 5;
    InputT *gK_base = reinterpret_cast<InputT *>(params.kv);

    Tensor sV0L = get_half_V<T, 0>(cur_sK0);
    Tensor sV1L = get_half_V<T, 0>(cur_sK1);

    auto nxt_sK1 = cur_sK0;

    // Both formats carry the WG0 first packet across preceding QK.

    Tensor rPb = paired_wg0_bunch_0<T, IS_BLK0_LAST || IS_BLK1_LAST, T::kAllValidNoLengths> (rP0, rO0, sScale0, sM, rL, params.sm_scale_div_log2, start_token_idx,
                               idx_in_warpgroup, smem_valid_indices, (block_idx / 2) % 2);
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale0Ready);
    if constexpr (T::kDecodeFP4WG0Scale0SchedBound && T::kDecode && T::kFP4Cache && !T::kMixedFP4 &&
                  T::kDecodeFixedExtraIndex && T::kAllValidNoLengths && !IS_BLK0_LAST && !IS_BLK1_LAST) {
        static_assert(T::Arch_value == 89 && T::kUsePairedCubes && T::kBlockM == 128 && T::kBlockN == 32 &&
                      T::kHeadDim == 512);
        __ppu_sched_bound();
    }
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        dsa_issue_K_load<0, 4, T, true, true, false, T::kQuantCache>(
            &barriers_K0[0], precomp_ptr0, precomp_valid0,  nxt_block0,
            real_end_block_idx, nxt_token_idx0, nxt_sK0, idx_in_warpgroup, params, nxt_token_idx0,
            &first_register_packet, block_idx + 2);
    }
    V41QuantRegisterPacket<T> second_register_packet{};
    if constexpr (T::kQuantCache) {
        if (!IS_BLK0_LAST && !IS_BLK1_LAST && __builtin_expect(block_idx + 3 < end_block_idx, true)) {
            second_register_packet = quant_cache_register_load_prepared<0, 4, T>(
                params, idx_in_warpgroup, nxt_token_idx1, precomp_valid1, block_idx + 3, precomp_ptr1);
        }
    }
    wg0_scale0_rO0<T>(rO0, sScale0, idx_in_warpgroup);
    warpgroup_cooperative_pv_gemm_localP<T>(rPb, sV0L, rO0, idx_in_warpgroup, wg_idx);
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale1Ready);
    constexpr bool finish_after_p0 = T::kDecodeFP8WG0FinishAfterP0 && T::kDecode && !T::kFP4Cache &&
                                     !T::kMixedFP4 && T::kDecodeFixedExtraIndex && T::kAllValidNoLengths &&
                                     !IS_BLK0_LAST && !IS_BLK1_LAST;
    if constexpr (!finish_after_p0) {
        if (!IS_BLK0_LAST && !IS_BLK1_LAST && __builtin_expect(block_idx + 3 < end_block_idx, true)) {
            dsa_issue_K_load<0, 4, T, true, false, false, true>(
                &barriers_K1[0], precomp_ptr1, precomp_valid1,
                nxt_block1, real_end_block_idx, nxt_token_idx1, nxt_sK1, idx_in_warpgroup, params,
                nxt_token_idx1, &second_register_packet, block_idx + 3);
        }
    }
    wg0_scale_rP0<T>(sScale1, rP0, rPb, idx_in_warpgroup);
    save_rP0_to_sP<T>(rPb, sP0, idx_in_warpgroup);
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sP0Ready);
    if constexpr (finish_after_p0) {
        static_assert(T::kUsePairedCubes && T::kBlockN == 32 && T::kHeadDim == 512);
        if (!IS_BLK0_LAST && !IS_BLK1_LAST && __builtin_expect(block_idx + 3 < end_block_idx, true)) {
            dsa_issue_K_load<0, 4, T, true, false, false, true>(
                &barriers_K1[0], precomp_ptr1, precomp_valid1,
                nxt_block1, real_end_block_idx, nxt_token_idx1, nxt_sK1, idx_in_warpgroup, params,
                nxt_token_idx1, &second_register_packet, block_idx + 3);
        }
    }
    if constexpr (!IS_BLK0_LAST) {
        wg0_rescale_rO0<T>(rO0, sScale1, rL, idx_in_warpgroup);
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1L, rO0, idx_in_warpgroup, wg_idx);
    }
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        cute::clear(rP0);
        warpgroup_cooperative_qkt_gemm<T, 0>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0,
                                                 idx_in_warpgroup, wg_idx);
    }
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if (block_idx + 4 < end_block_idx) {
            dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx0, block_idx + 4,
                                        idx_in_warpgroup, smem_valid_indices, (block_idx / 2) % 2,
                                        precomp_ptr0, precomp_valid0);
        }
        if (block_idx + 5 < end_block_idx) {
            dsa_compute_K_addr<T, false>(params, gK_base, nxt_token_idx1, block_idx + 5,
                                         idx_in_warpgroup, smem_valid_indices, 0, precomp_ptr1,
                                         precomp_valid1);
        }
    }
    if constexpr (kFP4AsyncEarly && T::kQuantCache && !IS_BLK0_LAST && !IS_BLK1_LAST) {
        if (block_idx + 4 < end_block_idx) {
            first_register_packet = quant_cache_register_load_prepared<0, 4, T>(
                params, idx_in_warpgroup, nxt_token_idx0, precomp_valid0, block_idx + 4, precomp_ptr0);
        }
    }
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        warpgroup_cooperative_qkt_gemm<T, 2>(sQ, nxt_sK0, nxt_sK1, rP0, rQ8, barriers_K0, cur_phase_K0,
                                                 idx_in_warpgroup, wg_idx);
    }
    kv_idx = (kv_idx + 2) % 3;
    cur_sK0 = sK(_, _, kv_idx);
    cur_sK1 = sK(_, _, (kv_idx + 1) % 3);
    nxt_sK0 = sK(_, _, (kv_idx + 2) % 3);
}

template <typename T, bool IS_BLK0_LAST, bool IS_BLK1_LAST, typename Engine1, typename Layout1, typename Engine2, typename Layout2,
          typename Engine3, typename Layout3, typename Engine4, typename Layout4, typename Engine5,
          typename Layout5, typename Engine6, typename Layout6, typename Engine7, typename Layout7,
          typename Engine8, typename Layout8, typename Engine9, typename Layout9, typename Engine10,
          typename Layout10, typename Engine11, typename Layout11, typename Engine12, typename Layout12,
          typename Engine13, typename Layout13, typename EngineVI, typename LayoutVI>
__forceinline__ __device__ void dsa_wg1_subroutine(
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> sK, Tensor<Engine3, Layout3> &cur_sK1, Tensor<Engine4, Layout4> &cur_sK0,
    Tensor<Engine5, Layout5> &nxt_sK1, Tensor<Engine6, Layout6> &sP0, Tensor<Engine7, Layout7> &sP1,
    Tensor<Engine8, Layout8> &sM, Tensor<Engine9, Layout9> &sScale0, Tensor<Engine10, Layout10> &sScale1,
    Tensor<Engine11, Layout11> &rQ8, Tensor<Engine12, Layout12> &rP1, Tensor<Engine13, Layout13> &rO1,
    float rL[2], __mbarrier_t barriers_K0[T::kHeadDim / 256], __mbarrier_t barriers_K1[T::kHeadDim / 256],
    bool &cur_phase_K1, const typename T::Params &params, int block_idx,
    int end_block_idx,      // [Even-align] ROUNDED-up (even) block count: loop/issue/compute guards
    int real_end_block_idx, // [Even-align] REAL topk block count: gIndices prefetch guards only
    int idx_in_warpgroup, int wg_idx, int &kv_idx, int &nxt_token_idx0, int &nxt_token_idx1,
    V41PrecomputedKAddress<T> &precomp_ptr0, V41PrecomputedKAddress<T> &precomp_ptr1, bool &precomp_valid0,
    bool &precomp_valid1, Tensor<EngineVI, LayoutVI> &smem_valid_indices) {
    using InputT = typename T::InputT;
    int start_token_idx = block_idx * T::kBlockN;
    int nxt_block0 = block_idx + 4;
    int nxt_block1 = block_idx + 5;
    InputT *gK_base = reinterpret_cast<InputT *>(params.kv);

    auto nxt_sK0 = cur_sK1;
    Tensor sV0R = get_half_V<T, 1>(cur_sK0);
    Tensor sV1R = get_half_V<T, 1>(cur_sK1);

    constexpr bool first_finish_after_scale1 = T::kDecodeFP4WG1FirstFinishAfterScale1 && T::kDecode &&
                                               T::kFP4Cache && !T::kMixedFP4 && T::kDecodeFixedExtraIndex &&
                                               T::kAllValidNoLengths && !IS_BLK0_LAST && !IS_BLK1_LAST;
    float r_cur_max[2];
    constexpr bool supply_after_scale1 = T::kDecodeFP8WG1SupplyAfterScale1 && T::kDecode && !T::kFP4Cache &&
                                         !T::kMixedFP4 && T::kDecodeFixedExtraIndex &&
                                         T::kAllValidNoLengths && !IS_BLK0_LAST && !IS_BLK1_LAST;
    auto early_rP1b = [&]() {
        V41QuantRegisterPacket<T> first_register_packet{};
        if constexpr (T::kDecodeFP8WG1EarlyPacket && T::kAllValidNoLengths && !IS_BLK0_LAST &&
                      !IS_BLK1_LAST) {
            static_assert(T::kDecode && !T::kMixedFP4 && !T::kFP4Cache && T::kDecodeFP8ScopeRow32);
            first_register_packet = quant_cache_register_load_prepared<4, T::NUM_TILES, T>(
                params, idx_in_warpgroup, nxt_token_idx1, true, block_idx + 3, precomp_ptr0);
        }
        if constexpr (kFP4AsyncEarly && T::kFP4Cache && !IS_BLK0_LAST && !IS_BLK1_LAST) {
            const int staged_token = nxt_token_idx1;
            const bool staged_valid = precomp_valid0;
            T::CacheAdapter::template async_issue<4, T::NUM_TILES, T>(params, idx_in_warpgroup, staged_token,
                                                                      staged_valid, block_idx + 3);
        } else if constexpr (kFP4AsyncEarly && T::kDecode && T::kMixedFP4 && T::kDecodeExtraFP4AsyncEarly &&
                             !IS_BLK0_LAST && !IS_BLK1_LAST) {
            T::CacheAdapter::template async_issue<4, T::NUM_TILES, T>(
                params, idx_in_warpgroup, nxt_token_idx1, precomp_valid0, block_idx + 3);
        }
        ::wg1_bunch_0_pre<T, IS_BLK0_LAST, T::kAllValidNoLengths>(r_cur_max, rP1, params.sm_scale_div_log2,
                                                           idx_in_warpgroup,
                                                           smem_valid_indices, 2 + (block_idx / 2) % 2);
        NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale0Ready);
        auto published_rP1b = [&]() {
            if constexpr (supply_after_scale1) {
                static_assert(T::Arch_value == 89 && T::kDecodeFP8WG1EarlyPacket && T::kUsePairedCubes &&
                              T::kBlockM == 128 && T::kBlockN == 32 && T::kHeadDim == 512);
                Tensor rP1b = paired_wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST>(
                    sScale1, rO1, sM, rL, sScale0, rP1, params.sm_scale_div_log2,
                    start_token_idx + T::kBlockN, idx_in_warpgroup, smem_valid_indices,
                    2 + (block_idx / 2) % 2, r_cur_max);
                if constexpr (!IS_BLK0_LAST) {
                    save_rP1_to_sP<T>(rP1b, sP1, idx_in_warpgroup);
                }
                NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale1Ready);
                __ppu_sched_bound();
                return rP1b;
            } else {
                return cute::make_tuple();
            }
        }();
        if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST &&
                      !first_finish_after_scale1) {
            dsa_issue_K_load<4, T::NUM_TILES, T, true, false,
                             (kFP4AsyncEarly &&
                              (T::kFP4Cache || (T::kDecode && T::kMixedFP4 && T::kDecodeExtraFP4AsyncEarly))),
                             (T::kDecodeFP8WG1EarlyPacket && T::kAllValidNoLengths)>(
                &barriers_K1[1], precomp_ptr0,
                (T::kDecodeFP8WG1EarlyPacket && T::kAllValidNoLengths) ? true : precomp_valid0,
                nxt_block1, real_end_block_idx, nxt_token_idx1, nxt_sK1, idx_in_warpgroup, params,
                nxt_token_idx1,
                (T::kDecodeFP8WG1EarlyPacket && T::kAllValidNoLengths) ? &first_register_packet : nullptr,
                block_idx + 3);
        }
        return published_rP1b;
    }();
    V41QuantRegisterPacket<T> second_register_packet{};
    if constexpr (T::kQuantCache && !IS_BLK0_LAST && !IS_BLK1_LAST) {
        second_register_packet = quant_cache_register_load_prepared<4, T::NUM_TILES, T>(
            params, idx_in_warpgroup, nxt_token_idx0, precomp_valid1, block_idx + 2, precomp_ptr1);
    }
    Tensor rP1b = [&]() {
        if constexpr (supply_after_scale1) {
            return early_rP1b;
        } else {
            return paired_wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST>(
                sScale1, rO1, sM, rL, sScale0, rP1, params.sm_scale_div_log2, start_token_idx + T::kBlockN,
                idx_in_warpgroup, smem_valid_indices, 2 + (block_idx / 2) % 2, r_cur_max);
        }
    }();
    if constexpr (!supply_after_scale1) {
        if constexpr (!IS_BLK0_LAST) {
            save_rP1_to_sP<T>(rP1b, sP1, idx_in_warpgroup);
        }
        NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale1Ready);
    }
    if constexpr (first_finish_after_scale1) {
        static_assert(T::Arch_value == 89 && T::kUsePairedCubes && T::kBlockM == 128 && T::kBlockN == 32 &&
                      T::kHeadDim == 512 && kFP4AsyncEarly && !T::kDecodeFP8WG1EarlyPacket &&
                      T::kDecodeFP4WG1RemotePVBeforeFinish);
        dsa_issue_K_load<4, T::NUM_TILES, T, true, false,
                         (kFP4AsyncEarly &&
                          (T::kFP4Cache || (T::kDecode && T::kMixedFP4 && T::kDecodeExtraFP4AsyncEarly))),
                         (T::kDecodeFP8WG1EarlyPacket && T::kAllValidNoLengths)>(
            &barriers_K1[1], precomp_ptr0,
            (T::kDecodeFP8WG1EarlyPacket && T::kAllValidNoLengths) ? true : precomp_valid0,
            nxt_block1, real_end_block_idx, nxt_token_idx1, nxt_sK1, idx_in_warpgroup, params, nxt_token_idx1,
            nullptr, block_idx + 3);
    }
    wg1_scale0_rO1<T>(rO1, sScale0, sScale1, idx_in_warpgroup);
    if constexpr (!IS_BLK0_LAST) {
        warpgroup_cooperative_pv_gemm_localP<T>(rP1b, sV1R, rO1, idx_in_warpgroup, wg_idx);
    }
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sP0Ready);
    constexpr bool remote_pv_before_finish = T::kDecodeFP4WG1RemotePVBeforeFinish && T::kDecode &&
                                             T::kFP4Cache && T::kAllValidNoLengths && !IS_BLK0_LAST &&
                                             !IS_BLK1_LAST;
    if constexpr (remote_pv_before_finish) {
        static_assert(T::kUsePairedCubes && T::kBlockN == 32 && T::kHeadDim == 512);
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0R, rO1, idx_in_warpgroup, wg_idx);
    }
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        dsa_issue_K_load<4, T::NUM_TILES, T, true, false, false, true>(
            &barriers_K0[1], precomp_ptr1, precomp_valid1,  nxt_block0,
            real_end_block_idx, nxt_token_idx0, nxt_sK0, idx_in_warpgroup, params, nxt_token_idx0,
            &second_register_packet, block_idx + 2);
    }
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        if (block_idx + 5 < end_block_idx) {
            dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx1, block_idx + 5,
                                        idx_in_warpgroup, smem_valid_indices, 2 + (block_idx / 2) % 2,
                                        precomp_ptr0, precomp_valid0);
        }
        if (block_idx + 4 < end_block_idx) {
            dsa_compute_K_addr<T, false>(params, gK_base, nxt_token_idx0, block_idx + 4,
                                         idx_in_warpgroup, smem_valid_indices, 0, precomp_ptr1,
                                         precomp_valid1);
        }
    }
    if constexpr (!remote_pv_before_finish) {
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0R, rO1, idx_in_warpgroup, wg_idx);
    }
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        cute::clear(rP1);
        warpgroup_cooperative_qkt_gemm<T, 1>(sQ, nxt_sK0, nxt_sK1, rP1, rQ8, barriers_K1, cur_phase_K1,
                                                 idx_in_warpgroup, wg_idx);
    }
    kv_idx = (kv_idx + 2) % 3;
    cur_sK1 = sK(_, _, kv_idx);
    cur_sK0 = sK(_, _, (kv_idx + 1) % 3);
    nxt_sK1 = sK(_, _, (kv_idx + 2) % 3);
}

template<typename T>
__forceinline__ __device__ void
sparse_wi_tile_driver(const typename T::Params &params, int m_block_idx) {
    static_assert(T::kDecode && T::kQuantCache);
    const int warpgroup_idx = __builtin_ppu_to_uniform_b32(threadIdx.x / 256);
    const int idx_in_warpgroup = threadIdx.x % 256;
    const int warp_idx = __builtin_ppu_to_uniform_b32(threadIdx.x / 32);

    const int tidx = threadIdx.x;
    using InputT = typename T::InputT;
    typename T::TiledMma tiled_mma;

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    using SharedMemoryPlan = typename T::SharedMemoryPlan;
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan *>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.smem_sQ.data()), (typename T::SmemLayoutQ){});
    Tensor sK = make_tensor(make_smem_ptr(plan.smem_sK.data()), (typename T::SmemLayoutK){});
    auto p_base = [&]() {
        if constexpr (T::kUsePairedCubes && T::kHeadDim == 512) {
            return make_smem_ptr(plan.smem_sP.data());
        } else {
            return flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, Int<T::PAGE_BLOCK_SIZE>>{})(_, _, _0{}, Int<T::NUM_TILES - 1>{}).data();
        }
    };
    Tensor sP0 = make_tensor(p_base(), (typename T::SmemLayoutP0){});
    Tensor sP1 = make_tensor(sP0.data() + sP0.size(), (typename T::SmemLayoutP0){});
    Tensor sM = make_tensor(make_smem_ptr(plan.smem_sM.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sL_reduction_wksp = make_tensor(make_smem_ptr(plan.sL_reduction_wksp.data()), make_shape(Int<2 * T::BLOCK_SIZE_M>{}));
    Tensor sScale0 = make_tensor(make_smem_ptr(plan.smem_sScale0.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sScale1 = make_tensor(make_smem_ptr(plan.smem_sScale1.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(plan.smem_valid_indices.data()),
        Shape<_4, Int<T::kBlockN>>{}, Stride<Int<T::kBlockN>, _1>{});
    char *sO_addr = (char *)plan.smem_sQ.data(); // Overlap with sK0 and sK1
    InputT* gK_base = reinterpret_cast<InputT*>(params.kv);
    int nxt_token_idx0 = -1;
    int nxt_token_idx1 = -1;
    V41PrecomputedKAddress<T> precomp_ptr0 = {}, precomp_ptr1 = {};
    bool precomp_valid0 = false, precomp_valid1 = false;
    constexpr int kBlockN = T::kBlockN;

    // Define TMA stuffs
    __mbarrier_t *barrier_Q = &(plan.barrier_Q);
    __mbarrier_t *barriers_K0 = plan.barriers_K0;
    __mbarrier_t *barriers_K1 = plan.barriers_K1;

    // Initialize TMA barriers
    wg_init_copy_barriers<32 * (T::kBlockM / T::kQRowsPerLoad)>(barrier_Q, barriers_K0, barriers_K1);
    if (threadIdx.x < size(sM)) {
        sM[threadIdx.x] = MAX_INIT_VAL_SM;
    }
    __syncthreads();

    // Tasks cover padded main/extra block ranges. Index lookup checks each
    // scope's stored and valid length before reading its index row.
    int seqlen_k=params.topk;
    int start_block_idx=params.start_block;
    int real_end_block_idx=cute::ceil_div(seqlen_k,kBlockN);
    int end_block_idx=params.end_block;

    if (warpgroup_idx == 0) {
        nxt_token_idx0 = quant_wi_token_index<T>(params,start_block_idx,real_end_block_idx,idx_in_warpgroup);
    } else {
        nxt_token_idx1 = quant_wi_token_index<T>(params,start_block_idx+1,real_end_block_idx,idx_in_warpgroup);
    }

    bool cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;
    // Copy the first Q
    launch_sparse_decode_v41_q<T>(params, m_block_idx, sQ, tidx, warp_idx, barrier_Q);

    // Copy K0 and K1

    Tensor cur_sK0 = sK(_, _, 0);
    Tensor cur_sK1 = sK(_, _, 1);
    Tensor nxt_sK0 = sK(_, _, 2);

    if (warpgroup_idx == 0 && seqlen_k != 0) {
        V41PrecomputedKAddress<T> addr_blk0; bool valid_blk0;
        dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx0, start_block_idx, idx_in_warpgroup,
            smem_valid_indices, (start_block_idx/2)%2, addr_blk0, valid_blk0);
        dsa_issue_K_load<4, T::NUM_TILES, T, false, false>(&barriers_K0[1],
            addr_blk0, valid_blk0,  0, real_end_block_idx, nxt_token_idx0, cur_sK1, idx_in_warpgroup, params, nxt_token_idx0,nullptr,start_block_idx);
        dsa_issue_K_load<0, 4, T, false, false>(&barriers_K0[0],
            addr_blk0, valid_blk0,  0, real_end_block_idx, nxt_token_idx0, cur_sK0, idx_in_warpgroup, params, nxt_token_idx0,nullptr,start_block_idx);
        // [Even-align] guarded by the REAL block count; padding blocks -> -1 (invalid).
        nxt_token_idx0 = quant_wi_token_index<T>(params,start_block_idx+2,real_end_block_idx,idx_in_warpgroup);
        nxt_token_idx1 = quant_wi_token_index<T>(params,start_block_idx+3,real_end_block_idx,idx_in_warpgroup);
        // WG-scope barrier: makes the flag stores above visible to all 256 threads
        // of WG0 before iteration-0's wg0_bunch_0 reads them (no other barrier sits
        // between the prolog write and that read). WG0-only: does not couple the
        // WG0/WG1 arrive/wait skew, hence no cross-WG sync stall.
        NamedBarrier::arrive_and_wait(T::NUM_THREADS / 2,
            static_cast<cutlass::arch::ReservedNamedBarriers>(1));
        if (start_block_idx + 2 < end_block_idx) {
            dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx0, start_block_idx + 2, idx_in_warpgroup,
                smem_valid_indices, (start_block_idx/2+1)%2, precomp_ptr0, precomp_valid0);
        }
        if (start_block_idx + 3 < end_block_idx) {
            dsa_compute_K_addr<T, false>(params, gK_base, nxt_token_idx1, start_block_idx + 3, idx_in_warpgroup,
                smem_valid_indices, 0, precomp_ptr1, precomp_valid1);
        }
    }

    if (start_block_idx+1 < end_block_idx) {
        if (warpgroup_idx == 1) {
            V41PrecomputedKAddress<T> addr_blk1; bool valid_blk1;
            dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx1, start_block_idx + 1, idx_in_warpgroup,
                smem_valid_indices, 2+(start_block_idx/2)%2, addr_blk1, valid_blk1);
            dsa_issue_K_load<4, T::NUM_TILES, T, false, false>(&barriers_K1[1],
                addr_blk1, valid_blk1,  0, real_end_block_idx, nxt_token_idx0, cur_sK0, idx_in_warpgroup, params, nxt_token_idx1,nullptr,start_block_idx+1);
            dsa_issue_K_load<0, 4, T, false, false>(&barriers_K1[0],
                addr_blk1, valid_blk1,  0, real_end_block_idx, nxt_token_idx0, cur_sK1, idx_in_warpgroup, params, nxt_token_idx1,nullptr,start_block_idx+1);
            // [Even-align] guarded by the REAL block count; padding blocks -> -1 (invalid).
            nxt_token_idx0 = quant_wi_token_index<T>(params,start_block_idx+2,real_end_block_idx,idx_in_warpgroup);
            nxt_token_idx1 = quant_wi_token_index<T>(params,start_block_idx+3,real_end_block_idx,idx_in_warpgroup);
            // Publish block-1 validity to WG1 before its first softmax read,
            // which precedes the cross-WG sScale0Ready rendezvous.
            NamedBarrier::arrive_and_wait(T::NUM_THREADS / 2,
            static_cast<cutlass::arch::ReservedNamedBarriers>(2));
            if (start_block_idx + 3 < end_block_idx) {
                dsa_compute_K_addr<T, true>(params, gK_base, nxt_token_idx1, start_block_idx + 3, idx_in_warpgroup,
                    smem_valid_indices, 2+(start_block_idx/2+1)%2, precomp_ptr0, precomp_valid0);
            }
            if (start_block_idx + 2 < end_block_idx) {
                dsa_compute_K_addr<T, false>(params, gK_base, nxt_token_idx0, start_block_idx + 2, idx_in_warpgroup,
                    smem_valid_indices, 0, precomp_ptr1, precomp_valid1);
            }
        }
    }

    Tensor rO = partition_fragment_C((typename T::TiledMma){}, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV / 2>>{});	// ((2, 2, 32), 1, 1)
    float rL[2];
    rL[0] = rL[1] = 0.0f;

    // Clear buffers
    cute::fill(rO, 0.);

    while(!cutlass::arch::test_wait(barrier_Q, cur_phase_Q, 1)) {
        kernel_sleep_ns();
    }
    cur_phase_Q = (cur_phase_Q + 1) & 1;

    static_assert(!T::kCacheLastQ);
    Tensor rQ8 = make_tensor<InputT>(Shape<Shape<_2, _2, _2>, _1, _4>{});

    if (warpgroup_idx == 0) {
        // Warpgroup 0
        Tensor rP0 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});  // MMA, MMA_M, MMA_K
        const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

        int idx = 0;
        int block_idx = start_block_idx;
        bool native_phase=false;
        int native_begin=0;
        if constexpr(T::kDecodeNativeFP4Steady || T::kDecodeFixedExtraFP8Steady) {
            native_phase=params.main_padded_length==128 && params.extra_length>0
                && end_block_idx>params.main_padded_length/T::kBlockN;
            native_begin=max(start_block_idx,params.main_padded_length/T::kBlockN-2);
        }
        using SteadyFirstPacket=std::conditional_t<T::kDecodeNativeFP4Steady,
            V41Fp4RegisterPacket,std::conditional_t<T::kDecodeFixedExtraFP8Steady,
            V41QuantRegisterPacket<T>,int>>;
        SteadyFirstPacket native_first{};
        #define DSA_LAUNCH_WG0_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST)                     \
            dsa_wg0_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST>(                            \
            sQ, sK, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
            rP0, rO, rL,                   \
            barriers_K0, barriers_K1, cur_phase_K0, params, \
            block_idx, end_block_idx, real_end_block_idx, idx_in_warpgroup, wg_idx, idx, nxt_token_idx0, nxt_token_idx1, precomp_ptr0, precomp_ptr1, precomp_valid0, precomp_valid1 \
            , first_register_packet, smem_valid_indices);
        #define V41_QUANT_WG0_TYPED(TRAITS, PACKET, IS_BLK0_LAST, IS_BLK1_LAST)                     \
            dsa_wg0_subroutine<TRAITS, IS_BLK0_LAST, IS_BLK1_LAST>(                            \
            sQ, sK, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
            rP0, rO, rL,                   \
            barriers_K0, barriers_K1, cur_phase_K0, params, \
            block_idx, end_block_idx, real_end_block_idx, idx_in_warpgroup, wg_idx, idx, nxt_token_idx0, nxt_token_idx1, precomp_ptr0, precomp_ptr1, precomp_valid0, precomp_valid1 \
            , PACKET, smem_valid_indices);
        {
            V41QuantRegisterPacket<T> first_register_packet{};
            if constexpr(kFP4AsyncEarly && T::kQuantCache) {
                if (start_block_idx + 2 < end_block_idx) {
                    first_register_packet=quant_cache_register_load_prepared<0,4,T>(params,idx_in_warpgroup,nxt_token_idx0,precomp_valid0,start_block_idx+2,precomp_ptr0);
                }
            }

            // NOTE We don't use the pipelined version of Q K^T here since it leads
            // to a slow-down (or even register spilling, thanks to the great NVCC)
            // Issue P0 = Q @ K0^T, wait
            if (seqlen_k !=0) {
                cute::clear(rP0);
                warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK0, cur_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
            }

            if constexpr(T::kDecodeNativeFP4Steady || T::kDecodeFixedExtraFP8Steady) {
                if(native_phase) {
                    #pragma unroll 1
                    for (; block_idx<end_block_idx-2 && block_idx<native_begin; block_idx += 2) {
                        DSA_LAUNCH_WG0_SUBROUTINE(false,false);
                    }
                    // A two-block tail has no preloaded future packet.
                    if(block_idx+2<end_block_idx) {
                        if constexpr(T::kDecodeNativeFP4Steady) native_first=first_register_packet.fp4;
                        else native_first=first_register_packet;
                    }
                } else {
                    #pragma unroll 1
                    for (; block_idx < end_block_idx-2; block_idx += 2) {
                        DSA_LAUNCH_WG0_SUBROUTINE(false, false);
                    }

                    // [Even-align] end_block_idx is always even, so the loop leaves exactly
                    // one uniform 2-block tail; the IS_BLK0_LAST single-block branch is
                    // unreachable and removed (mirrors decode's single tail form).
                    if (block_idx < end_block_idx) {
                        DSA_LAUNCH_WG0_SUBROUTINE(false, true);
                    }
                }
            } else {
                #pragma unroll 1
                for (; block_idx < end_block_idx-2; block_idx += 2) {
                    DSA_LAUNCH_WG0_SUBROUTINE(false, false);
                }

                // [Even-align] end_block_idx is always even, so the loop leaves exactly
                // one uniform 2-block tail; the IS_BLK0_LAST single-block branch is
                // unreachable and removed (mirrors decode's single tail form).
                if (block_idx < end_block_idx) {
                    DSA_LAUNCH_WG0_SUBROUTINE(false, true);
                }
            }
        } // The parent packet is dead before the fixed-scope steady loop.
        if constexpr(T::kDecodeNativeFP4Steady || T::kDecodeFixedExtraFP8Steady) {
            if(native_phase) {
                using NativeT=std::conditional_t<T::kDecodeNativeFP4Steady,
                    SparseDecodeV41NativeFP4SteadyTraits<T>,SparseDecodeV41FixedExtraFP8SteadyTraits<T>>;
                static_assert(std::is_same_v<typename NativeT::TiledMma,typename T::TiledMma>);
                static_assert(std::is_same_v<typename NativeT::SharedMemoryPlan,typename T::SharedMemoryPlan>);
                static_assert(std::is_same_v<typename NativeT::SmemLayoutK,typename T::SmemLayoutK>);
                static_assert(std::is_same_v<typename NativeT::SmemLayoutP0,typename T::SmemLayoutP0>);
                #pragma unroll 1
                for (; block_idx<end_block_idx-2; block_idx += 2) {
                    V41_QUANT_WG0_TYPED(NativeT,native_first,false,false);
                }
                if (block_idx < end_block_idx) {
                    V41_QUANT_WG0_TYPED(NativeT,native_first,false,true);
                }
            }
        }
        #undef V41_QUANT_WG0_TYPED
    }
    else {
        // // Warpgroup 1
        // Tensor rP1 = make_tensor<float>((typename T::rP0Layout){});
        Tensor rP1 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});  // MMA, MMA_M, MMA_K
        const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

        if (start_block_idx+1 < end_block_idx) {
            // Issue rP1 = sQ @ sK1, wait
            warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK1, cur_sK0, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup, wg_idx);
        }

        int idx = 0;
        #define DSA_LAUNCH_WG1_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST)                     \
            dsa_wg1_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST>(                            \
            sQ, sK, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
            rP1, rO, rL,                 \
            barriers_K0, barriers_K1, cur_phase_K1, params, \
            block_idx, end_block_idx, real_end_block_idx, idx_in_warpgroup, wg_idx, idx, nxt_token_idx0, nxt_token_idx1, precomp_ptr0, precomp_ptr1, precomp_valid0, precomp_valid1 \
            , smem_valid_indices);
        #define V41_QUANT_WG1_TYPED(TRAITS, IS_BLK0_LAST, IS_BLK1_LAST)                     \
            dsa_wg1_subroutine<TRAITS, IS_BLK0_LAST, IS_BLK1_LAST>(                            \
            sQ, sK, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
            rP1, rO, rL,                 \
            barriers_K0, barriers_K1, cur_phase_K1, params, \
            block_idx, end_block_idx, real_end_block_idx, idx_in_warpgroup, wg_idx, idx, nxt_token_idx0, nxt_token_idx1, precomp_ptr0, precomp_ptr1, precomp_valid0, precomp_valid1 \
            , smem_valid_indices);
        int block_idx = start_block_idx;
        if constexpr(T::kDecodeNativeFP4Steady || T::kDecodeFixedExtraFP8Steady) {
            const bool native_phase=params.main_padded_length==128 && params.extra_length>0
                && end_block_idx>params.main_padded_length/T::kBlockN;
            const int native_begin=max(start_block_idx,params.main_padded_length/T::kBlockN-2);
            if(native_phase) {
                using NativeT=std::conditional_t<T::kDecodeNativeFP4Steady,
                    SparseDecodeV41NativeFP4SteadyTraits<T>,SparseDecodeV41FixedExtraFP8SteadyTraits<T>>;
                static_assert(std::is_same_v<typename NativeT::TiledMma,typename T::TiledMma>);
                static_assert(std::is_same_v<typename NativeT::SharedMemoryPlan,typename T::SharedMemoryPlan>);
                static_assert(std::is_same_v<typename NativeT::SmemLayoutK,typename T::SmemLayoutK>);
                #pragma unroll 1
                for (; block_idx<end_block_idx-2 && block_idx<native_begin; block_idx += 2) {
                    DSA_LAUNCH_WG1_SUBROUTINE(false,false);
                }
                #pragma unroll 1
                for (; block_idx<end_block_idx-2; block_idx += 2) {
                    V41_QUANT_WG1_TYPED(NativeT,false,false);
                }
                if (block_idx < end_block_idx) {
                    V41_QUANT_WG1_TYPED(NativeT,false,true);
                }
            } else {
                // [Even-align] end_block_idx is always even: bound end-2 gives the same
                // trip count as the old end-3, and the loop leaves exactly one uniform
                // 2-block tail.
                #pragma unroll 1
                for (; block_idx < end_block_idx-2; block_idx += 2) {
                    DSA_LAUNCH_WG1_SUBROUTINE(false, false);
                }

                if (block_idx < end_block_idx) {
                    DSA_LAUNCH_WG1_SUBROUTINE(false, true);
                }
            }
        } else {
            // [Even-align] end_block_idx is always even: bound end-2 gives the same
            // trip count as the old end-3, and the loop leaves exactly one uniform
            // 2-block tail.
            #pragma unroll 1
            for (; block_idx < end_block_idx-2; block_idx += 2) {
                DSA_LAUNCH_WG1_SUBROUTINE(false, false);
            }

            if (block_idx < end_block_idx) {
                DSA_LAUNCH_WG1_SUBROUTINE(false, true);
            }
        }
        #undef V41_QUANT_WG1_TYPED
    }

    // Reduce rL across threads within the same warp
    rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
    rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
    rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
    rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

    // Reduce rL across warpgroups
    int my_row = get_AorC_row_idx(0, idx_in_warpgroup);
    float pre_sink0 = 0.0f, pre_sink1 = 0.0f;
    const bool apply_sink=!params.partial_output && params.main_length+params.extra_length>0;
    if (params.attn_sink != nullptr && apply_sink) {
        int head_block_idx = m_block_idx % (params.h_q / T::kBlockM);
        pre_sink0 = __ldg(params.attn_sink + head_block_idx * T::BLOCK_SIZE_M + my_row);
        pre_sink1 = __ldg(params.attn_sink + head_block_idx * T::BLOCK_SIZE_M + my_row + 8);
    }
    wg_reduce_softmax_sums<T>(rL, sL_reduction_wksp, my_row, warpgroup_idx, idx_in_warpgroup);

    // Apply attention sink: adjust rL to account for the sink term in the normalization denominator.
    // rL_adjusted = rL + exp2f(sink_log2 - sM[row])
    // where sink_log2 = attn_sink[head_idx] * M_LOG2E (convert natural log to log2 space)
    // and sM[row] is the log2-space scaled max (max(raw_score) * sm_scale_div_log2)
    // This ensures output = rO / rL_adjusted = sum(P*V) / (sum(P) + exp(attn_sink))
    // LSE is NOT affected (it reads from sL_reduction_wksp, not rL)
    if (params.attn_sink != nullptr && apply_sink) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
            int row_idx = get_AorC_row_idx(i, idx_in_warpgroup);
            // attn_sink value pre-issued above (pre_sink0/pre_sink1, before the
            // rL-reduction __syncthreads); only the arithmetic stays here.
            float sink_log2 = (i == 0 ? pre_sink0 : pre_sink1) * (float)M_LOG2E;
            rL[i] += exp2f(sink_log2 - sM(row_idx));
        }
    }

    // Epilogue
    int num_valid_seq_q = min(params.s_q * params.h_q - m_block_idx * T::BLOCK_SIZE_M, T::BLOCK_SIZE_M);
    if(params.partial_output) {
        // WG1 has just published the reduced denominator in the lower stats bank.
        __syncthreads();
        float *o_ptr=static_cast<float*>(params.out)+int64_t(m_block_idx)*T::kBlockM*T::kHeadDimV;
        Tensor gOAccum=make_tensor(make_gmem_ptr(o_ptr),make_layout(
            Shape<Int<T::kBlockM>,Int<T::kHeadDimV>>{},make_stride(T::kHeadDimV,_1{})));
        // The full FP32 output buffer overlays the entire shared plan.
        // Read and publish LSE before output copies can overwrite its stats.
        if(threadIdx.x<num_valid_seq_q) {
            const int row=threadIdx.x;
            const float cur_L=sL_reduction_wksp[row];
            static_cast<float*>(params.lse)[m_block_idx*T::kBlockM+row]=
                (cur_L==0.0f || cur_L!=cur_L)?-INFINITY:log2f(cur_L)+sM(row);
        }
        __syncthreads();
        store_o<T,false>(rO,gOAccum,rL,sO_addr,num_valid_seq_q,warpgroup_idx,idx_in_warpgroup);
        return;
    }
    InputT *o_ptr = (InputT *)params.out + (int64_t)m_block_idx * T::BLOCK_SIZE_M * T::kHeadDimV;
    float *softmax_lse_ptr = (float *)params.lse + m_block_idx * T::BLOCK_SIZE_M;

    Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                                                        Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV>>{},
                                                        make_stride(T::kHeadDimV, _1{})));
    Tensor gSoftmaxLse = make_tensor(make_gmem_ptr(softmax_lse_ptr), Layout<
                                                                            Shape<Int<T::BLOCK_SIZE_M>>,
                                                                            Stride<_1>>{});

    store_o<T,true>(rO, gO, rL, sO_addr, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

    int i = threadIdx.x;
    if (i < num_valid_seq_q) {
        float cur_L = sL_reduction_wksp[i];
        gSoftmaxLse(i) = (cur_L == 0.0f || cur_L != cur_L) ? INFINITY : logf(cur_L) + sM(i) / (float)M_LOG2E;
    }
}

#undef QKT_GEMM_ONE_TILE
#undef DSA_LAUNCH_WG0_SUBROUTINE
#undef DSA_LAUNCH_WG1_SUBROUTINE

}

namespace flashmla::dsa::v41 {

struct SparseDecodeV41DirectScheduler;
struct SparseDecodeV41BoundedPrefixScheduler;

template<bool ExtraFP4>
struct SparseDecodeV41PairedTraits : SparseDecodeV41Traits {
    using Params=SparseDecodeV41TileParams;
    using KernelParams=Flash_fwd_params;
    using Scheduler=SparseDecodeV41Scheduler;
    using CacheAdapter=SparseDecodeV41CacheAdapter;
    static constexpr bool kDecode=true;
    static constexpr bool kMixedFP4=ExtraFP4;
    static constexpr bool kDecodeExtraFP4AsyncEarly=true;
    static constexpr bool kTaskAllValidGuard=false;
    static constexpr bool kDecodePageBase64=!ExtraFP4;
    static constexpr bool kSingleTaskCTA=false;
    static constexpr bool kLengthGuardAllowed=false;
};
template<bool ExtraFP4>
struct SparseDecodeV41GuardedPairedTraits : SparseDecodeV41PairedTraits<ExtraFP4> {
    static_assert(ExtraFP4);
    static constexpr bool kTaskAllValidGuard=true;
};
struct SparseDecodeV41ScopeNarrowGuardedTraits : SparseDecodeV41GuardedPairedTraits<true> {
    static constexpr bool kCacheOffset32=true;
};
template<bool ExtraFP4>
struct SparseDecodeV41DirectPairedTraits : SparseDecodeV41PairedTraits<ExtraFP4> {
    using Scheduler=SparseDecodeV41DirectScheduler;
    static constexpr bool kSingleTaskCTA=true;
};
struct SparseDecodeV41Sq1CheckedU32FP8Traits : SparseDecodeV41DirectPairedTraits<false> {
    static constexpr bool kDecodeFP8ScopeRow32=true;
    static constexpr bool kDecodePageBase64=false;
};
struct SparseDecodeV41Sq1CheckedOffset32FP4Traits : SparseDecodeV41DirectPairedTraits<true> {
    static constexpr bool kCacheOffset32=true;
};
struct SparseDecodeV41Sq1PersistentU32FP8Traits : SparseDecodeV41Sq1CheckedU32FP8Traits {
    using Scheduler=SparseDecodeV41Scheduler;
    static constexpr bool kSingleTaskCTA=false;
};
struct SparseDecodeV41Sq1PersistentOffset32FP4Traits : SparseDecodeV41Sq1CheckedOffset32FP4Traits {
    using Scheduler=SparseDecodeV41Scheduler;
    static constexpr bool kSingleTaskCTA=false;
};

template<bool ExtraFP4>
struct SparseDecodeV41DirectGuardedPairedTraits : SparseDecodeV41DirectPairedTraits<ExtraFP4> {
    static constexpr bool kTaskAllValidGuard=true;
};
struct SparseDecodeV41DirectScopeNarrowGuardedTraits : SparseDecodeV41DirectGuardedPairedTraits<true> {
    static constexpr bool kCacheOffset32=true;
};

struct SparseDecodeV41DirectScopeNarrowNativeFP4WG1PublishFirstRemotePVTraits : SparseDecodeV41DirectScopeNarrowGuardedTraits {
    static constexpr bool kDecodeNativeFP4Steady=true;
    static constexpr bool kDecodeFP4WG1RemotePVBeforeFinish=true;
    static constexpr bool kDecodeFP4WG1FirstFinishAfterScale1=true;
    static constexpr bool kDecodeFP4WG0Scale0SchedBound=true;
};
struct SparseDecodeV41DirectScopeU32GuardedPairedTraits : SparseDecodeV41DirectGuardedPairedTraits<false> {
    static constexpr bool kDecodeFP8ScopeRow32=true;
    static constexpr bool kDecodePageBase64=false;
};
struct SparseDecodeV41DirectScopeU32FP8WG1PublishBoundWG0PostP0Traits : SparseDecodeV41DirectScopeU32GuardedPairedTraits {
    static constexpr bool kDecodeFixedExtraFP8Steady=true;
    static constexpr bool kDecodeFP8WG1EarlyPacket=true;
    static constexpr bool kDecodeFP8WG1SupplyAfterScale1=true;
    static constexpr bool kDecodeFP8WG0FinishAfterP0=true;
};
struct SparseDecodeV41Sq1LengthsGuardedFP8Traits
    : SparseDecodeV41DirectScopeU32FP8WG1PublishBoundWG0PostP0Traits {
    static constexpr bool kLengthGuardAllowed=true;
};
struct SparseDecodeV41Sq1LengthsGuardedFP4Traits
    : SparseDecodeV41DirectScopeNarrowNativeFP4WG1PublishFirstRemotePVTraits {
    static constexpr bool kLengthGuardAllowed=true;
};
struct SparseDecodeV41Sq1LengthsPrefixFP8Traits : SparseDecodeV41Sq1LengthsGuardedFP8Traits {
    using Scheduler=SparseDecodeV41BoundedPrefixScheduler;
    static constexpr bool kSingleTaskCTA=true;
};
struct SparseDecodeV41Sq1LengthsPrefixFP4Traits : SparseDecodeV41Sq1LengthsGuardedFP4Traits {
    using Scheduler=SparseDecodeV41BoundedPrefixScheduler;
    static constexpr bool kSingleTaskCTA=true;
};

template<typename T>
struct SparseDecodeV41AllValidTaskTraits : T {
    static constexpr bool kAllValidNoLengths=true;
};

template<typename T,bool FP4>
struct SparseDecodeV41ScopeTraits : T {
    using Params=SparseDecodeV41CacheParams;
    static constexpr bool kDecode=false;
    static constexpr bool kMixedFP4=false;
    static constexpr bool kFP4Cache=FP4;
};

__forceinline__ __device__ SparseDecodeV41CacheParams sparse_decode_v41_cache_scope(
        const SparseDecodeV41TileParams &params,int block_idx) {
    const bool extra=block_idx*32>=params.main_padded_length;
    SparseDecodeV41CacheParams scope{};
    scope.kv=extra?params.extra_kv:params.kv;
    scope.quant_page_size=extra?params.extra_page_size:params.quant_page_size;
    scope.quant_block_stride=extra?params.extra_block_stride:params.quant_block_stride;
    return scope;
}
struct SparseDecodeV41FixedExtraFP4CacheAdapter {
    template<typename T>
    static __forceinline__ __device__ SparseDecodeV41CacheParams scope(const typename T::Params &p) {
        static_assert(T::kDecode && T::kFP4Cache && !T::kMixedFP4);
        SparseDecodeV41CacheParams result{};
        result.kv=p.extra_kv;
        result.quant_page_size=p.extra_page_size;
        result.quant_block_stride=p.extra_block_stride;
        return result;
    }

    template<int S,int E,typename T>
    static __forceinline__ __device__ V41Fp4RegisterPacket register_load(
            const typename T::Params &p,int tidx,int token_idx,bool valid,int) {
        return quant_cache_register_load<S,E,SparseDecodeV41ScopeTraits<T,true>>(
            scope<T>(p),tidx,token_idx,valid);
    }

    template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK,typename Packet>
    static __forceinline__ __device__ void register_finish(
            const typename T::Params &,RawK &raw_sK,int tidx,int,bool,const Packet &packet) {
        static_assert(std::is_same_v<Packet,V41Fp4RegisterPacket>);
        quant_cache_fp4_register_finish<S,E,SparseDecodeV41ScopeTraits<T,true>,FRONT_DRAIN>(
            raw_sK,tidx,packet);
    }

    template<int S,int E,typename T>
    static __forceinline__ __device__ void async_issue(
            const typename T::Params &p,int tidx,int token_idx,bool valid,int) {
        quant_cache_fp4_async_issue<S,E,SparseDecodeV41ScopeTraits<T,true>>(
            scope<T>(p),tidx,token_idx,valid);
    }

    template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
    static __forceinline__ __device__ void load(
            const typename T::Params &p,RawK &raw_sK,int tidx,int token_idx,
            bool valid,int loaded_block,bool async_preissued) {
        if(!async_preissued) async_issue<S,E,T>(p,tidx,token_idx,valid,loaded_block);
        quant_cache_fp4_async_finish<S,E,SparseDecodeV41ScopeTraits<T,true>,FRONT_DRAIN>(raw_sK,tidx);
    }
};
struct SparseDecodeV41FixedExtraFP8CacheAdapter : SparseDecodeV41CacheAdapter {
    template<typename T>
    static __forceinline__ __device__ V41RowAddress32 prepare_scope_row32(
            const typename T::Params &p,int token_idx,bool valid,int) {
        static_assert(T::kDecode && T::kDecodeFP8ScopeRow32 && !T::kMixedFP4 && !T::kFP4Cache);
        V41RowAddress32 row{};
        if(valid) {
            const uint32_t page=uint32_t(token_idx)>>__builtin_ctz(p.extra_page_size);
            const uint32_t off=uint32_t(token_idx)&uint32_t(p.extra_page_size-1);
            row.row_payload_offset=page*uint32_t(p.extra_block_stride)+off*528u;
            row.scale_delta=512u;
        }
        return row;
    }

    template<typename T>
    static __forceinline__ __device__ V41RowAddress64 widen(
            const typename T::Params &p,bool valid,const V41RowAddress32 &source) {
        V41RowAddress64 row{};
        if(valid) {
            row.payload_row=reinterpret_cast<const uint8_t*>(p.extra_kv)
                +uint64_t(source.row_payload_offset);
            row.scale_delta=int64_t(uint64_t(source.scale_delta));
        }
        return row;
    }

    template<int S,int E,typename T>
    static __forceinline__ __device__ auto register_load_scope_row32(
            const typename T::Params &p,int tidx,int token_idx,bool valid,
            int,const V41RowAddress32 &source) {
        const auto row=widen<T>(p,valid,source);
        return quant_cache_register_load<S,E,SparseDecodeV41ScopeTraits<T,false>>(
            tidx,valid,row);
    }

    template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
    static __forceinline__ __device__ void load_scope_row32(
            const typename T::Params &p,RawK &raw_sK,int tidx,int token_idx,
            bool valid,int,const V41RowAddress32 &source) {
        const auto row=widen<T>(p,valid,source);
        quant_cache_load_fp8<S,E,SparseDecodeV41ScopeTraits<T,false>,FRONT_DRAIN>(
            raw_sK,tidx,valid,row);
    }
};

template<typename T>
__forceinline__ __device__ V41RowAddress32 SparseDecodeV41CacheAdapter::prepare_scope_row32(
        const typename T::Params &params,int token_idx,bool valid,int loaded_block) {
    static_assert(T::kDecode && T::kDecodeFP8ScopeRow32 && !T::kMixedFP4);
    V41RowAddress32 row{};
    if(valid) {
        const auto scope=sparse_decode_v41_cache_scope(params,loaded_block);
        const uint32_t page=uint32_t(token_idx)>>__builtin_ctz(scope.quant_page_size);
        const uint32_t off=uint32_t(token_idx)&uint32_t(scope.quant_page_size-1);
        row.row_payload_offset=page*uint32_t(scope.quant_block_stride)+off*528u;
        row.scale_delta=512u;
    }
    return row;
}
template<typename T>
__forceinline__ __device__ V41RowAddress64 sparse_decode_v41_scope_row32_widen(
        const typename T::Params &params,bool valid,int loaded_block,
        const V41RowAddress32 &source) {
    V41RowAddress64 row{};
    if(valid) {
        const auto scope=sparse_decode_v41_cache_scope(params,loaded_block);
        row.payload_row=reinterpret_cast<const uint8_t*>(scope.kv)
            +uint64_t(source.row_payload_offset);
        row.scale_delta=int64_t(uint64_t(source.scale_delta));
    }
    return row;
}

template<int S,int E,typename T>
__forceinline__ __device__ auto SparseDecodeV41CacheAdapter::register_load_scope_row32(
        const typename T::Params &params,int tidx,int token_idx,bool valid,
        int loaded_block,const V41RowAddress32 &source) {
    static_assert(T::kDecode && T::kDecodeFP8ScopeRow32 && !T::kMixedFP4);
    const auto row=sparse_decode_v41_scope_row32_widen<T>(params,valid,loaded_block,source);
    return quant_cache_register_load<S,E,SparseDecodeV41ScopeTraits<T,false>>(
        tidx,valid,row);
}

template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
__forceinline__ __device__ void SparseDecodeV41CacheAdapter::load_scope_row32(
        const typename T::Params &params,RawK &raw_sK,int tidx,int token_idx,
        bool valid,int loaded_block,const V41RowAddress32 &source) {
    static_assert(T::kDecode && T::kDecodeFP8ScopeRow32 && !T::kMixedFP4);
    const auto row=sparse_decode_v41_scope_row32_widen<T>(params,valid,loaded_block,source);
    quant_cache_load_fp8<S,E,SparseDecodeV41ScopeTraits<T,false>,FRONT_DRAIN>(
        raw_sK,tidx,valid,row);
}

template<typename T>
__forceinline__ __device__ typename T::InputT *SparseDecodeV41CacheAdapter::prepare_page_base64(
        const typename T::Params &params,int token_idx,bool valid,int loaded_block) {
    static_assert(T::kDecode && T::kDecodePageBase64 && !T::kMixedFP4);
    if(!valid) return nullptr;
    const auto scope=sparse_decode_v41_cache_scope(params,loaded_block);
    const int page=token_idx>>__builtin_ctz(scope.quant_page_size);
    return reinterpret_cast<typename T::InputT*>(reinterpret_cast<uint8_t*>(scope.kv)
        +int64_t(page)*scope.quant_block_stride);
}

template<typename T>
__forceinline__ __device__ V41RowAddress64 sparse_decode_v41_local_page_row64(
        const typename T::Params &params,int token_idx,bool valid,int loaded_block,
        const typename T::InputT *page_base) {
    V41RowAddress64 row{};
    if(valid) {
        const auto scope=sparse_decode_v41_cache_scope(params,loaded_block);
        const int off=token_idx&(scope.quant_page_size-1);
        row.payload_row=reinterpret_cast<const uint8_t*>(page_base)+int64_t(off)*528;
        row.scale_delta=512;
    }
    return row;
}

template<int S,int E,typename T>
__forceinline__ __device__ auto SparseDecodeV41CacheAdapter::register_load_page64(
        const typename T::Params &params,int tidx,int token_idx,bool valid,
        int loaded_block,const typename T::InputT *page_base) {
    static_assert(T::kDecode && T::kDecodePageBase64 && !T::kMixedFP4);
    const auto row=sparse_decode_v41_local_page_row64<T>(params,token_idx,valid,loaded_block,page_base);
    return quant_cache_register_load<S,E,SparseDecodeV41ScopeTraits<T,false>>(
        tidx,valid,row);
}

template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
__forceinline__ __device__ void SparseDecodeV41CacheAdapter::load_page64(
        const typename T::Params &params,RawK &raw_sK,int tidx,int token_idx,
        bool valid,int loaded_block,const typename T::InputT *page_base) {
    static_assert(T::kDecode && T::kDecodePageBase64 && !T::kMixedFP4);
    const auto row=sparse_decode_v41_local_page_row64<T>(params,token_idx,valid,loaded_block,page_base);
    quant_cache_load_fp8<S,E,SparseDecodeV41ScopeTraits<T,false>,FRONT_DRAIN>(
        raw_sK,tidx,valid,row);
}

template<int S,int E,typename T>
__forceinline__ __device__ auto SparseDecodeV41CacheAdapter::register_load(
        const typename T::Params &params,int tidx,int token_idx,bool valid,int block_idx) {
    const auto scope=sparse_decode_v41_cache_scope(params,block_idx);
    if constexpr(T::kMixedFP4) {
        V41MixedRegisterPacket packet{};
        packet.fp4_format=block_idx*T::kBlockN>=params.main_padded_length;
        if(packet.fp4_format) {
            packet.fp4=quant_cache_register_load<S,E,SparseDecodeV41ScopeTraits<T,true>>(
                scope,tidx,token_idx,valid);
        } else {
            packet.fp8=quant_cache_register_load<S,E,SparseDecodeV41ScopeTraits<T,false>>(
                scope,tidx,token_idx,valid);
        }
        return packet;
    } else {
        return quant_cache_register_load<S,E,SparseDecodeV41ScopeTraits<T,false>>(
            scope,tidx,token_idx,valid);
    }
}

template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK,typename Packet>
__forceinline__ __device__ void SparseDecodeV41CacheAdapter::register_finish(
        const typename T::Params &,RawK &raw_sK,int tidx,int token_idx,
        bool valid,const Packet &packet) {
    if constexpr(T::kMixedFP4) {
        if(packet.fp4_format) {
            quant_cache_fp4_register_finish<S,E,SparseDecodeV41ScopeTraits<T,true>,FRONT_DRAIN>(
                raw_sK,tidx,packet.fp4);
        } else {
            quant_cache_fp8_register_finish<S,E,SparseDecodeV41ScopeTraits<T,false>,FRONT_DRAIN>(
                raw_sK,tidx,valid,packet.fp8);
        }
    } else {
        quant_cache_fp8_register_finish<S,E,SparseDecodeV41ScopeTraits<T,false>,FRONT_DRAIN>(
            raw_sK,tidx,valid,packet);
    }
}

template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
__forceinline__ __device__ void SparseDecodeV41CacheAdapter::load(
        const typename T::Params &params,RawK &raw_sK,int tidx,int token_idx,
        bool valid,int block_idx,bool async_preissued) {
    const auto scope=sparse_decode_v41_cache_scope(params,block_idx);
    if constexpr(T::kMixedFP4) {
        if(block_idx*T::kBlockN>=params.main_padded_length) {
            using ScopeT=SparseDecodeV41ScopeTraits<T,true>;
            if(!async_preissued) quant_cache_fp4_async_issue<S,E,ScopeT>(scope,tidx,token_idx,valid);
            quant_cache_fp4_async_finish<S,E,ScopeT,FRONT_DRAIN>(raw_sK,tidx);
        } else {
            quant_cache_load_fp8<S,E,SparseDecodeV41ScopeTraits<T,false>,FRONT_DRAIN>(
                scope,raw_sK,tidx,token_idx,valid);
        }
    } else {
        quant_cache_load_fp8<S,E,SparseDecodeV41ScopeTraits<T,false>,FRONT_DRAIN>(
            scope,raw_sK,tidx,token_idx,valid);
    }
}

template<int S,int E,typename T>
__forceinline__ __device__ void SparseDecodeV41CacheAdapter::async_issue(
        const typename T::Params &params,int tidx,int token_idx,bool valid,int block_idx) {
    static_assert(T::kDecode && T::kMixedFP4);
    if(block_idx*T::kBlockN>=params.main_padded_length) {
        const auto scope=sparse_decode_v41_cache_scope(params,block_idx);
        quant_cache_fp4_async_issue<S,E,SparseDecodeV41ScopeTraits<T,true>>(
            scope,tidx,token_idx,valid);
    }
}
template<int kThreads>
__forceinline__ __device__ bool sparse_decode_v41_row_allvalid_scan32(
        const int *indices,uint32_t storage,uint32_t capacity,uint32_t rank) {
    bool valid=true;
    if((reinterpret_cast<uint64_t>(indices)&15u)==0 && storage%4u==0) {
        const uint32_t words=storage/4u;
        const int4 *vector_indices=reinterpret_cast<const int4*>(indices);
        #pragma unroll 1
        for(uint32_t position=rank;position<words;position+=kThreads) {
            const int4 values=__ldg(vector_indices+position);
            valid&=(uint32_t(values.x)<capacity)&(uint32_t(values.y)<capacity)
                &(uint32_t(values.z)<capacity)&(uint32_t(values.w)<capacity);
        }
    } else {
        #pragma unroll 1
        for(uint32_t position=rank;position<storage;position+=kThreads) {
            const int index=__ldg(indices+position);
            valid&=uint32_t(index)<capacity;
        }
    }
    return valid;
}

template<typename T, bool FullLengths = false>
__forceinline__ __device__ bool sparse_decode_v41_task_allvalid(
        const Flash_fwd_params &p,const SparseDecodeV41TileParams &task) {
    static_assert(!FullLengths || (T::kLengthGuardAllowed && T::kTaskAllValidGuard));
    constexpr int kWarps=T::NUM_THREADS/32;
    constexpr size_t flags_base=(sizeof(typename T::SharedMemoryPlan)+alignof(int)-1)
        /alignof(int)*alignof(int);
    static_assert(T::NUM_THREADS==512 && kWarps==16);
    static_assert(sizeof(typename T::SharedMemoryPlan)==248880);
    static_assert(flags_base+kWarps*sizeof(int)<=V41_FP4_ASYNC_STAGE_BASE);
    static_assert(V41_FP4_ASYNC_STAGE_BASE==248960);
    static_assert(V41_FP4_ASYNC_STAGE_BASE+T::NUM_THREADS*V41_FP4_ASYNC_STAGE_STRIDE
        <=sizeof(typename T::SharedMemoryOutPut));
    extern __shared__ char wksp_buf[];
    int *flags=reinterpret_cast<int*>(wksp_buf+flags_base);
    const int tidx=threadIdx.x;
    const int lane=tidx&31;
    const int warp=tidx/32;
    const int64_t main_capacity=int64_t(p.num_blocks)*p.page_block_size;
    const int64_t extra_capacity=int64_t(p.extra_num_blocks)*p.extra_page_block_size;
    const bool eligible=(FullLengths || (p.topk_len_ptr==nullptr && p.extra_topk_len_ptr==nullptr))
        && task.main_length>0 && task.main_length==task.main_storage_topk
        && task.extra_length>=0 && task.extra_length==task.extra_storage_topk
        && task.main_length%64==0 && task.extra_length%64==0
        && task.main_padded_length==task.main_length
        && int64_t(task.topk)==int64_t(task.main_length)+task.extra_length
        && task.start_block>=0 && task.start_block%2==0
        && task.end_block>task.start_block && task.end_block%2==0
        && int64_t(task.end_block)*T::kBlockN<=task.topk
        && task.kv!=nullptr && task.indices!=nullptr && main_capacity>0
        && (task.extra_length==0 ||
            (task.extra_kv!=nullptr && task.extra_indices!=nullptr && extra_capacity>0));
    if constexpr(FullLengths) {
        if(!eligible) return false;
    }
    int lane_valid=eligible?1:0;
    if(eligible) {
        if(main_capacity<=int64_t(0x7fffffff) &&
           (task.extra_storage_topk==0 || extra_capacity<=int64_t(0x7fffffff))) {
            lane_valid&=sparse_decode_v41_row_allvalid_scan32<T::NUM_THREADS>(task.indices,
                uint32_t(task.main_storage_topk),uint32_t(main_capacity),uint32_t(tidx));
            lane_valid&=sparse_decode_v41_row_allvalid_scan32<T::NUM_THREADS>(task.extra_indices,
                uint32_t(task.extra_storage_topk),uint32_t(extra_capacity),uint32_t(tidx));
        } else {
            #pragma unroll 1
            for(int64_t position=tidx;position<task.main_storage_topk;position+=T::NUM_THREADS) {
                const int index=__ldg(task.indices+position);
                lane_valid&=index>=0 && int64_t(index)<main_capacity;
            }
            #pragma unroll 1
            for(int64_t position=tidx;position<task.extra_storage_topk;position+=T::NUM_THREADS) {
                const int index=__ldg(task.extra_indices+position);
                lane_valid&=index>=0 && int64_t(index)<extra_capacity;
            }
        }
    }
    const int warp_valid=__all_sync(0xffffffffu,lane_valid);
    if(lane==0) flags[warp]=warp_valid;
    __syncthreads();
    const int all_flag_vote=lane<kWarps?flags[lane]:1;
    const int task_valid=__all_sync(0xffffffffu,all_flag_vote);
    return __builtin_ppu_to_uniform_b32(task_valid)!=0;
}
template<typename T>
__forceinline__ __device__ SparseDecodeV41TileParams make_sparse_decode_v41_task(
    const Flash_fwd_params &p,const int4 &schedule,int first_split,int batch,int query) {
    const int main_length=p.topk_len_ptr?__ldg(p.topk_len_ptr+batch):p.topk;
    const int extra_length=p.extra_topk>=0
        ?(p.extra_topk_len_ptr?__ldg(p.extra_topk_len_ptr+batch):p.extra_topk):0;
    const int main_padded=cute::round_up(max(main_length,1),64);
    const int total=main_padded+cute::round_up(extra_length,64);
    const int start=batch==schedule.x?schedule.y/32:0;
    const int end=batch==schedule.z?cute::round_up(schedule.w,64)/32:total/32;
    const bool no_split=start==0 && end==total/32;
    const int split=p.num_splits_ptr[batch]+(batch==schedule.x?first_split:0);

    SparseDecodeV41TileParams task{};
    task.s_q=1; task.s_kv=p.num_blocks*p.page_block_size;
    task.h_q=p.ngroups; task.d_qk=p.d;
    task.topk=total; task.sm_scale_div_log2=p.scale_softmax_log2;
    task.q=static_cast<cutlass::bfloat16_t*>(p.q_ptr)+int64_t(batch)*p.q_batch_stride
        +int64_t(query)*p.q_row_stride;
    task.kv=p.k_ptr;
    task.indices=p.indices_ptr+int64_t(batch)*p.indices_batch_stride+int64_t(query)*p.indices_row_stride;
    task.attn_sink=p.attn_sink_ptr;
    task.stride_q_s_q=p.q_row_stride;
    task.stride_q_h_q=p.q_head_stride;
    task.quant_page_size=p.page_block_size; task.quant_block_stride=p.k_batch_stride;
    task.extra_kv=p.extra_k_ptr;
    task.extra_indices=p.extra_topk>=0
        ?p.extra_indices_ptr+int64_t(batch)*p.extra_indices_batch_stride+int64_t(query)*p.extra_indices_row_stride:nullptr;
    task.extra_s_kv=p.extra_num_blocks*p.extra_page_block_size;
    task.extra_page_size=p.extra_page_block_size; task.extra_block_stride=p.extra_k_batch_stride;
    task.main_length=main_length; task.extra_length=extra_length;
    task.main_storage_topk=p.topk; task.extra_storage_topk=max(p.extra_topk,0);
    task.main_padded_length=main_padded; task.start_block=start; task.end_block=end;
    task.partial_output=!no_split;
    if(no_split) {
        task.out=static_cast<cutlass::bfloat16_t*>(p.o_ptr)+int64_t(batch)*p.o_batch_stride
            +int64_t(query)*p.ngroups*p.o_row_stride;
        task.lse=static_cast<float*>(p.softmax_lse_ptr)+int64_t(batch)*p.seqlen_q+int64_t(query)*p.ngroups;
    } else {
        task.out=static_cast<float*>(p.oaccum_ptr)+int64_t(split)*p.seqlen_q*T::kHeadDimV
            +int64_t(query)*p.ngroups*T::kHeadDimV;
        task.lse=static_cast<float*>(p.softmax_lseaccum_ptr)+int64_t(split)*p.seqlen_q+int64_t(query)*p.ngroups;
    }
    return task;
}

template<typename T, bool FullLengths = false>
__forceinline__ __device__ void run_sparse_decode_v41_task(
        const Flash_fwd_params &p,const int4 &schedule,int first_split,
        int batch,int query,int head_tile) {
    SparseDecodeV41TileParams task = make_sparse_decode_v41_task<T>(
        p,schedule,first_split,batch,query);
    if constexpr(T::kTaskAllValidGuard) {
        const bool task_allvalid=sparse_decode_v41_task_allvalid<T,FullLengths>(p,task);
        if(task_allvalid) {
            sparse_wi_tile_driver<SparseDecodeV41AllValidTaskTraits<T>>(task,head_tile);
        } else {
            if constexpr(FullLengths) {
                using CheckedT=std::conditional_t<T::kMixedFP4,
                    SparseDecodeV41Sq1CheckedOffset32FP4Traits,
                    SparseDecodeV41Sq1CheckedU32FP8Traits>;
                static_assert(std::is_same_v<typename CheckedT::Params,typename T::Params>);
                static_assert(std::is_same_v<typename CheckedT::SharedMemoryPlan,typename T::SharedMemoryPlan>);
                static_assert(std::is_same_v<typename CheckedT::SharedMemoryOutPut,typename T::SharedMemoryOutPut>);
                sparse_wi_tile_driver<CheckedT>(task,head_tile);
            } else {
                sparse_wi_tile_driver<T>(task,head_tile);
            }
        }
    } else {
        sparse_wi_tile_driver<T>(task,head_tile);
    }
    __syncthreads();
}

struct SparseDecodeV41Scheduler {
    template<typename T>
    static __forceinline__ __device__ void run(const Flash_fwd_params &p) {
        const int partition=blockIdx.z;
        const int tile=blockIdx.x;
        const int head_tiles=p.ngroups/T::kBlockM;
        const int query=tile/head_tiles;
        const int head_tile=tile%head_tiles;
        const int *metadata=p.tile_scheduler_metadata_ptr+partition*TileSchedulerMetaDataSize;
        const int4 schedule=*reinterpret_cast<const int4*>(metadata);
        if(schedule.x>=p.b) return;
        const int first_split=metadata[4];

        #pragma unroll 1
        for(int batch=schedule.x;batch<=schedule.z;++batch) {
            run_sparse_decode_v41_task<T>(p,schedule,first_split,batch,query,head_tile);
        }
    }
};

struct SparseDecodeV41DirectScheduler {
    template<typename T>
    static __forceinline__ __device__ void run(const Flash_fwd_params &p) {
        const int64_t ordinal=int64_t(blockIdx.z);
        const int tile=blockIdx.x;
        const int head_tiles=p.ngroups/T::kBlockM;
        const int query=tile/head_tiles;
        const int head_tile=tile%head_tiles;
        int64_t prefix=0;
        int partition=-1,batch=-1;
        int4 schedule{};
        #pragma unroll 1
        for(int candidate=0;candidate<p.num_sm_parts;++candidate) {
            const int *candidate_metadata=p.tile_scheduler_metadata_ptr
                +int64_t(candidate)*TileSchedulerMetaDataSize;
            const int4 candidate_schedule=*reinterpret_cast<const int4*>(candidate_metadata);
            if(candidate_schedule.x>=p.b) continue;
            const int64_t task_count=int64_t(candidate_schedule.z)-candidate_schedule.x+1;
            if(ordinal>=prefix && ordinal-prefix<task_count) {
                partition=candidate;
                batch=candidate_schedule.x+int(ordinal-prefix);
                schedule=candidate_schedule;
                break;
            }
            prefix+=task_count;
        }
        if(partition<0) return;
        const int *metadata=p.tile_scheduler_metadata_ptr
            +int64_t(partition)*TileSchedulerMetaDataSize;
        const int first_split=metadata[4];
        run_sparse_decode_v41_task<T,T::kLengthGuardAllowed>(
            p,schedule,first_split,batch,query,head_tile);
    }
};

struct SparseDecodeV41BoundedPrefixScheduler {
    template<typename T>
    static __forceinline__ __device__ void run(const Flash_fwd_params &p) {
        const int64_t ordinal=int64_t(blockIdx.z);
        const int tile=blockIdx.x;
        const int head_tiles=p.ngroups/T::kBlockM;
        const int query=tile/head_tiles;
        const int head_tile=tile%head_tiles;
        const int64_t split_origin=int64_t(p.num_splits_ptr[0]);
        const int64_t total_tasks=int64_t(p.num_splits_ptr[p.b])-split_origin;
        if(ordinal>=total_tasks) return;
        int partition=-1,batch=-1,first_split=0;
        int4 schedule{};
        if(total_tasks==int64_t(p.b)) {
            batch=int(ordinal);
            schedule={p.b,0,p.b,0};
        } else {
            int lower=0,upper=p.num_sm_parts;
            #pragma unroll 1
            while(lower<upper) {
                const int candidate=lower+(upper-lower)/2;
                const int *candidate_metadata=p.tile_scheduler_metadata_ptr
                    +int64_t(candidate)*TileSchedulerMetaDataSize;
                const int4 candidate_schedule=*reinterpret_cast<const int4*>(candidate_metadata);
                const int64_t candidate_prefix=candidate_schedule.x>=p.b?total_tasks:
                    int64_t(p.num_splits_ptr[candidate_schedule.x])-split_origin
                        +int64_t(candidate_metadata[4]);
                if(candidate_prefix<=ordinal) lower=candidate+1;
                else upper=candidate;
            }
            partition=lower-1;
            if(partition<0) return;
            const int *metadata=p.tile_scheduler_metadata_ptr
                +int64_t(partition)*TileSchedulerMetaDataSize;
            schedule=*reinterpret_cast<const int4*>(metadata);
            if(schedule.x>=p.b) return;
            const int64_t prefix=int64_t(p.num_splits_ptr[schedule.x])-split_origin
                +int64_t(metadata[4]);
            const int64_t task_count=int64_t(schedule.z)-schedule.x+1;
            const int64_t local=ordinal-prefix;
            if(local<0 || local>=task_count) return;
            batch=schedule.x+int(local);
            first_split=metadata[4];
        }
        static_assert(T::kTaskAllValidGuard && T::kLengthGuardAllowed);
        run_sparse_decode_v41_task<T,true>(p,schedule,first_split,batch,query,head_tile);
    }
};

template<typename T>
void launch_sparse_decode_wg_v41_main(Flash_fwd_params &p,hggcStream_t stream) {
    auto kernel=&::flash_sparse_decode_wg_kernel<T>;
    constexpr size_t bytes=std::max(sizeof(typename T::SharedMemoryPlan),sizeof(typename T::SharedMemoryOutPut));
    static_assert(bytes==262144 && sizeof(typename T::SharedMemoryPlan)==248880);
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel,hggcFuncAttributeMaxDynamicSharedMemorySize,bytes));
    const int m_blocks=cute::ceil_div(p.seqlen_q,T::kBlockM);
    hggcLaunchAttribute attributes[1];
    attributes[0].id=hggcLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed=1;
    const int64_t z_blocks=T::kSingleTaskCTA?int64_t(p.b)+p.num_sm_parts-1:p.num_sm_parts;
    hggcLaunchConfig_t config={dim3(m_blocks,p.h,unsigned(z_blocks)),dim3(T::NUM_THREADS,1,1),bytes,stream,attributes,1};
    hggcLaunchKernelEx(&config,kernel,p);
    CHECK_CUDA_KERNEL_LAUNCH();
}

__forceinline__ bool sparse_decode_v41_fp4_guard_shape(const Flash_fwd_params &p) {
    const int extra_storage=p.extra_topk>=0?p.extra_topk:0;
    return p.topk_len_ptr==nullptr && p.extra_topk_len_ptr==nullptr
        && p.topk>0 && p.topk%64==0 && extra_storage%64==0
        && (int64_t(p.topk)+extra_storage>=2048
            || (p.ngroups==128 && p.topk==128 && extra_storage>=128
                && int64_t(p.topk)+extra_storage>=256));
}
__forceinline__ bool sparse_decode_v41_cache_scope_offset32(
        const void *base,int num_blocks,int page_size,int64_t block_stride,
        int payload_bytes,int scale_bytes) {
    constexpr uint64_t limit=uint64_t(1)<<31;
    if(base==nullptr || num_blocks<=0 || page_size<=0 ||
       (page_size&(page_size-1))!=0 || block_stride<=0) return false;
    const uint64_t page=uint64_t(page_size);
    const uint64_t stride=uint64_t(block_stride);
    const uint64_t row_bytes=uint64_t(payload_bytes)+uint64_t(scale_bytes);
    if(page>limit/row_bytes || stride>limit) return false;
    if(uint64_t(num_blocks)>uint64_t(0x7fffffff)/page) return false;
    const uint64_t page_bytes=page*row_bytes;
    const uint64_t last_page=uint64_t(num_blocks-1);
    if(last_page>(limit-page_bytes)/stride) return false;
    const uint64_t max_end=last_page*stride+page_bytes;
    return max_end<=limit;
}

__forceinline__ bool sparse_decode_v41_fp4_cache_offset32(const Flash_fwd_params &p) {
    return sparse_decode_v41_cache_scope_offset32(
            p.k_ptr,p.num_blocks,p.page_block_size,p.k_batch_stride,512,16)
        && sparse_decode_v41_cache_scope_offset32(
            p.extra_k_ptr,p.extra_num_blocks,p.extra_page_block_size,
            p.extra_k_batch_stride,256,32);
}

__forceinline__ bool sparse_decode_v41_direct_grid_gate(const Flash_fwd_params &p) {
    const int64_t tasks=int64_t(p.b)+p.num_sm_parts-1;
    return p.b>0 && p.num_sm_parts>0 && tasks>=1 && tasks<=65535;
}
__forceinline__ bool sparse_decode_v41_fp8_scope_u32_span(
        const void *base,int blocks,int page,int64_t stride) {
    constexpr uint64_t limit=uint64_t(1)<<32;
    if(base==nullptr || blocks<=0 || page<=0 || (page&(page-1))!=0 ||
       stride<=0 || uint64_t(stride)>limit-1) return false;
    const uint64_t page_bytes=uint64_t(page)*(512u+16u);
    if(page_bytes>limit || uint64_t(stride)<page_bytes ||
       int64_t(blocks)*page>2147483647LL) return false;
    return uint64_t(blocks-1)<=(limit-page_bytes)/uint64_t(stride);
}
__forceinline__ bool sparse_decode_v41_fp8_scope_u32_gate(const Flash_fwd_params &p) {
    return sparse_decode_v41_fp8_scope_u32_span(p.k_ptr,p.num_blocks,p.page_block_size,p.k_batch_stride)
        && (p.extra_topk<=0 || sparse_decode_v41_fp8_scope_u32_span(
            p.extra_k_ptr,p.extra_num_blocks,p.extra_page_block_size,p.extra_k_batch_stride));
}

template<bool ExtraFP4>
void launch_sparse_decode_wg_v41(Flash_fwd_params &p,hggcStream_t stream) {
    using PairedT=SparseDecodeV41PairedTraits<ExtraFP4>;
    FLASH_ASSERT(p.topk%64==0);
    FLASH_ASSERT(sparse_decode_wg_v41_q_alignment_supported(
        p.q_ptr, p.q_batch_stride, p.q_row_stride, p.q_head_stride, p.d));
    const int extra_storage=p.extra_topk>=0?p.extra_topk:0;
    const int64_t storage_total=int64_t(p.topk)+extra_storage;
    const bool sq1_lengths=(p.topk_len_ptr!=nullptr || p.extra_topk_len_ptr!=nullptr)
        && p.h==1 && p.ngroups==128 && p.seqlen_q==p.ngroups && p.topk==128
        && extra_storage>=128 && extra_storage%64==0
        && storage_total>=256
        && sparse_decode_v41_direct_grid_gate(p);
    bool checked_address_entry=false;
    if(sq1_lengths) {
        if constexpr(ExtraFP4) checked_address_entry=sparse_decode_v41_fp4_cache_offset32(p);
        else checked_address_entry=sparse_decode_v41_fp8_scope_u32_gate(p);
    }
    const bool length_guard_entry=checked_address_entry && storage_total<=2048;
    if(length_guard_entry) {
        if constexpr(ExtraFP4) {
            launch_sparse_decode_wg_v41_main<SparseDecodeV41Sq1LengthsPrefixFP4Traits>(p,stream);
        } else {
            launch_sparse_decode_wg_v41_main<SparseDecodeV41Sq1LengthsPrefixFP8Traits>(p,stream);
        }
    } else if(checked_address_entry) {
        if constexpr(ExtraFP4) {
            launch_sparse_decode_wg_v41_main<SparseDecodeV41Sq1PersistentOffset32FP4Traits>(p,stream);
        } else {
            launch_sparse_decode_wg_v41_main<SparseDecodeV41Sq1PersistentU32FP8Traits>(p,stream);
        }
    } else if constexpr(ExtraFP4) {
        const bool direct=sparse_decode_v41_direct_grid_gate(p);
        if(sparse_decode_v41_fp4_guard_shape(p)) {
            const bool narrow=sparse_decode_v41_fp4_cache_offset32(p);
            if(direct) {
                if(narrow) {
                    if(p.topk==128 && p.extra_topk>0) {
                        launch_sparse_decode_wg_v41_main<SparseDecodeV41DirectScopeNarrowNativeFP4WG1PublishFirstRemotePVTraits>(p,stream);
                    } else {
                        launch_sparse_decode_wg_v41_main<SparseDecodeV41DirectScopeNarrowGuardedTraits>(p,stream);
                    }
                } else {
                    launch_sparse_decode_wg_v41_main<SparseDecodeV41DirectGuardedPairedTraits<true>>(p,stream);
                }
            } else {
                if(narrow) {
                    launch_sparse_decode_wg_v41_main<SparseDecodeV41ScopeNarrowGuardedTraits>(p,stream);
                } else {
                    launch_sparse_decode_wg_v41_main<SparseDecodeV41GuardedPairedTraits<true>>(p,stream);
                }
            }
        } else {
            if(direct) {
                launch_sparse_decode_wg_v41_main<SparseDecodeV41DirectPairedTraits<true>>(p,stream);
            } else {
                launch_sparse_decode_wg_v41_main<PairedT>(p,stream);
            }
        }
    } else {
        if(sparse_decode_v41_direct_grid_gate(p)) {
            if(sparse_decode_v41_fp4_guard_shape(p)) {
                if(sparse_decode_v41_fp8_scope_u32_gate(p)) {
                    if(p.topk==128 && p.extra_topk>0) {
                        launch_sparse_decode_wg_v41_main<SparseDecodeV41DirectScopeU32FP8WG1PublishBoundWG0PostP0Traits>(p,stream);
                    } else {
                        launch_sparse_decode_wg_v41_main<SparseDecodeV41DirectScopeU32GuardedPairedTraits>(p,stream);
                    }
                } else {
                    launch_sparse_decode_wg_v41_main<SparseDecodeV41DirectGuardedPairedTraits<false>>(p,stream);
                }
            } else {
                launch_sparse_decode_wg_v41_main<SparseDecodeV41DirectPairedTraits<false>>(p,stream);
            }
        } else {
            launch_sparse_decode_wg_v41_main<PairedT>(p,stream);
        }
    }
    run_flash_mla_combine_kernel<typename PairedT::InputT>(p,stream);
}

} // namespace flashmla::dsa::v41


// [SPARSE-WI Stage B] Is_causal template parameter dropped: sparse decode
// expresses causal masking via indices, not via traditional causal mask path.
template<typename T>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
flash_sparse_decode_wg_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
    if constexpr (T::kV41PairedPipeline) {
        static_assert(T::kDecode && T::Arch_value == 89 && T::kUsePairedCubes && T::kHeadDim == 512);
        using Scheduler = typename T::Scheduler;
        Scheduler::template run<T>(params);
    } else {
    // grid shape: [
    // 	num_m_blocks (=ceil_div(seqlen_q_ori*(num_q_heads//num_kv_heads))),
    // 	num_kv_heads,
    // 	num_sm_parts
    // ]
    // An "sm part" is responsible for all the BLOCK_SIZE_M q_heads in the m_block (as specified by m_block_idx), under one kv head (as specified by k_head_idx), of a segment (as specified by [start_block_idx, end_block_idx]) of one request (as specified by batch_idx).
    // If is_no_split is True, then this request is exclusively assigned to this sm_part, so we shall write the result directly into params.o_ptr and params.softmax_lse_ptr. Otherwise, write to oaccum_ptr and softmax_lseaccum_ptr, with the corresponding split idx being (n_split_idx + num_splits_ptr[batch_idx])
    // For the complete schedule of the kernel, please read our deep-dive write-up (link can be found in the README.md file).

    const int m_block_idx = blockIdx.x;
    const int k_head_idx = blockIdx.y;
    // Keep the parent's metadata identity; child parity changes only batch ownership.
    const int partition_idx = blockIdx.z / T::kMetadataStripeFactor;
    const int metadata_stripe_idx = blockIdx.z % T::kMetadataStripeFactor;
    const int warpgroup_idx = __builtin_ppu_to_uniform_b32(threadIdx.x / 256);
    const int idx_in_warpgroup = threadIdx.x % 256;
    const int warp_idx = __builtin_ppu_to_uniform_b32(threadIdx.x / 32);

    const int tidx = threadIdx.x;
    using InputT = typename T::InputT;
    typename T::TiledMma tiled_mma;

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    using SharedMemoryPlan = typename T::SharedMemoryPlan;
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan *>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.smem_sQ.data()), (typename T::SmemLayoutQ){});
    // Use swizzled layout so scalar stores in load_and_dequant_sparse_K
    // are TSM_LD_SWZL-compatible (same pattern as sV_direct using SmemLayoutVDirect).
    Tensor sK = make_tensor(make_smem_ptr(plan.smem_sK.data()), (typename T::SmemLayoutKDirect){});
    // [FP8-WI fix10] For MODEL1, sQ only has 8 tiles (512/64) -- no spare tile 8.
    // Use dedicated smem_sP_model1; for 576-dim (FP8 V3.2 or BF16), reuse sQ tile 8 (consumed into rQ8).
    typename T::InputT *sP_base;
    if constexpr (T::kHasExtraRopeTile) {
        // 576-dim: sQ has 9 tiles (576/64), tile 8 already consumed into rQ8.
        sP_base = flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, Int<T::PAGE_BLOCK_SIZE>>{})(_, _, _0{}, _8{}).data().get();
    } else {
        // MODEL1: sQ has exactly 8 tiles (512/64), no spare tile 8; use dedicated buffer.
        sP_base = plan.smem_sP_model1.data();
    }
    Tensor sP0 = make_tensor(make_smem_ptr(sP_base), (typename T::SmemLayoutP0){});
    Tensor sP1 = make_tensor(sP0.data() + sP0.size(), (typename T::SmemLayoutP0){});
    Tensor sM = make_tensor(make_smem_ptr(plan.smem_sM.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sL_reduction_wksp = make_tensor(make_smem_ptr(plan.sL_reduction_wksp.data()), make_shape(Int<T::BLOCK_SIZE_M + 128>{}));
    Tensor sScale0 = make_tensor(make_smem_ptr(plan.smem_sScale0.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sScale1 = make_tensor(make_smem_ptr(plan.smem_sScale1.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    // Valid indices mask: (4 buffers, kBlockN tokens)
    // Cross-WG: WG0 uses bufs 0/1, WG1 uses bufs 2/3 (alternating preload/softmax)
    Tensor smem_valid_indices = make_tensor(make_smem_ptr(plan.smem_valid_indices.data()),
        Shape<_4, Int<T::kBlockN>>{}, Stride<Int<T::kBlockN>, _1>{});
    // char* sO_addr = (char*)plan.smem_sK0.data();	// Overlap with sK0 and sK1
    char *sO_addr = (char *)plan.smem_sQ.data(); // Overlap with sK0 and sK1

    // // Define TMA stuffs
    __mbarrier_t *barrier_Q = &(plan.barrier_Q);
    __mbarrier_t *barriers_K0 = plan.barriers_K0;
    __mbarrier_t *barriers_K1 = plan.barriers_K1;

    // // Initialize TMA barriers
    static_assert(T::SharedMemoryPlan::kNumKBarriers == 2, "FP8 WI kernel expects exactly 2 K sub-stage barriers");
    wg_init_copy_barriers<32 * (T::kBlockM / T::kBlockMPerLoad)>(barrier_Q, barriers_K0, barriers_K1);
    __syncthreads();  // ensure all threads see initialized mbarriers
    bool cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;

    // // Programmatic Dependent Launch: Wait for the previous kernel to finish
    // cudaGridDependencySynchronize();

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    // We don't use __ldg here, otherwise NVCC (ptxas, in particular) will do instruction reorder and place __ldg (LDG.E.128.CONSTANT in SASS) in front of cudaGridDependencySynchronize() (ACQBULK in SASS), leading to data race.
    int4 tile_scheduler_metadata = *(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int begin_seqlen = tile_scheduler_metadata.y;
    int end_idx = tile_scheduler_metadata.z;
    int end_seqlen = tile_scheduler_metadata.w;

    if (begin_idx >= params.b)
        return;
    int begin_n_split_idx = *(tile_scheduler_metadata_ptr + 4);

    // Descriptor begin/end stay unchanged: they still identify partial K ranges
    // and the original begin_n_split_idx. Only the first owned batch is shifted.
    const int first_batch_idx = begin_idx + metadata_stripe_idx;
    if constexpr (T::kMetadataStripeFactor > 1) {
        // CTA-uniform exit before any Q load for an empty child stripe.
        if (first_batch_idx > end_idx || first_batch_idx >= params.b) return;
    }
    // Copy the first owned Q.
    launch_q_copy<T>(params, first_batch_idx, m_block_idx, k_head_idx, sQ, tidx, warp_idx, barrier_Q);

#pragma unroll 1
#pragma clang loop licm(disable)
    for (int batch_idx = first_batch_idx; batch_idx <= end_idx;
         batch_idx += T::kMetadataStripeFactor) {
        constexpr int kBlockN = T::kBlockN;
        const int n_split_idx = batch_idx == begin_idx ? begin_n_split_idx : 0;
        int seqlen_k;
        if (params.topk_len_ptr) {
            seqlen_k = __ldg(params.topk_len_ptr + batch_idx);
        } else {
            seqlen_k = params.topk;
        }
        int seqlen_kpad = max(seqlen_k, 1);
        // Always round main cache up to kBlockN*2 (64) so that:
        //   - the main/extra cache boundary is on an even block, keeping
        //     the mainloop's main/extra switch simple;
        //   - it matches the metadata kernel's block_size_n=64 tiling, so
        //     is_no_split (start==0 && end==total/kBlockN) stays accurate
        //     without an extra __ldg on num_splits_ptr.
        seqlen_kpad = cute::round_up(seqlen_kpad, kBlockN * 2);
        int extra_seqlen_k = 0;
        int ori_block_max = -1;
        if (params.extra_topk >= 0) {
            if (params.extra_topk_len_ptr) {
                extra_seqlen_k = __ldg(params.extra_topk_len_ptr + batch_idx);
            } else {
                extra_seqlen_k = params.extra_topk;
            }
            ori_block_max = cute::ceil_div(seqlen_kpad, kBlockN);
        }
        const int total_k = seqlen_kpad + cute::round_up(extra_seqlen_k, kBlockN * 2);
        const int start_block_idx = batch_idx == begin_idx ? begin_seqlen / kBlockN : 0;
        int end_block_idx = batch_idx == end_idx
            ? cute::round_up(end_seqlen, kBlockN * 2) / kBlockN
            : total_k / kBlockN;
        const bool is_no_split = start_block_idx == 0 && end_block_idx == (total_k / kBlockN);

        // Pre-fetch pipeline: each WG has TWO pre_token_idx values.
        // pre_token_idx0: for block0/nxt_block0 (even block)
        // pre_token_idx1: for block1/nxt_block1 (odd block)
        // Prolog initializes them, and each load call updates its own idx.
        int pre_token_idx0 = -1;   // for block0/nxt_block0 (even block)
        int pre_token_idx1 = -1;   // for block1/nxt_block1 (odd block)

        // Explicit __ldg pre-fetch for the FIRST round (block 0 for wg0, block 1 for wg1).
        // This __ldg is issued before load_and_dequant_sparse_K, so its latency
        // is hidden by the subsequent tensor setup and cp.async inside the function.
        if constexpr (!T::IsFP8) {
            int _token_id = idx_in_warpgroup / 8;
            // Helper lambda to prefetch token index for a given block
            auto prefetch_tok = [&](int _blk) -> int {
                if (_blk < 0 || _blk >= end_block_idx) return -1;
                bool _use_extra = (ori_block_max >= 0) && (_blk >= ori_block_max);
                int _eff = _use_extra ? (_blk - ori_block_max) : _blk;
                const int *_base = _use_extra
                    ? (wg_query_indices_bf16<true, T>(params) + static_cast<int64_t>(batch_idx) * params.extra_indices_batch_stride + _eff * T::kBlockN)
                    : (wg_query_indices_bf16<false, T>(params) + static_cast<int64_t>(batch_idx) * params.indices_batch_stride + _eff * T::kBlockN);
                return __ldg(_base + _token_id);
            };
            if (warpgroup_idx == 0) {
                // WG0: prolog loads block0 (uses idx0), main loop first loads nxt_block0=block2, nxt_block1=block3
                // idx0: prefetch block0 (prolog), will be updated to block2 by prolog load
                // idx1: prefetch block3 (first main loop nxt_block1)
                pre_token_idx0 = prefetch_tok(start_block_idx);
                pre_token_idx1 = prefetch_tok(start_block_idx + 3);
            } else {
                // WG1: prolog loads block1 (uses idx1), main loop first loads nxt_block1=block3, nxt_block0=block2
                // idx1: prefetch block1 (prolog), will be updated to block3 by prolog load
                // idx0: prefetch block2 (first main loop nxt_block0)
                pre_token_idx1 = prefetch_tok(start_block_idx + 1);
                pre_token_idx0 = prefetch_tok(start_block_idx + 2);
            }
        }

        // [SPARSE-WI Stage B] causal branch removed -- sparse path always treats
        // every kv token in [0, seqlen_k) as candidate; mask is driven by
        // smem_valid_indices in wg{0,1}_bunch_0.

        // [FP8-WI fix6] block_table is nullptr for sparse decode;
        // gK/tKgK/tKsK0/tKsK1/gmem_tiled_copy_K are unused in BF16 path.
        // Create dummy values only for FP8 path API compatibility.
        int* block_table_ptr = nullptr;
        typename T::GmemTiledCopyKV gmem_tiled_copy_K{};
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);
        Tensor gK_dummy = make_tensor(make_gmem_ptr(reinterpret_cast<InputT*>(params.k_ptr)),
                        Shape<Int<kBlockN>, Int<T::kHeadDim>>{},
                        make_stride(params.k_row_stride, _1{}));
        Tensor tKgK = gmem_thr_copy_K.partition_S(make_mix_tensor_like(gK_dummy));

        Tensor cur_sK0 = sK(_, _, 0);
        Tensor cur_sK1 = sK(_, _, 1);
        Tensor nxt_sK0 = sK(_, _, 2);

        // sV_direct: needed by FP8 load_and_dequant_sparse_K; BF16 path uses get_half_V from sK directly.
        Tensor sV_direct = make_tensor(make_smem_ptr(reinterpret_cast<InputT*>(&sK(0, 0, 0))),
                                        (typename T::SmemLayoutVDirect){});
        // [SPARSE-WI Stage C-step2b-2 / FP8-WI fix3] FP8 sparse cutover:
        // load entire blocks via register-direct global -> dequant -> sK
        // pipeline (no SMEM staging).  Each block's ALL 576 dims are written
        // to a SINGLE sK buf slot.  The GEMM consumer reads tiles 0-8 from
        // the same buf, and V GEMM also reads dims 0-511 from one buf.
        // Mapping:
        //   block 0 -> all 576 dims in buf 0
        //   block 1 -> all 576 dims in buf 1

        // [FP8-WI fix9] Zero-fill removed: sK is naturally covered by K load.
        // load_K_tiles_bf16 writes zeros for invalid tiles, and 3-buffer rotation
        // ensures every read buffer was loaded first.
        // {
        //     using InputT_local = typename T::InputT;
        //     InputT_local* raw_sK = reinterpret_cast<InputT_local*>(&sK(0, 0, 0));
        //     constexpr int sK_total_elems = cute::cosize_v<typename T::SmemLayoutK>;
        //     for (int i = threadIdx.x; i < sK_total_elems; i += T::NUM_THREADS) {
        //         raw_sK[i] = InputT_local(0.0f);
        //     }
        //     __syncthreads();
        // }

        // Cross-WG prolog: each block's K split across two sK buffers.
        // Block 0: tiles 0-3 → buf 0 (cur_sK0), tiles 4-8 → buf 1 (cur_sK1)
        // Block 1: tiles 0-3 → buf 1 (cur_sK1), tiles 4-8 → buf 0 (cur_sK0)
        // smem_valid_indices uses 4 bufs: WG0 bufs 0/1, WG1 bufs 2/3 (alternating)
        // Prolog blocks become "current" in first iteration, so use softmax bufs.
        int prolog_vi_wg0 = (start_block_idx / 2) % 2;       // 0 or 1
        int prolog_vi_wg1 = 2 + (start_block_idx / 2) % 2;    // 2 or 3
        constexpr int kNumTiles = T::kHeadDim / T::kBlockKSmem;
        {
            if (warpgroup_idx == 0) {
                if (start_block_idx < end_block_idx) {
                    // wg0 loads block 0: stage 1 (tiles 0-3 → buf 0) + stage 2 (tiles 4-8 → buf 1)
                    if constexpr (!T::IsFP8) {
                        auto vi_wg0 = smem_valid_indices(prolog_vi_wg0, _);
                        load_K_tiles_bf16<4, kNumTiles, T, false>(params, cur_sK1, batch_idx, start_block_idx, seqlen_k, &barriers_K0[1],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg0, 
                            &pre_token_idx0, start_block_idx + 2, end_block_idx, extra_seqlen_k);
                        load_K_tiles_bf16<0, 4, T>(params, cur_sK0, batch_idx, start_block_idx, seqlen_k, &barriers_K0[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg0, 
                            &pre_token_idx0, start_block_idx + 2, end_block_idx, extra_seqlen_k);
                    } else {
                        load_and_dequant_sparse_K_staged<4, kNumTiles, T>(
                            params, sK, batch_idx, start_block_idx, seqlen_k,
                            /*buf_idx=*/1, &barriers_K0[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg0);
                        load_and_dequant_sparse_K_staged<0, 4, T>(
                            params, sK, batch_idx, start_block_idx, seqlen_k,
                            /*buf_idx=*/0, &barriers_K0[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg0);
                    }
                }
            } else {
                if (start_block_idx+1 < end_block_idx) {
                    // wg1 loads block 1: stage 1 (tiles 0-3 → buf 1) + stage 2 (tiles 4-8 → buf 0)
                    if constexpr (!T::IsFP8) {
                        auto vi_wg1 = smem_valid_indices(prolog_vi_wg1, _);
                        load_K_tiles_bf16<4, kNumTiles, T, false>(params, cur_sK0, batch_idx, start_block_idx + 1, seqlen_k, &barriers_K1[1],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg1, 
                            &pre_token_idx1, start_block_idx + 3, end_block_idx, extra_seqlen_k);
                        load_K_tiles_bf16<0, 4, T>(params, cur_sK1, batch_idx, start_block_idx + 1, seqlen_k, &barriers_K1[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg1, 
                            &pre_token_idx1, start_block_idx + 3, end_block_idx, extra_seqlen_k);
                    } else {
                        load_and_dequant_sparse_K_staged<4, kNumTiles, T>(
                            params, sK, batch_idx, start_block_idx + 1, seqlen_k,
                            /*buf_idx=*/0, &barriers_K1[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg1);
                        load_and_dequant_sparse_K_staged<0, 4, T>(
                            params, sK, batch_idx, start_block_idx + 1, seqlen_k,
                            /*buf_idx=*/1, &barriers_K1[0],
                            idx_in_warpgroup, ori_block_max, smem_valid_indices, prolog_vi_wg1);
                    }
                }
            }
        }

        // CTA-wide barrier after prolog loads.
        // Each warpgroup's load_and_dequant_sparse_K uses a warpgroup-internal
        // bar.sync (256 threads) which only ensures intra-wg SMEM visibility.
        // The remote PV GEMM later reads V from the OTHER wg's K buffer.
        // Without this __syncthreads(), cross-wg SMEM visibility is NOT
        // guaranteed on PPU (__threadfence_block() is a no-op).  This causes
        // non-deterministic precision collapse for d_qk=512 + h_q=128 where
        // tight timing (fewer QKT tiles) + all rows active exposes the race.
        __syncthreads();

        Tensor rO = partition_fragment_C((typename T::TiledMmaPV){}, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV / 2>>{});
        float rL[2];
        rL[0] = rL[1] = 0.0f;

        // Clear buffers
        cute::fill(rO, 0.);
        if (threadIdx.x < size(sM)) {
            sM[threadIdx.x] = MAX_INIT_VAL_SM;
        }

        while(!cutlass::arch::test_wait(barrier_Q, cur_phase_Q, 1)) {
            kernel_sleep_ns();
        }
        cur_phase_Q = (cur_phase_Q + 1) & 1;


        Tensor rQ8 = make_tensor<InputT>(Shape<Shape<_2, _2, _2>, _1, _4>{});
        if constexpr (T::kHasExtraRopeTile && !T::kIsCrossCut) {
            retrieve_rP_from_sP<T>(rQ8, local_tile(sQ, Shape<Int<T::kBlockM>, _64>{}, Coord<_0, _8>{}), idx_in_warpgroup);
        } else {
            cute::clear(rQ8);
        }

        if (warpgroup_idx == 0) {
            // Warpgroup 0
            // Tensor rP0 = make_tensor<float>((typename T::rP0Layout){});
            Tensor rP0 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});  // MMA, MMA_M, MMA_K
            const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

            // NOTE We don't use the pipelined version of Q K^T here since it leads
            // to a slow-down (or even register spilling, thanks to the great NVCC)
            // Issue P0 = Q @ K0^T, wait
            // Guard on block range, not seqlen_k.  When
            // seqlen_k=0 but extra blocks exist, block 0 is a valid extra
            // block that must participate in QK GEMM.
            if (start_block_idx < end_block_idx) {

                cute::clear(rP0);
                warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK0, cur_sK1, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup, wg_idx);
            }

            int idx = 0;

            // Precomputed K addresses for SIMT/TC overlap
            WgKAddr<T> precomp_ptr0{};
            WgKAddr<T> precomp_ptr1{};
            bool precomp_valid0 = false;
            bool precomp_valid1 = false;

            // Precompute initial addresses for first iteration
            if constexpr (!T::IsFP8) {
                int first_vi_preload_wg0 = 1 - (start_block_idx / 2) % 2;
                int first_vi_preload_wg1 = 3 - (start_block_idx / 2) % 2;
                bool init_load_extra = (ori_block_max >= 0) && (start_block_idx + 2 >= ori_block_max);
                precomp_ptr0 = compute_K_addr_bf16_dynamic<0, T>(
                    params, batch_idx, start_block_idx + 2, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, first_vi_preload_wg0,
                    &pre_token_idx0, extra_seqlen_k, precomp_valid0, init_load_extra);
                precomp_ptr1 = compute_K_addr_bf16_dynamic<0, T>(
                    params, batch_idx, start_block_idx + 3, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, first_vi_preload_wg1,
                    &pre_token_idx1, extra_seqlen_k, precomp_valid1, init_load_extra);
            }

            #define LAUNCH_WG0_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA)    \
                wg0_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA>(                \
                gmem_tiled_copy_K, tKgK, sQ, sK, sV_direct, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
                rP0, rO, rL,                                     \
                barriers_K0, barriers_K1, cur_phase_K0, params,                       \
                block_table_ptr, seqlen_k, total_k, block_idx, end_block_idx, idx_in_warpgroup, wg_idx, idx, \
                batch_idx, ori_block_max, smem_valid_indices,                         \
                T::IsFP8 ? nullptr : &pre_token_idx0,                                \
                T::IsFP8 ? nullptr : &pre_token_idx1,                                \
                extra_seqlen_k,                                                      \
                precomp_ptr0, precomp_ptr1, precomp_valid0, precomp_valid1);          \

            int block_idx = start_block_idx;
            // 3-phase loop: NEXT_EXTRA controls prefetch/compute_addr cache selection
            // Phase 1: prefetch/compute for primary cache (NEXT_EXTRA=false)
            int p1_end = min(ori_block_max >= 0 ? ori_block_max - 4 : end_block_idx - 2, end_block_idx - 2);
            #pragma unroll 1
            for (; block_idx < p1_end; block_idx += 2) {
                LAUNCH_WG0_SUBROUTINE(false, false, false);
            }
            // Phase 2: prefetch/compute for extra cache (NEXT_EXTRA=true)
            #pragma unroll 1
            for (; block_idx < end_block_idx-2; block_idx += 2) {
                LAUNCH_WG0_SUBROUTINE(false, false, true);
            }
            // Phase 3: last 2 blocks (no K load/prefetch/compute_addr)
            LAUNCH_WG0_SUBROUTINE(false, true, false);
        } else {
            // // Warpgroup 1
            // Tensor rP1 = make_tensor<float>((typename T::rP0Layout){});
            Tensor rP1 = partition_fragment_C(tiled_mma, Shape<Int<T::BLOCK_SIZE_M>, Int<T::kBlockN>>{});  // MMA, MMA_M, MMA_K
            const int wg_idx = __builtin_ppu_to_uniform_b32(idx_in_warpgroup / 32);

            if (start_block_idx+1 < end_block_idx) {
                warpgroup_cooperative_qkt_gemm<T, 1>(sQ, cur_sK1, cur_sK0, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup, wg_idx);
            } else {
                // [FP8-WI fix9] When wg1 has no initial K block, rP1 must be zero-
                // initialized.  Otherwise wg1_subroutine runs softmax on garbage
                // registers, producing corrupt rL that contaminates the final output
                // via the cross-warpgroup rL reduction.
                cute::clear(rP1);
            }

            int idx = 0;

            // Precomputed K addresses for SIMT/TC overlap (WG1)
            WgKAddr<T> precomp_ptr0_wg1{};
            WgKAddr<T> precomp_ptr1_wg1{};
            bool precomp_valid0_wg1 = false;
            bool precomp_valid1_wg1 = false;

            // Precompute initial addresses for first iteration
            if constexpr (!T::IsFP8) {
                bool init_load_extra = (ori_block_max >= 0) && (start_block_idx + 3 >= ori_block_max);
                precomp_ptr0_wg1 = compute_K_addr_bf16_dynamic<4, T>(
                    params, batch_idx, start_block_idx + 3, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                    &pre_token_idx1, extra_seqlen_k, precomp_valid0_wg1, init_load_extra);
                precomp_ptr1_wg1 = compute_K_addr_bf16_dynamic<4, T>(
                    params, batch_idx, start_block_idx + 2, seqlen_k,
                    idx_in_warpgroup, ori_block_max, smem_valid_indices, 0,
                    &pre_token_idx0, extra_seqlen_k, precomp_valid1_wg1, init_load_extra);
                if constexpr (T::kBlockM == 128 && T::kUsePv4x2 &&
                              T::CvtGemmSwzlLd &&
                              std::is_same_v<typename T::InputT, cutlass::bfloat16_t>) {
                    if (start_block_idx < end_block_idx - 2) {
                        // The third bank has no previous reader on bootstrap.
                        // Leave its commit and index prefetch in the first pair.
                        constexpr int kNumTiles = T::kHeadDim / T::kBlockKSmem;
                        issue_K_load_bf16<4, kNumTiles, false, T, false, true, false>(params, nxt_sK0, batch_idx, start_block_idx + 3, seqlen_k, &barriers_K1[1],
                            idx_in_warpgroup, ori_block_max, precomp_ptr0_wg1, precomp_valid0_wg1,
                            &pre_token_idx1, start_block_idx + 5, end_block_idx, extra_seqlen_k);
                    }
                }
            }

            #define LAUNCH_WG1_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA)  \
                wg1_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST, NEXT_EXTRA>(          \
                gmem_tiled_copy_K, tKgK, sQ, sK, sV_direct, cur_sK0, cur_sK1, nxt_sK0, sP0, sP1, sM, sScale0, sScale1, rQ8, \
                rP1, rO, rL,                                     \
                barriers_K0, barriers_K1, cur_phase_K1, params,                       \
                block_table_ptr, seqlen_k, total_k, block_idx, end_block_idx, idx_in_warpgroup, wg_idx, idx, \
                batch_idx, ori_block_max, smem_valid_indices,                         \
                T::IsFP8 ? nullptr : &pre_token_idx1,                                \
                T::IsFP8 ? nullptr : &pre_token_idx0,                                \
                extra_seqlen_k,                                                      \
                precomp_ptr0_wg1, precomp_ptr1_wg1, precomp_valid0_wg1, precomp_valid1_wg1);  \

            int block_idx = start_block_idx;
            // 3-phase loop: same phase boundaries as WG0
            // Phase 1: prefetch/compute for primary cache (NEXT_EXTRA=false)
            int p1_end = min(ori_block_max >= 0 ? ori_block_max - 4 : end_block_idx - 2, end_block_idx - 2);
            #pragma unroll 1
            for (; block_idx < p1_end; block_idx += 2) {
                LAUNCH_WG1_SUBROUTINE(false, false, false);
            }
            // Phase 2: prefetch/compute for extra cache (NEXT_EXTRA=true)
            #pragma unroll 1
            for (; block_idx < end_block_idx-2; block_idx += 2) {
                LAUNCH_WG1_SUBROUTINE(false, false, true);
            }
            // Phase 3: last 2 blocks
            LAUNCH_WG1_SUBROUTINE(false, true, false);
        }

        // Reduce rL across threads within the same warp
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

        // Reduce rL across warpgroups
        int my_row = get_AorC_row_idx<T::kAtomLayoutM>(0, idx_in_warpgroup);
        // Pre-issue attn_sink __ldg before syncthreads to hide GMEM latency
        float pre_sink0 = 0.0f, pre_sink1 = 0.0f;
        if (is_no_split) {
            if (params.attn_sink_ptr != nullptr) {
                const int row0 = my_row;
                const int row1 = my_row + 8;
                const int q_head_idx_0 = (m_block_idx * T::BLOCK_SIZE_M + row0) % params.ngroups;
                const int q_head_idx_1 = (m_block_idx * T::BLOCK_SIZE_M + row1) % params.ngroups;
                pre_sink0 = __ldg(params.attn_sink_ptr + q_head_idx_0);
                pre_sink1 = __ldg(params.attn_sink_ptr + q_head_idx_1);
            }
        }
        wg_reduce_softmax_sums<T>(rL, sL_reduction_wksp, my_row, warpgroup_idx, idx_in_warpgroup);

        // [Task #105] CTA-wide barrier for sM visibility before attn_sink.
        // For BlockM=128, the rL reduction above doesn't include a CTA-wide
        // barrier (WG0 uses register add, WG1 uses __syncwarp).  This
        // __syncthreads() guarantees sM coherence for the attn_sink epilogue.
        // For BlockM=64, the second __syncthreads() in the rL reduction already
        // serves this purpose, so we skip it here.
        if constexpr (T::kBlockM != 64) {
            __syncthreads();
        }

        // Softmax keeps its original two-row QK ownership; PV has four rows.
        float pv_norm[4];
        if constexpr (T::kUsePv4x2) {
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < 4; ++mi) {
                float sum = sL_reduction_wksp[pv_row_idx<T>(mi, idx_in_warpgroup)];
                pv_norm[mi] = (sum == 0.0f || sum != sum) ? 1.0f : sum;
            }
        }
        float *output_norm = T::kUsePv4x2 ? pv_norm : rL;

        // Epilogue
        int num_valid_seq_q = min(params.seqlen_q - m_block_idx * T::BLOCK_SIZE_M, T::BLOCK_SIZE_M);
        if (is_no_split) {
            InputT *o_ptr = (InputT *)params.o_ptr + batch_idx * params.o_batch_stride + m_block_idx * T::BLOCK_SIZE_M * params.o_row_stride + k_head_idx * params.o_head_stride; // (BLOCK_SIZE_M, HEAD_DIM_V) : (params.o_row_stride, 1)
            float *softmax_lse_ptr = (float *)params.softmax_lse_ptr + (batch_idx * params.h + k_head_idx) * params.seqlen_q + m_block_idx * T::BLOCK_SIZE_M;                     // (BLOCK_SIZE_M) : (1)

            Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                                                              Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV>>{},
                                                              make_stride(params.o_row_stride, _1{})));
            Tensor gSoftmaxLse = make_tensor(make_gmem_ptr(softmax_lse_ptr), Layout<
                                                                                 Shape<Int<T::BLOCK_SIZE_M>>,
                                                                                 Stride<_1>>{});

            // attn_sink: SM90-style combined scale (don't modify rL, preserve rO/rL error cancellation)
            if (params.attn_sink_ptr != nullptr) {
                if constexpr (T::kUsePv4x2) {
                    auto rc = make_tensor(rO.data(), pv_rowcol_layout(rO.layout()));
                    CUTLASS_PRAGMA_UNROLL
                    for (int mi = 0; mi < 4; ++mi) {
                        int row = pv_row_idx<T>(mi, idx_in_warpgroup);
                        int head = (m_block_idx * T::BLOCK_SIZE_M + row) % params.ngroups;
                        float sink_exp = expf(__ldg(params.attn_sink_ptr + head) - sM(row) * (float)M_LN2);
                        float scale = __fdividef(1.0f, pv_norm[mi] + sink_exp);
                        CUTLASS_PRAGMA_UNROLL
                        for (int ni = 0; ni < size<1>(rc); ++ni) rc(mi, ni) *= scale;
                        pv_norm[mi] = 1.0f;
                    }
                } else {
                const int row0 = my_row;
                const int row1 = my_row + 8;
                // Use pre-issued __ldg values (latency hidden by __syncthreads above)
                const float sink_exp0 = expf(pre_sink0 - sM(row0) * (float)M_LN2);
                const float sink_exp1 = expf(pre_sink1 - sM(row1) * (float)M_LN2);
                if constexpr (T::kBlockM == 64) {
                    // BlockM=64: Read original pre-prune rL to detect empty attention (all tokens masked).
                    const float orig_rL0 = sL_reduction_wksp[my_row];
                    const float orig_rL1 = sL_reduction_wksp[my_row + 8];
                    const bool row0_empty = (orig_rL0 == 0.0f || orig_rL0 != orig_rL0);
                    const bool row1_empty = (orig_rL1 == 0.0f || orig_rL1 != orig_rL1);
                    const float o_scale0 = row0_empty ? 0.0f : __fdividef(1.0f, rL[0] + sink_exp0);
                    const float o_scale1 = row1_empty ? 0.0f : __fdividef(1.0f, rL[1] + sink_exp1);
                    CUTLASS_PRAGMA_UNROLL
                    for (int idx = 0; idx < size(rO); ++idx) {
#if ACOMPUTE_VERSION == 10000
                        bool is_row0 = ((idx / 4) % 2 == 0);
#else
                        bool is_row0 = (idx % 4 < 2);
#endif
                        bool empty = is_row0 ? row0_empty : row1_empty;
                        float scale = is_row0 ? o_scale0 : o_scale1;
                        rO(idx) = empty ? 0.0f : (rO(idx) * scale);
                    }
                } else {
                    // BlockM=128: optimized path (rL already pruned, 0->1.0)
                    const float o_scale0 = rL[0] == 0.0f ? 0.0f : __fdividef(1.0f, rL[0] + sink_exp0);
                    const float o_scale1 = rL[1] == 0.0f ? 0.0f : __fdividef(1.0f, rL[1] + sink_exp1);
                    CUTLASS_PRAGMA_UNROLL
                    for (int idx = 0; idx < size(rO); ++idx) {
#if ACOMPUTE_VERSION == 10000
                        bool is_row0 = ((idx / 4) % 2 == 0);
#else
                        bool is_row0 = (idx % 4 < 2);
#endif
                        float scale = is_row0 ? o_scale0 : o_scale1;
                        rO(idx) *= scale;
                    }
                }
                // Set rL to 1.0 so store_o's division becomes a no-op
                rL[0] = 1.0f;
                rL[1] = 1.0f;
                }
            }

            if constexpr (T::kBlockM == 64) {
                CUTLASS_PRAGMA_UNROLL
                for (int idx = 0; idx < size(rO); ++idx) {
                    if (rO(idx) != rO(idx)) rO(idx) = 0.0f;
                }
            }

            store_o<T, true>(rO, gO, output_norm, sO_addr, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

            int i = threadIdx.x;
            if (i < num_valid_seq_q) {
                float cur_L = sL_reduction_wksp[i];
                float sM_val = sM(i);
                if constexpr (T::kBlockM == 64) {
                    bool sM_nan = ((__float_as_uint(sM_val) & 0x7FFFFFFFu) > 0x7F800000u);
                    gSoftmaxLse(i) = (cur_L == 0.0f || cur_L != cur_L || sM_nan) ? INFINITY : logf(cur_L) + sM_val / (float)M_LOG2E;
                } else {
                    gSoftmaxLse(i) = (cur_L == 0.0f || cur_L != cur_L) ? INFINITY : logf(cur_L) + sM_val / (float)M_LOG2E;
                }
            }

            if (batch_idx + T::kMetadataStripeFactor <= end_idx) {
                // Skip mbarrier reinit: barrier phases are consistent across batches.
                // store_o<T, true> writes directly to GMEM, no sO_addr SMEM race.
                launch_q_copy<T>(params, batch_idx + T::kMetadataStripeFactor, m_block_idx, k_head_idx, sQ, tidx, warp_idx, barrier_Q);
            } else {
                // Allow the next kernel (the combine kernel) to launch
                // The next kernel MUST be the combine kernel
            }
        } else {
            // Don't use __ldg because of PDL and instruction reordering
            int split_idx = params.num_splits_ptr[batch_idx] + n_split_idx;

            float *oaccum_ptr = (float *)params.oaccum_ptr + ((split_idx * params.h + k_head_idx) * params.seqlen_q + m_block_idx * T::BLOCK_SIZE_M) * T::kHeadDimV;    // (BLOCK_SIZE_M, HEAD_DIM_V) : (HEAD_DIM_V, 1)
            float *softmax_lseaccum_ptr = (float *)params.softmax_lseaccum_ptr + (split_idx * params.h + k_head_idx) * params.seqlen_q + m_block_idx * T::BLOCK_SIZE_M; // (BLOCK_SIZE_M) : (1)
            Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), Layout<
                                                                        Shape<Int<T::BLOCK_SIZE_M>, Int<T::kHeadDimV>>,
                                                                        Stride<Int<T::kHeadDimV>, _1>>{});
            Tensor gSoftmaxLseAccum = make_tensor(make_gmem_ptr(softmax_lseaccum_ptr), Layout<
                                                                                           Shape<Int<T::BLOCK_SIZE_M>>,
                                                                                           Stride<_1>>{});

            int i = threadIdx.x;
            if (i < num_valid_seq_q) {
                float cur_L = sL_reduction_wksp[i];
                float sM_val = sM(i);
                if constexpr (T::kBlockM == 64) {
                    bool sM_nan = ((__float_as_uint(sM_val) & 0x7FFFFFFFu) > 0x7F800000u);
                    gSoftmaxLseAccum(i) = (cur_L == 0.0f || cur_L != cur_L || sM_nan) ? -INFINITY : log2f(cur_L) + sM_val;
                } else {
                    gSoftmaxLseAccum(i) = (cur_L == 0.0f || cur_L != cur_L) ? -INFINITY : log2f(cur_L) + sM_val;
                }
            }

            if constexpr (T::kBlockM == 64) {
                CUTLASS_PRAGMA_UNROLL
                for (int idx = 0; idx < size(rO); ++idx) {
                    if (rO(idx) != rO(idx)) rO(idx) = 0.0f;
                }
            }

            store_o<T, false>(rO, gOAccum, output_norm, sO_addr, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

            if (batch_idx + T::kMetadataStripeFactor <= end_idx) {
                // Keep mbarrier reinit for split path (store_o<T,false> uses sO_addr)
                wg_init_copy_barriers<32 * (T::kBlockM / T::kBlockMPerLoad)>(barrier_Q, barriers_K0, barriers_K1);
                __syncthreads();
                cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;
                launch_q_copy<T>(params, batch_idx + T::kMetadataStripeFactor, m_block_idx, k_head_idx, sQ, tidx, warp_idx, barrier_Q);
            } else {
                // Allow the next kernel (the combine kernel) to launch
            }
        }
        if constexpr (T::kMetadataStripeFactor > 1) {
            // A child's last batch may be end_idx-1. Do not prefetch or reset
            // phases for the other child, or retain an unmatched async Q copy.
            if (batch_idx + T::kMetadataStripeFactor <= end_idx)
                __syncthreads();
        } else {
            // Keep the original generic image's exact terminal condition.
            if (batch_idx != end_idx)
                __syncthreads();
        }
    }
    }
}

template <typename InputT, int Arch, bool IsFP8, int BlockM,
          bool IsV41FP8, bool IsExtraFP4>
void run_flash_sparse_decode_wg_kernel(Flash_fwd_mla_params &params, hggcStream_t stream)
{
    if constexpr (Arch == 89 && IsFP8 && IsV41FP8 && BlockM == 128 &&
                  std::is_same_v<InputT, cutlass::bfloat16_t>) {
        if (params.ngroups % 128 == 0) {
            flashmla::dsa::v41::launch_sparse_decode_wg_v41<IsExtraFP4>(params, stream);
            return;
        }
    }
    if constexpr (!IsFP8) {
        FLASH_ASSERT(params.ngroups > 0 && params.ngroups % BlockM == 0);
    } else {
        FLASH_ASSERT(params.ngroups > 0);
        if (params.seqlen_q > params.ngroups) {
            FLASH_ASSERT(params.ngroups % BlockM == 0);
        }
    }
    // Correct the internal launch geometry without changing API metadata,
    // num_splits, scratch capacity, or params.num_sm_parts. Host SDK version
    // 10000 differs from device10500, so use explicit type/Arch scope here.
    bool use_metadata_stripe2 = false;
    if constexpr (Arch == 89 && !IsFP8 && BlockM == 128 &&
                  std::is_same_v<InputT, cutlass::bfloat16_t>) {
        const int64_t physical_parts = int64_t(params.num_sm_parts) * 2;
        const int64_t kv_tokens = int64_t(params.topk) + std::max(params.extra_topk, 0);
        if (params.ngroups == 128 && params.h == 1 && params.q_orig == 1 &&
            params.seqlen_q == 128 && params.num_sm_parts > 0 &&
            int64_t(params.b) >= physical_parts &&
            int64_t(params.b) * kv_tokens > physical_parts * 256) {
            const auto *dprops = at::cuda::getCurrentDeviceProperties();
            int sm_count = dprops->multiProcessorCount;
            if (std::string(dprops->name).find("810E") != std::string::npos) sm_count = 20;
            use_metadata_stripe2 = int64_t(sm_count) == physical_parts;
        }
    }
    bool use_compact_k_addr = false;
    if constexpr (Arch == 89 && !IsFP8 && BlockM == 128 &&
                  std::is_same_v<InputT, cutlass::bfloat16_t>) {
        const int64_t launched_parts = int64_t(params.num_sm_parts) *
                                       (use_metadata_stripe2 ? 2 : 1);
        // Amortize pointer reconstruction over at least 128 N32 tiles per CTA.
        if (launched_parts > 0 && int64_t(params.b) * params.topk > launched_parts * 4096 &&
            params.extra_topk <= 0 && params.extra_k_ptr == nullptr &&
            params.num_blocks > 0 && params.page_block_size > 0 &&
            params.k_batch_stride > 0 && (params.k_batch_stride & 7) == 0) {
            const uint64_t stride_units = uint64_t(params.k_batch_stride) / 8;
            const uint64_t tail_units = uint64_t(params.page_block_size) *
                                        uint64_t(params.d / 8) - 1;
            use_compact_k_addr = tail_units <= UINT32_MAX && stride_units <= UINT32_MAX &&
                uint64_t(params.num_blocks - 1) <= (UINT32_MAX - tail_units) / stride_units;
        }
    }
    // [SPARSE-WI Stage B] BOOL_SWITCH on is_causal removed -- only one
    // instantiation; sparse path ignores params.is_causal.
    if constexpr (!IsV41FP8) {
    if (params.d == 576) {
        // [SPARSE-WI Stage D-step1] Switch to Traits_v2 -- adds fp8 nope/scales
        // and indices SMEM regions; everything else inherits 1:1 from splitkv.
        using T = Traits_v2<InputT, 576, IsFP8, BlockM, IsV41FP8, IsExtraFP4>;

        auto mla_kernel = &flash_sparse_decode_wg_kernel<T>;
        if constexpr (Arch == 89 && !IsFP8 && BlockM == 128 &&
                      std::is_same_v<InputT, cutlass::bfloat16_t>) {
            if (use_compact_k_addr) mla_kernel = &flash_sparse_decode_wg_kernel<CompactKAddrTraits<T>>;
        }
        int launch_parts = params.num_sm_parts;
        if constexpr (Arch == 89 && !IsFP8 && BlockM == 128 &&
                      std::is_same_v<InputT, cutlass::bfloat16_t>) {
            using StripeT = MetadataStripeTraits<T>;
            static_assert(sizeof(typename StripeT::SharedMemoryPlan) ==
                          sizeof(typename T::SharedMemoryPlan));
            static_assert(sizeof(typename StripeT::SharedMemoryOutPut) ==
                          sizeof(typename T::SharedMemoryOutPut));
            if (use_metadata_stripe2) {
                mla_kernel = &flash_sparse_decode_wg_kernel<StripeT>;
                launch_parts = params.num_sm_parts * 2;
            }
        }
        constexpr size_t smem_size = std::max(sizeof(typename T::SharedMemoryPlan), sizeof(typename T::SharedMemoryOutPut));

        hggcFuncSetAttribute(mla_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        const int num_m_block = cute::ceil_div(params.seqlen_q, T::kBlockM);

        int ctas_per_sm;
        hggcError status_ = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
            &ctas_per_sm, mla_kernel, T::NUM_THREADS, smem_size);

        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            printf("[splitkv_mla]:\n");
            printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);

            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, mla_kernel);
            const auto *dprops = at::cuda::getCurrentDeviceProperties();
            int sm_count = dprops->multiProcessorCount;
            if (std::string(dprops->name).find("810E") != std::string::npos) sm_count = 20;

            printf("blockM:%d, blockN:%d, threads:%d, block_size:%d\n",
                    T::kBlockM, T::kBlockN, T::NUM_THREADS, params.page_block_size);
            // [SPARSE-WI Stage B] Is_causal removed; sparse decode is mask-by-indices.
            printf("Is_causal:%d (sparse: ignored)\n", int(params.is_causal));
            printf("grid_n[%d, %d, %d]\n",
                    num_m_block, params.h, launch_parts);
            printf("verg:%d, stack:%d, sm:%d, occpuancy:%0.3f, Arch:%d\n", int(attr.numRegs), int(attr.localSizeBytes), sm_count,
                    float(num_m_block * params.h * launch_parts) / float(sm_count * ctas_per_sm), Arch);
        }

        // Use hggcLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
        hggcLaunchAttribute mla_kernel_attributes[1];
        mla_kernel_attributes[0].id = hggcLaunchAttributeProgrammaticStreamSerialization;
        mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
        hggcLaunchConfig_t mla_kernel_config = {
            dim3(num_m_block, params.h, launch_parts),
            dim3(T::NUM_THREADS, 1, 1),
            smem_size,
            stream,
            mla_kernel_attributes,
            1
        };
        hggcLaunchKernelEx(&mla_kernel_config, mla_kernel, params);

        CHECK_CUDA_KERNEL_LAUNCH();

        run_flash_mla_combine_kernel<InputT>(params, stream);
        return;
    }
    }
    {
        // Headdim=512
        using T = Traits_v2<InputT, 512, IsFP8, BlockM, IsV41FP8, IsExtraFP4>;

        auto mla_kernel = &flash_sparse_decode_wg_kernel<T>;
        if constexpr (Arch == 89 && !IsFP8 && BlockM == 128 &&
                      std::is_same_v<InputT, cutlass::bfloat16_t>) {
            if (use_compact_k_addr) mla_kernel = &flash_sparse_decode_wg_kernel<CompactKAddrTraits<T>>;
        }
        int launch_parts = params.num_sm_parts;
        if constexpr (Arch == 89 && !IsFP8 && BlockM == 128 &&
                      std::is_same_v<InputT, cutlass::bfloat16_t>) {
            using StripeT = MetadataStripeTraits<T>;
            static_assert(sizeof(typename StripeT::SharedMemoryPlan) ==
                          sizeof(typename T::SharedMemoryPlan));
            static_assert(sizeof(typename StripeT::SharedMemoryOutPut) ==
                          sizeof(typename T::SharedMemoryOutPut));
            if (use_metadata_stripe2) {
                mla_kernel = &flash_sparse_decode_wg_kernel<StripeT>;
                launch_parts = params.num_sm_parts * 2;
            }
        }
        constexpr size_t smem_size = std::max(sizeof(typename T::SharedMemoryPlan), sizeof(typename T::SharedMemoryOutPut));

        hggcFuncSetAttribute(mla_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        const int num_m_block = cute::ceil_div(params.seqlen_q, T::kBlockM);

        int ctas_per_sm;
        hggcError status_ = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
            &ctas_per_sm, mla_kernel, T::NUM_THREADS, smem_size);

        char *pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            printf("[splitkv_mla (d=512)]:\n");
            printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);

            hggcFuncAttributes attr;
            hggcFuncGetAttributes(&attr, mla_kernel);
            const auto *dprops = at::cuda::getCurrentDeviceProperties();
            int sm_count = dprops->multiProcessorCount;
            if (std::string(dprops->name).find("810E") != std::string::npos) sm_count = 20;

            printf("blockM:%d, blockN:%d, threads:%d, block_size:%d\n",
                    T::kBlockM, T::kBlockN, T::NUM_THREADS, params.page_block_size);
            printf("Is_causal:%d (sparse: ignored)\n", int(params.is_causal));
            printf("grid_n[%d, %d, %d]\n",
                    num_m_block, params.h, launch_parts);
            printf("verg:%d, stack:%d, sm:%d, occpuancy:%0.3f, Arch:%d\n", int(attr.numRegs), int(attr.localSizeBytes), sm_count,
                    float(num_m_block * params.h * launch_parts) / float(sm_count * ctas_per_sm), Arch);
        }

        // Use hggcLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
        hggcLaunchAttribute mla_kernel_attributes[1];
        mla_kernel_attributes[0].id = hggcLaunchAttributeProgrammaticStreamSerialization;
        mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
        hggcLaunchConfig_t mla_kernel_config = {
            dim3(num_m_block, params.h, launch_parts),
            dim3(T::NUM_THREADS, 1, 1),
            smem_size,
            stream,
            mla_kernel_attributes,
            1
        };
        hggcLaunchKernelEx(&mla_kernel_config, mla_kernel, params);

        CHECK_CUDA_KERNEL_LAUNCH();

        run_flash_mla_combine_kernel<InputT>(params, stream);
    }
}
