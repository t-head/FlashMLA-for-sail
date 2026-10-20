#pragma once

#include "decode/sparse/quant_decode_task.h"
#include "decode/sparse/sparse_quant_decode.h"
#include "decode/sparse/shared_quant_q_layout.h"
#include "kerutils/host/host.h"
#define FLASHMLA_V41_QUANT_ALLVALID_NO_LENGTHS 1
#define FLASHMLA_V41_QUANT_DECODE_PAGE_BASE64 1
#define FLASHMLA_V41_QUANT_DECODE_SCOPE_ROW32 1
#define FLASHMLA_V41_QUANT_DECODED_STORE_RELEASE 1
#include "prefill/sparse/quant_wi.cuh"
#include "ppuxx/decode/combine/combine.h"

namespace flashmla::v41prefill {

struct QuantDecodeDirectScheduler;
struct QuantDecodeBoundedPrefixScheduler;

template<bool ExtraFP4>
struct QuantDecodePairedTraits : QuantTraits<false> {
    using Params=QuantDecodeTileParams;
    using KernelParams=Flash_fwd_params;
    using Scheduler=QuantDecodeScheduler;
    using CacheAdapter=QuantDecodeCacheAdapter;
    static constexpr bool kDecode=true;
    static constexpr bool kMixedFP4=ExtraFP4;
    static constexpr bool kDecodeExtraFP4AsyncEarly=true;
    static constexpr bool kTaskAllValidGuard=false;
    static constexpr bool kTaskAllValidVotes=false;
    static constexpr bool kDecodePageBase64=!ExtraFP4;
    static constexpr bool kSingleTaskCTA=false;
    static constexpr bool kLengthGuardAllowed=false;
};

// The guarded entry is reserved for FP4. Its scheduler verifies live row
// contents every task before choosing the common tile specialization.
template<bool ExtraFP4>
struct QuantDecodeGuardedPairedTraits : QuantDecodePairedTraits<ExtraFP4> {
    static_assert(ExtraFP4);
    static constexpr bool kTaskAllValidGuard=true;
};

// Independent cache scopes are proven bounded on every host call. All other
// addresses and the per-task live validity guard retain their original types.
struct QuantDecodeScopeNarrowGuardedTraits : QuantDecodeGuardedPairedTraits<true> {
    static constexpr bool kCacheOffset32=true;
};

// Explicit new main names keep direct scheduling distinguishable in traces.
template<bool ExtraFP4>
struct QuantDecodeDirectPairedTraits : QuantDecodePairedTraits<ExtraFP4> {
    using Scheduler=QuantDecodeDirectScheduler;
    static constexpr bool kSingleTaskCTA=true;
};

// These checked suppliers retain live length/index masks. Only their per-cache
// address state narrows after the independent host span proofs pass.
struct QuantDecodeSq1CheckedU32FP8Traits : QuantDecodeDirectPairedTraits<false> {
    static constexpr bool kDecodeFP8ScopeRow32=true;
    static constexpr bool kDecodePageBase64=false;
};
struct QuantDecodeSq1CheckedOffset32FP4Traits : QuantDecodeDirectPairedTraits<true> {
    static constexpr bool kCacheOffset32=true;
};

// Preserve the checked supplier and task projection while assigning each
// large-storage CTA one existing metadata partition, as in the legacy grid.
struct QuantDecodeSq1PersistentU32FP8Traits : QuantDecodeSq1CheckedU32FP8Traits {
    using Scheduler=QuantDecodeScheduler;
    static constexpr bool kSingleTaskCTA=false;
};
struct QuantDecodeSq1PersistentOffset32FP4Traits : QuantDecodeSq1CheckedOffset32FP4Traits {
    using Scheduler=QuantDecodeScheduler;
    static constexpr bool kSingleTaskCTA=false;
};

struct QuantDecodeDirectGuardedPairedTraits : QuantDecodeDirectPairedTraits<true> {
    static constexpr bool kTaskAllValidGuard=true;
};

// Both independent cache spans are proven before selecting this direct entry.
struct QuantDecodeDirectScopeNarrowGuardedTraits : QuantDecodeDirectGuardedPairedTraits {
    static constexpr bool kCacheOffset32=true;
};

struct QuantDecodeDirectScopeNarrowNativeFP4WG1PublishFirstRemotePVTraits : QuantDecodeDirectScopeNarrowGuardedTraits {
    static constexpr bool kDecodeNativeFP4Steady=true;
    static constexpr bool kDecodeFP4WG1RemotePVBeforeFinish=true;
    static constexpr bool kDecodeFP4WG1FirstFinishAfterScale1=true;
    static constexpr bool kDecodeFP4WG0Scale0SchedBound=true;
};

// Only the pure FP8 Direct entry consumes fresh per-call partition votes.
template<bool ExtraFP4>
struct QuantDecodeDirectVotesPairedTraits : QuantDecodeDirectPairedTraits<ExtraFP4> {
    static_assert(!ExtraFP4);
    static constexpr bool kTaskAllValidVotes=true;
};

// A private pure-FP8 entry; host proves each independent cache span fits UInt32.
struct QuantDecodeDirectVotesScopeU32PairedTraits : QuantDecodeDirectVotesPairedTraits<false> {
    static constexpr bool kDecodeFP8ScopeRow32=true;
    static constexpr bool kDecodePageBase64=false;
};

// One additional entry keeps the steady-supply experiment explicit in traces.
struct QuantDecodeDirectVotesScopeU32FP8WG1PublishBoundWG0PostP0Traits : QuantDecodeDirectVotesScopeU32PairedTraits {
    static constexpr bool kDecodeFixedExtraFP8Steady=true;
    static constexpr bool kDecodeFP8WG1EarlyPacket=true;
    static constexpr bool kDecodeFP8WG1SupplyAfterScale1=true;
    static constexpr bool kDecodeFP8WG0FinishAfterP0=true;
};

// Bounded-storage Sq1 entries prove full live lengths and all indices per task.
// The FP8 task proof replaces partition votes; neither entry caches validity.
struct QuantDecodeSq1LengthsGuardedFP8Traits
    : QuantDecodeDirectVotesScopeU32FP8WG1PublishBoundWG0PostP0Traits {
    static constexpr bool kTaskAllValidVotes=false;
    static constexpr bool kTaskAllValidGuard=true;
    static constexpr bool kLengthGuardAllowed=true;
};
struct QuantDecodeSq1LengthsGuardedFP4Traits
    : QuantDecodeDirectScopeNarrowNativeFP4WG1PublishFirstRemotePVTraits {
    static constexpr bool kLengthGuardAllowed=true;
};

// The canonical paired metadata/split table permits a bounded ordinal lookup.
// Guarded bodies and the single-task CTA grid retain the original Direct form.
struct QuantDecodeSq1LengthsPrefixFP8Traits : QuantDecodeSq1LengthsGuardedFP8Traits {
    using Scheduler=QuantDecodeBoundedPrefixScheduler;
    static constexpr bool kSingleTaskCTA=true;
};
struct QuantDecodeSq1LengthsPrefixFP4Traits : QuantDecodeSq1LengthsGuardedFP4Traits {
    using Scheduler=QuantDecodeBoundedPrefixScheduler;
    static constexpr bool kSingleTaskCTA=true;
};

template<typename T>
struct QuantDecodeAllValidTaskTraits : T {
    static constexpr bool kAllValidNoLengths=true;
};

template<typename T,bool FP4>
struct QuantDecodeScopeTraits : T {
    using Params=SparseQuantPrefillParams;
    static constexpr bool kDecode=false;
    static constexpr bool kMixedFP4=false;
    static constexpr bool kFP4Cache=FP4;
};

__forceinline__ __device__ SparseQuantPrefillParams quant_decode_cache_scope(
        const QuantDecodeTileParams &params,int block_idx) {
    const bool extra=block_idx*32>=params.main_padded_length;
    SparseQuantPrefillParams scope{};
    scope.kv=extra?params.extra_kv:params.kv;
    scope.quant_page_size=extra?params.extra_page_size:params.quant_page_size;
    scope.quant_block_stride=extra?params.extra_block_stride:params.quant_block_stride;
    return scope;
}

// The caller proves every future raw key belongs to extra before selecting
// this policy. Main/extra token positions and softmax masks remain original.
struct QuantDecodeFixedExtraFP4CacheAdapter {
    template<typename T>
    static __forceinline__ __device__ SparseQuantPrefillParams scope(const typename T::Params &p) {
        static_assert(T::kDecode && T::kFP4Cache && !T::kMixedFP4);
        SparseQuantPrefillParams result{};
        result.kv=p.extra_kv;
        result.quant_page_size=p.extra_page_size;
        result.quant_block_stride=p.extra_block_stride;
        return result;
    }

    template<int S,int E,typename T>
    static __forceinline__ __device__ V41Fp4RegisterPacket register_load(
            const typename T::Params &p,int tidx,int token_idx,bool valid,int) {
        return v41_prefill_register_load<S,E,QuantDecodeScopeTraits<T,true>>(
            scope<T>(p),tidx,token_idx,valid);
    }

    template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK,typename Packet>
    static __forceinline__ __device__ void register_finish(
            const typename T::Params &,RawK &raw_sK,int tidx,int,bool,const Packet &packet) {
        static_assert(std::is_same_v<Packet,V41Fp4RegisterPacket>);
        v41_prefill_fp4_register_finish<S,E,QuantDecodeScopeTraits<T,true>,FRONT_DRAIN>(
            raw_sK,tidx,packet);
    }

    template<int S,int E,typename T>
    static __forceinline__ __device__ void async_issue(
            const typename T::Params &p,int tidx,int token_idx,bool valid,int) {
        v41_prefill_fp4_async_issue<S,E,QuantDecodeScopeTraits<T,true>>(
            scope<T>(p),tidx,token_idx,valid);
    }

    template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
    static __forceinline__ __device__ void load(
            const typename T::Params &p,RawK &raw_sK,int tidx,int token_idx,
            bool valid,int loaded_block,bool async_preissued) {
        if(!async_preissued) async_issue<S,E,T>(p,tidx,token_idx,valid,loaded_block);
        v41_prefill_fp4_async_finish<S,E,QuantDecodeScopeTraits<T,true>,FRONT_DRAIN>(raw_sK,tidx);
    }
};
// Every supplied block is extra after the two-block lookahead crosses main.
// Keep the existing UInt32 offset/delta and only specialize its cache scope.
struct QuantDecodeFixedExtraFP8CacheAdapter : QuantDecodeCacheAdapter {
    template<typename T>
    static __forceinline__ __device__ V41QuantRowAddress32 prepare_scope_row32(
            const typename T::Params &p,int token_idx,bool valid,int) {
        static_assert(T::kDecode && T::kDecodeFP8ScopeRow32 && !T::kMixedFP4 && !T::kFP4Cache);
        V41QuantRowAddress32 row{};
        if(valid) {
            const uint32_t page=uint32_t(token_idx)>>__builtin_ctz(p.extra_page_size);
            const uint32_t off=uint32_t(token_idx)&uint32_t(p.extra_page_size-1);
            row.row_payload_offset=page*uint32_t(p.extra_block_stride)+off*512u;
            row.scale_delta=uint32_t(p.extra_page_size)*512u-off*(512u-16u);
        }
        return row;
    }

    template<typename T>
    static __forceinline__ __device__ V41QuantRowAddress64 widen(
            const typename T::Params &p,bool valid,const V41QuantRowAddress32 &source) {
        V41QuantRowAddress64 row{};
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
            int,const V41QuantRowAddress32 &source) {
        const auto row=widen<T>(p,valid,source);
        SparseQuantPrefillParams unused_scope{};
        return v41_prefill_register_load<S,E,QuantDecodeScopeTraits<T,false>>(
            unused_scope,tidx,token_idx,valid,row);
    }

    template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
    static __forceinline__ __device__ void load_scope_row32(
            const typename T::Params &p,RawK &raw_sK,int tidx,int token_idx,
            bool valid,int,const V41QuantRowAddress32 &source) {
        const auto row=widen<T>(p,valid,source);
        SparseQuantPrefillParams unused_scope{};
        v41_prefill_load_fp8<S,E,QuantDecodeScopeTraits<T,false>,FRONT_DRAIN>(
            unused_scope,raw_sK,tidx,token_idx,valid,row);
    }
    // Inherited register_finish is packet-only; page64/generic suppliers and
    // FP4 async_issue are compile-time unreachable for this UInt32 FP8 trait.
};

template<typename T>
__forceinline__ __device__ V41QuantRowAddress32 QuantDecodeCacheAdapter::prepare_scope_row32(
        const typename T::Params &params,int token_idx,bool valid,int loaded_block) {
    static_assert(T::kDecode && T::kDecodeFP8ScopeRow32 && !T::kMixedFP4);
    V41QuantRowAddress32 row{};
    if(valid) {
        const auto scope=quant_decode_cache_scope(params,loaded_block);
        const uint32_t page=uint32_t(token_idx)>>__builtin_ctz(scope.quant_page_size);
        const uint32_t off=uint32_t(token_idx)&uint32_t(scope.quant_page_size-1);
        row.row_payload_offset=page*uint32_t(scope.quant_block_stride)+off*512u;
        row.scale_delta=uint32_t(scope.quant_page_size)*512u-off*(512u-16u);
    }
    return row;
}

// The carry state stays two UInt32 values. Materialize the wide base only at
// the consumer, with the actual loaded block selecting its independent scope.
template<typename T>
__forceinline__ __device__ V41QuantRowAddress64 quant_decode_scope_row32_widen(
        const typename T::Params &params,bool valid,int loaded_block,
        const V41QuantRowAddress32 &source) {
    V41QuantRowAddress64 row{};
    if(valid) {
        const auto scope=quant_decode_cache_scope(params,loaded_block);
        row.payload_row=reinterpret_cast<const uint8_t*>(scope.kv)
            +uint64_t(source.row_payload_offset);
        row.scale_delta=int64_t(uint64_t(source.scale_delta));
    }
    return row;
}

template<int S,int E,typename T>
__forceinline__ __device__ auto QuantDecodeCacheAdapter::register_load_scope_row32(
        const typename T::Params &params,int tidx,int token_idx,bool valid,
        int loaded_block,const V41QuantRowAddress32 &source) {
    static_assert(T::kDecode && T::kDecodeFP8ScopeRow32 && !T::kMixedFP4);
    const auto row=quant_decode_scope_row32_widen<T>(params,valid,loaded_block,source);
    SparseQuantPrefillParams unused_scope{};
    return v41_prefill_register_load<S,E,QuantDecodeScopeTraits<T,false>>(
        unused_scope,tidx,token_idx,valid,row);
}

template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
__forceinline__ __device__ void QuantDecodeCacheAdapter::load_scope_row32(
        const typename T::Params &params,RawK &raw_sK,int tidx,int token_idx,
        bool valid,int loaded_block,const V41QuantRowAddress32 &source) {
    static_assert(T::kDecode && T::kDecodeFP8ScopeRow32 && !T::kMixedFP4);
    const auto row=quant_decode_scope_row32_widen<T>(params,valid,loaded_block,source);
    SparseQuantPrefillParams unused_scope{};
    v41_prefill_load_fp8<S,E,QuantDecodeScopeTraits<T,false>,FRONT_DRAIN>(
        unused_scope,raw_sK,tidx,token_idx,valid,row);
}

template<typename T>
__forceinline__ __device__ typename T::InputT *QuantDecodeCacheAdapter::prepare_page_base64(
        const typename T::Params &params,int token_idx,bool valid,int loaded_block) {
    static_assert(T::kDecode && T::kDecodePageBase64 && !T::kMixedFP4);
    if(!valid) return nullptr;
    const auto scope=quant_decode_cache_scope(params,loaded_block);
    const int page=token_idx>>__builtin_ctz(scope.quant_page_size);
    return reinterpret_cast<typename T::InputT*>(reinterpret_cast<uint8_t*>(scope.kv)
        +int64_t(page)*scope.quant_block_stride);
}

template<typename T>
__forceinline__ __device__ V41QuantRowAddress64 quant_decode_local_page_row64(
        const typename T::Params &params,int token_idx,bool valid,int loaded_block,
        const typename T::InputT *page_base) {
    V41QuantRowAddress64 row{};
    if(valid) {
        const auto scope=quant_decode_cache_scope(params,loaded_block);
        const int off=token_idx&(scope.quant_page_size-1);
        row.payload_row=reinterpret_cast<const uint8_t*>(page_base)+int64_t(off)*512;
        row.scale_delta=int64_t(scope.quant_page_size)*512-int64_t(off)*(512-16);
    }
    return row;
}

template<int S,int E,typename T>
__forceinline__ __device__ auto QuantDecodeCacheAdapter::register_load_page64(
        const typename T::Params &params,int tidx,int token_idx,bool valid,
        int loaded_block,const typename T::InputT *page_base) {
    static_assert(T::kDecode && T::kDecodePageBase64 && !T::kMixedFP4);
    const auto row=quant_decode_local_page_row64<T>(params,token_idx,valid,loaded_block,page_base);
    SparseQuantPrefillParams unused_scope{};
    return v41_prefill_register_load<S,E,QuantDecodeScopeTraits<T,false>>(
        unused_scope,tidx,token_idx,valid,row);
}

template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
__forceinline__ __device__ void QuantDecodeCacheAdapter::load_page64(
        const typename T::Params &params,RawK &raw_sK,int tidx,int token_idx,
        bool valid,int loaded_block,const typename T::InputT *page_base) {
    static_assert(T::kDecode && T::kDecodePageBase64 && !T::kMixedFP4);
    const auto row=quant_decode_local_page_row64<T>(params,token_idx,valid,loaded_block,page_base);
    SparseQuantPrefillParams unused_scope{};
    v41_prefill_load_fp8<S,E,QuantDecodeScopeTraits<T,false>,FRONT_DRAIN>(
        unused_scope,raw_sK,tidx,token_idx,valid,row);
}

template<int S,int E,typename T>
__forceinline__ __device__ auto QuantDecodeCacheAdapter::register_load(
        const typename T::Params &params,int tidx,int token_idx,bool valid,int block_idx) {
    const auto scope=quant_decode_cache_scope(params,block_idx);
    if constexpr(T::kMixedFP4) {
        V41MixedRegisterPacket packet{};
        packet.fp4_format=block_idx*T::kBlockN>=params.main_padded_length;
        if(packet.fp4_format) {
            packet.fp4=v41_prefill_register_load<S,E,QuantDecodeScopeTraits<T,true>>(
                scope,tidx,token_idx,valid);
        } else {
            packet.fp8=v41_prefill_register_load<S,E,QuantDecodeScopeTraits<T,false>>(
                scope,tidx,token_idx,valid);
        }
        return packet;
    } else {
        return v41_prefill_register_load<S,E,QuantDecodeScopeTraits<T,false>>(
            scope,tidx,token_idx,valid);
    }
}

template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK,typename Packet>
__forceinline__ __device__ void QuantDecodeCacheAdapter::register_finish(
        const typename T::Params &,RawK &raw_sK,int tidx,int token_idx,
        bool valid,const Packet &packet) {
    SparseQuantPrefillParams unused_scope{};
    if constexpr(T::kMixedFP4) {
        if(packet.fp4_format) {
            v41_prefill_fp4_register_finish<S,E,QuantDecodeScopeTraits<T,true>,FRONT_DRAIN>(
                raw_sK,tidx,packet.fp4);
        } else {
            v41_prefill_fp8_register_finish<S,E,QuantDecodeScopeTraits<T,false>,FRONT_DRAIN>(
                unused_scope,raw_sK,tidx,token_idx,valid,packet.fp8);
        }
    } else {
        v41_prefill_fp8_register_finish<S,E,QuantDecodeScopeTraits<T,false>,FRONT_DRAIN>(
            unused_scope,raw_sK,tidx,token_idx,valid,packet);
    }
}

template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
__forceinline__ __device__ void QuantDecodeCacheAdapter::load(
        const typename T::Params &params,RawK &raw_sK,int tidx,int token_idx,
        bool valid,int block_idx,bool async_preissued) {
    const auto scope=quant_decode_cache_scope(params,block_idx);
    if constexpr(T::kMixedFP4) {
        if(block_idx*T::kBlockN>=params.main_padded_length) {
            using ScopeT=QuantDecodeScopeTraits<T,true>;
            if(!async_preissued) v41_prefill_fp4_async_issue<S,E,ScopeT>(scope,tidx,token_idx,valid);
            v41_prefill_fp4_async_finish<S,E,ScopeT,FRONT_DRAIN>(raw_sK,tidx);
        } else {
            v41_prefill_load_fp8<S,E,QuantDecodeScopeTraits<T,false>,FRONT_DRAIN>(
                scope,raw_sK,tidx,token_idx,valid);
        }
    } else {
        v41_prefill_load_fp8<S,E,QuantDecodeScopeTraits<T,false>,FRONT_DRAIN>(
            scope,raw_sK,tidx,token_idx,valid);
    }
}

template<int S,int E,typename T>
__forceinline__ __device__ void QuantDecodeCacheAdapter::async_issue(
        const typename T::Params &params,int tidx,int token_idx,bool valid,int block_idx) {
    static_assert(T::kDecode && T::kMixedFP4);
    // Scope selection is CTA-uniform. Main FP8 never issues an FP4 packet.
    if(block_idx*T::kBlockN>=params.main_padded_length) {
        const auto scope=quant_decode_cache_scope(params,block_idx);
        v41_prefill_fp4_async_issue<S,E,QuantDecodeScopeTraits<T,true>>(
            scope,tidx,token_idx,valid);
    }
}

// The task guard already proves a complete current row. Under the extra
// capacity bound, reject negative int32 indices with one unsigned compare.
// All 512 threads scan their disjoint elements without an early exit on holes.
__forceinline__ __device__ bool quant_decode_task_row_allvalid_scan32(
        const int *indices,uint32_t storage,uint32_t capacity,uint32_t rank) {
    constexpr uint32_t kThreads=512;
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

template<typename T>
__forceinline__ __device__ bool quant_decode_task_allvalid(
        const Flash_fwd_params &p,const QuantDecodeTileParams &task) {
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

    // Prove the actual current row and work range, not cached metadata validity.
    // Complete storage rows also cover in-range lookahead beyond this partition.
    const bool eligible=p.topk_len_ptr==nullptr && p.extra_topk_len_ptr==nullptr
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
    int lane_valid=eligible?1:0;
    if(eligible) {
        if(main_capacity<=int64_t(0x7fffffff) &&
           (task.extra_storage_topk==0 || extra_capacity<=int64_t(0x7fffffff))) {
            lane_valid&=quant_decode_task_row_allvalid_scan32(task.indices,
                uint32_t(task.main_storage_topk),uint32_t(main_capacity),uint32_t(tidx));
            lane_valid&=quant_decode_task_row_allvalid_scan32(task.extra_indices,
                uint32_t(task.extra_storage_topk),uint32_t(extra_capacity),uint32_t(tidx));
        } else {
            // Preserve the existing wide-capacity semantics outside the proof.
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
    // Every warp consumes the same 16 votes; all 512 threads choose one body.
    const int task_valid=__all_sync(0xffffffffu,all_flag_vote);
    return __builtin_ppu_to_uniform_b32(task_valid)!=0;
}

// Length pointers are permitted only in these private bounded-storage entries.
// Live row length must equal storage before the original complete-row proof.
template<typename T>
__forceinline__ __device__ bool quant_decode_task_allvalid_full_lengths(
        const Flash_fwd_params &p,const QuantDecodeTileParams &task) {
    static_assert(T::kLengthGuardAllowed && T::kTaskAllValidGuard);
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

    // Prove the actual current row and work range, not cached metadata validity.
    // Complete storage rows also cover in-range lookahead beyond this partition.
    const bool eligible=task.main_length>0 && task.main_length==task.main_storage_topk
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
    // Every eligibility field is CTA-uniform. A shortened row keeps the
    // original checked body without scanning indices or entering guard votes.
    if(!eligible) return false;
    int lane_valid=1;
    if(eligible) {
        if(main_capacity<=int64_t(0x7fffffff) &&
           (task.extra_storage_topk==0 || extra_capacity<=int64_t(0x7fffffff))) {
            lane_valid&=quant_decode_task_row_allvalid_scan32(task.indices,
                uint32_t(task.main_storage_topk),uint32_t(main_capacity),uint32_t(tidx));
            lane_valid&=quant_decode_task_row_allvalid_scan32(task.extra_indices,
                uint32_t(task.extra_storage_topk),uint32_t(extra_capacity),uint32_t(tidx));
        } else {
            // Preserve the existing wide-capacity semantics outside the proof.
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
    // Every warp consumes the same 16 votes; all 512 threads choose one body.
    const int task_valid=__all_sync(0xffffffffu,all_flag_vote);
    return __builtin_ppu_to_uniform_b32(task_valid)!=0;
}


// The API allocates [b+parts,h,seqlen_q,d_v] FP32 accumulation every call.
// Valid disjoint metadata has at most b+parts-1 split rows, so its last
// allocated row is private spare storage. No metadata or public ABI is changed.
__host__ __device__ __forceinline__ bool quant_decode_votes_capacity(
        const Flash_fwd_params &p) {
    if(p.oaccum_ptr==nullptr || p.num_splits_ptr==nullptr || p.b<=0 ||
       p.num_sm_parts<=0 || p.h!=1 || p.seqlen_q<=0 || p.d_v!=512) return false;
    const int64_t row_floats=int64_t(p.seqlen_q)*p.d_v;
    const int64_t allocated_rows=int64_t(p.b)+p.num_sm_parts;
    return row_floats>=p.num_sm_parts &&
        allocated_rows<=int64_t(0x7fffffffffffffffLL)/(row_floats*int64_t(sizeof(float)));
}

__forceinline__ __device__ bool quant_decode_votes_split_safe(
        const Flash_fwd_params &p) {
    if(!quant_decode_votes_capacity(p)) return false;
    const int splits=__ldg(p.num_splits_ptr+p.b);
    const int64_t last_row=int64_t(p.b)+p.num_sm_parts-1;
    // Recheck live prefix end before any main CTA reads a flag. For the valid
    // monotone scheduler prefix this proves every partial row precedes votes.
    return splits>=p.b && int64_t(splits)<=last_row;
}

__forceinline__ __device__ float *quant_decode_votes_ptr(
        const Flash_fwd_params &p) {
    const int64_t last_row=int64_t(p.b)+p.num_sm_parts-1;
    return static_cast<float*>(p.oaccum_ptr)+last_row*int64_t(p.seqlen_q)*p.d_v;
}

// The caller proves storage/capacity <= INT_MAX in geometry. Pointer row
// offsets remain64-bit; unsigned index comparison also rejects every negative
// int32 index because capacity is at most INT_MAX. Do not exit early on holes.
__forceinline__ __device__ bool quant_decode_row_allvalid_scan32(
        const int *indices,uint32_t storage,uint32_t capacity,uint32_t rank) {
    constexpr uint32_t kThreads=256;
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

// Ordinary same-stream launch must finish before the votes entry starts.
__global__ void __launch_bounds__(256,1,1) quant_decode_allvalid_votes_kernel(
        __grid_constant__ const Flash_fwd_params p) {
    constexpr int kThreads=256,kWarps=8;
    const int partition=blockIdx.x;
    if(partition>=p.num_sm_parts || !quant_decode_votes_capacity(p)) return;
    __shared__ int warp_votes[kWarps];
    const int tidx=threadIdx.x,lane=tidx&31,warp=tidx/32;
    const int64_t main_capacity=int64_t(p.num_blocks)*p.page_block_size;
    const int64_t extra_capacity=int64_t(p.extra_num_blocks)*p.extra_page_block_size;
    const int extra_storage=p.extra_topk>=0?p.extra_topk:0;
    const int64_t total=int64_t(p.topk)+extra_storage;
    const int queries=p.ngroups>0?p.seqlen_q/p.ngroups:0;
    const bool geometry=p.topk_len_ptr==nullptr && p.extra_topk_len_ptr==nullptr
        && p.topk>0 && p.topk%64==0 && extra_storage%64==0
        && (total>=2048 || (p.ngroups==128 && p.topk==128
            && extra_storage>=128 && total>=256))
        && total<=int64_t(0x7fffffff)
        && p.ngroups>0 && p.ngroups%128==0 && queries>0
        && p.seqlen_q%p.ngroups==0
        && p.k_ptr!=nullptr && p.indices_ptr!=nullptr && p.tile_scheduler_metadata_ptr!=nullptr
        && main_capacity>0 && main_capacity<=int64_t(0x7fffffff)
        && (extra_storage==0 || (p.extra_k_ptr!=nullptr && p.extra_indices_ptr!=nullptr
            && extra_capacity>0 && extra_capacity<=int64_t(0x7fffffff)));
    const bool split_safe=quant_decode_votes_split_safe(p);
    int lane_valid=geometry && split_safe;
    if(lane_valid) {
        const int *metadata=p.tile_scheduler_metadata_ptr+int64_t(partition)*TileSchedulerMetaDataSize;
        const int4 schedule=*reinterpret_cast<const int4*>(metadata);
        const int first_split=metadata[4];
        const bool range=schedule.x>=0 && schedule.x<p.b && schedule.z>=schedule.x && schedule.z<p.b
            && schedule.y>=0 && schedule.y%64==0 && schedule.w>0 && schedule.w%64==0;
        lane_valid&=range;
        if(range) {
            #pragma unroll 1
            for(int batch=schedule.x;batch<=schedule.z;++batch) {
                const int64_t start=batch==schedule.x?int64_t(schedule.y)/32:0;
                const int64_t end=batch==schedule.z?int64_t(schedule.w)/32:total/32;
                const int split_begin=__ldg(p.num_splits_ptr+batch);
                const int split_end=__ldg(p.num_splits_ptr+batch+1);
                const int64_t split=int64_t(split_begin)+(batch==schedule.x?first_split:0);
                lane_valid&=start>=0 && start%2==0 && end>start && end%2==0
                    && end<=total/32 && split_begin>=0 && split_end>split_begin
                    && int64_t(split_end)<=int64_t(p.b)+p.num_sm_parts-1
                    && first_split>=0 && split>=split_begin && split<split_end;
                // Scan complete current rows, including safe lookahead beyond
                // this task's end. All Sq query rows share this partition vote.
                #pragma unroll 1
                for(int query=0;query<queries;++query) {
                    const int *main_indices=p.indices_ptr+int64_t(batch)*p.indices_batch_stride
                        +int64_t(query)*p.indices_row_stride;
                    const int *extra_indices=extra_storage>0
                        ?p.extra_indices_ptr+int64_t(batch)*p.extra_indices_batch_stride
                            +int64_t(query)*p.extra_indices_row_stride:nullptr;
                    lane_valid&=quant_decode_row_allvalid_scan32(main_indices,
                        uint32_t(p.topk),uint32_t(main_capacity),uint32_t(tidx));
                    lane_valid&=quant_decode_row_allvalid_scan32(extra_indices,
                        uint32_t(extra_storage),uint32_t(extra_capacity),uint32_t(tidx));
                }
            }
        }
    }
    const int warp_valid=__all_sync(0xffffffffu,lane_valid);
    if(lane==0) warp_votes[warp]=warp_valid;
    __syncthreads();
    if(warp==0) {
        const int vote=lane<kWarps?warp_votes[lane]:1;
        const int all_valid=__all_sync(0xffffffffu,vote);
        if(lane==0) quant_decode_votes_ptr(p)[partition]=all_valid?1.0f:0.0f;
    }
}

// Project one metadata task without changing live lengths or output storage.
// Both schedulers retain their own task enumeration and validity proofs.
template<typename T>
__forceinline__ __device__ QuantDecodeTileParams make_quant_decode_task(
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

    QuantDecodeTileParams task{};
    task.s_q=1; task.s_kv=p.num_blocks*p.page_block_size;
    task.h_q=p.ngroups; task.h_kv=1; task.d_qk=p.d; task.d_v=p.d_v;
    task.topk=total; task.sm_scale=p.scale_softmax; task.sm_scale_div_log2=p.scale_softmax_log2;
    task.q=static_cast<cutlass::bfloat16_t*>(p.q_ptr)+int64_t(batch)*p.q_batch_stride
        +int64_t(query)*p.q_row_stride;
    task.kv=p.k_ptr;
    task.indices=p.indices_ptr+int64_t(batch)*p.indices_batch_stride+int64_t(query)*p.indices_row_stride;
    task.attn_sink=p.attn_sink_ptr;
    task.stride_q_s_q=p.q_row_stride;
    task.stride_q_h_q=p.q_head_stride;
    task.stride_kv_s_kv=p.k_row_stride; task.stride_kv_h_kv=p.k_head_stride;
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

struct QuantDecodeScheduler {
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
            QuantDecodeTileParams task = make_quant_decode_task<T>(
                p,schedule,first_split,batch,query);
            if constexpr(T::kTaskAllValidGuard) {
                const bool task_allvalid=quant_decode_task_allvalid<T>(p,task);
                // Hold flags in registers before body initialization or the FP32
                // output overlay can reuse the shared flag region.
                if(task_allvalid) {
                    quant_wi_tile_body<QuantDecodeAllValidTaskTraits<T>,true>(task,head_tile);
                } else {
                    quant_wi_tile_body<T,true>(task,head_tile);
                }
            } else {
                // Paired entries, including all FP8 calls, compile no guard.
                quant_wi_tile_body<T,true>(task,head_tile);
            }
            // Every producer and output reader finishes before the shared plan
            // and barrier phases are initialized for the next metadata task.
            __syncthreads();
        }
    }
};

struct QuantDecodeDirectScheduler {
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
            // The generator's empty tail has begin==b and end==b-1. Active
            // records inherit the same disjoint/monotone metadata contract.
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
        // b+parts-1 is an upper bound; no barriers or Q/K loads for ghosts.
        if(partition<0) return;
        const int *metadata=p.tile_scheduler_metadata_ptr
            +int64_t(partition)*TileSchedulerMetaDataSize;
        const int first_split=metadata[4];
        bool partition_allvalid=false;
        if constexpr(T::kTaskAllValidVotes) {
            // The checker covers all tasks and all Sq rows in this partition.
            // A live-prefix recheck precedes every per-CTA vote read.
            if(quant_decode_votes_split_safe(p)) {
                const float vote=quant_decode_votes_ptr(p)[partition];
                partition_allvalid=__builtin_ppu_to_uniform_b32(vote==1.0f)!=0;
            }
        }
        {
            QuantDecodeTileParams task = make_quant_decode_task<T>(
                p,schedule,first_split,batch,query);
            if constexpr(T::kTaskAllValidVotes) {
                if(partition_allvalid) {
                    quant_wi_tile_body<QuantDecodeAllValidTaskTraits<T>,true>(task,head_tile);
                } else {
                    quant_wi_tile_body<T,true>(task,head_tile);
                }
            } else if constexpr(T::kTaskAllValidGuard) {
                const bool task_allvalid=[&]() {
                    if constexpr(T::kLengthGuardAllowed) {
                        return quant_decode_task_allvalid_full_lengths<T>(p,task);
                    } else {
                        return quant_decode_task_allvalid<T>(p,task);
                    }
                }();
                // Hold flags in registers before body initialization or the FP32
                // output overlay can reuse the shared flag region.
                if(task_allvalid) {
                    quant_wi_tile_body<QuantDecodeAllValidTaskTraits<T>,true>(task,head_tile);
                } else {
                    if constexpr(T::kLengthGuardAllowed) {
                        // The host already proved both cache spans; a failed
                        // live proof keeps every mask in the checked supplier.
                        using CheckedT=std::conditional_t<T::kMixedFP4,
                            QuantDecodeSq1CheckedOffset32FP4Traits,
                            QuantDecodeSq1CheckedU32FP8Traits>;
                        static_assert(std::is_same_v<typename CheckedT::Params,typename T::Params>);
                        static_assert(std::is_same_v<typename CheckedT::SharedMemoryPlan,typename T::SharedMemoryPlan>);
                        static_assert(std::is_same_v<typename CheckedT::SharedMemoryOutPut,typename T::SharedMemoryOutPut>);
                        quant_wi_tile_body<CheckedT,true>(task,head_tile);
                    } else {
                        quant_wi_tile_body<T,true>(task,head_tile);
                    }
                }
            } else {
                // Paired entries, including all FP8 calls, compile no guard.
                quant_wi_tile_body<T,true>(task,head_tile);
            }
            // Every producer and output reader finishes before the shared plan
            // and barrier phases are initialized for the next metadata task.
            __syncthreads();
        }
    }
};

struct QuantDecodeBoundedPrefixScheduler {
    template<typename T>
    static __forceinline__ __device__ void run(const Flash_fwd_params &p) {
        static_assert(!T::kTaskAllValidVotes,
            "Unsplit ordinal projection has no partition validity vote");
        const int64_t ordinal=int64_t(blockIdx.z);
        const int tile=blockIdx.x;
        const int head_tiles=p.ngroups/T::kBlockM;
        const int query=tile/head_tiles;
        const int head_tile=tile%head_tiles;
        // The standard producer publishes matching metadata and split prefixes
        // before this kernel on the same stream. Retain ordinary mutable loads.
        const int64_t split_origin=int64_t(p.num_splits_ptr[0]);
        const int64_t total_tasks=int64_t(p.num_splits_ptr[p.b])-split_origin;
        if(ordinal>=total_tasks) return;
        int partition=-1,batch=-1,first_split=0;
        int4 schedule{};
        if(total_tasks==int64_t(p.b)) {
            // The canonical producer assigns each row now_n_split_idx+1 >= 1
            // tasks. A total of b therefore proves one complete task per row.
            batch=int(ordinal);
            // This is a private full-row projection sentinel, not metadata.
            // ordinal < total_tasks == b makes both first/last-row tests false;
            // make_quant_decode_task retains live lengths and selects [0,total).
            schedule={p.b,0,p.b,0};
        } else {
            int lower=0,upper=p.num_sm_parts;
            #pragma unroll 1
            while(lower<upper) {
                const int candidate=lower+(upper-lower)/2;
                const int *candidate_metadata=p.tile_scheduler_metadata_ptr
                    +int64_t(candidate)*TileSchedulerMetaDataSize;
                const int4 candidate_schedule=*reinterpret_cast<const int4*>(candidate_metadata);
                // Empty producer tails have begin==b, first_split==0 and prefix==total.
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
        bool partition_allvalid=false;
        if constexpr(T::kTaskAllValidVotes) {
            // The checker covers all tasks and all Sq rows in this partition.
            // A live-prefix recheck precedes every per-CTA vote read.
            if(quant_decode_votes_split_safe(p)) {
                const float vote=quant_decode_votes_ptr(p)[partition];
                partition_allvalid=__builtin_ppu_to_uniform_b32(vote==1.0f)!=0;
            }
        }
        {
            QuantDecodeTileParams task = make_quant_decode_task<T>(
                p,schedule,first_split,batch,query);
            if constexpr(T::kTaskAllValidVotes) {
                if(partition_allvalid) {
                    quant_wi_tile_body<QuantDecodeAllValidTaskTraits<T>,true>(task,head_tile);
                } else {
                    quant_wi_tile_body<T,true>(task,head_tile);
                }
            } else if constexpr(T::kTaskAllValidGuard) {
                const bool task_allvalid=[&]() {
                    if constexpr(T::kLengthGuardAllowed) {
                        return quant_decode_task_allvalid_full_lengths<T>(p,task);
                    } else {
                        return quant_decode_task_allvalid<T>(p,task);
                    }
                }();
                // Hold flags in registers before body initialization or the FP32
                // output overlay can reuse the shared flag region.
                if(task_allvalid) {
                    quant_wi_tile_body<QuantDecodeAllValidTaskTraits<T>,true>(task,head_tile);
                } else {
                    if constexpr(T::kLengthGuardAllowed) {
                        // The host already proved both cache spans; a failed
                        // live proof keeps every mask in the checked supplier.
                        using CheckedT=std::conditional_t<T::kMixedFP4,
                            QuantDecodeSq1CheckedOffset32FP4Traits,
                            QuantDecodeSq1CheckedU32FP8Traits>;
                        static_assert(std::is_same_v<typename CheckedT::Params,typename T::Params>);
                        static_assert(std::is_same_v<typename CheckedT::SharedMemoryPlan,typename T::SharedMemoryPlan>);
                        static_assert(std::is_same_v<typename CheckedT::SharedMemoryOutPut,typename T::SharedMemoryOutPut>);
                        quant_wi_tile_body<CheckedT,true>(task,head_tile);
                    } else {
                        quant_wi_tile_body<T,true>(task,head_tile);
                    }
                }
            } else {
                // Paired entries, including all FP8 calls, compile no guard.
                quant_wi_tile_body<T,true>(task,head_tile);
            }
            // Every producer and output reader finishes before the shared plan
            // and barrier phases are initialized for the next metadata task.
            __syncthreads();
        }
    }
};

template<typename T>
void launch_sparse_quant_decode_wi_main(Flash_fwd_params &p,hggcStream_t stream) {
    auto kernel=&sparse_quant_wi_kernel<T,true>;
    constexpr size_t bytes=std::max(sizeof(typename T::SharedMemoryPlan),sizeof(typename T::SharedMemoryOutPut));
    static_assert(bytes==262144 && sizeof(typename T::SharedMemoryPlan)==248880);
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel,hggcFuncAttributeMaxDynamicSharedMemorySize,bytes));
    const int m_blocks=cute::ceil_div(p.seqlen_q,T::kBlockM);
    hggcLaunchAttribute attributes[1];
    attributes[0].id=hggcLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed=T::kTaskAllValidVotes?0:1;
    const int64_t z_blocks=T::kSingleTaskCTA?int64_t(p.b)+p.num_sm_parts-1:p.num_sm_parts;
    hggcLaunchConfig_t config={dim3(m_blocks,p.h,unsigned(z_blocks)),dim3(T::NUM_THREADS,1,1),bytes,stream,attributes,1};
    hggcLaunchKernelEx(&config,kernel,p);
    CHECK_CUDA_KERNEL_LAUNCH();
}

__forceinline__ bool quant_decode_fp4_guard_shape(const Flash_fwd_params &p) {
    const int extra_storage=p.extra_topk>=0?p.extra_topk:0;
    return p.topk_len_ptr==nullptr && p.extra_topk_len_ptr==nullptr
        && p.topk>0 && p.topk%64==0 && extra_storage%64==0
        && (int64_t(p.topk)+extra_storage>=2048
            || (p.ngroups==128 && p.topk==128 && extra_storage>=128
                && int64_t(p.topk)+extra_storage>=256));
}

// Bound the last page end without overflowing host arithmetic. Each individual
// field narrowed by the leaf loaders is also within the same 2^31 byte domain.
__forceinline__ bool quant_decode_cache_scope_offset32(
        const void *base,int num_blocks,int page_size,int64_t block_stride,
        int payload_bytes,int scale_bytes) {
    constexpr uint64_t limit=uint64_t(1)<<31;
    if(base==nullptr || num_blocks<=0 || page_size<=0 ||
       (page_size&(page_size-1))!=0 || block_stride<=0) return false;
    const uint64_t page=uint64_t(page_size);
    const uint64_t stride=uint64_t(block_stride);
    const uint64_t row_bytes=uint64_t(payload_bytes)+uint64_t(scale_bytes);
    if(page>limit/row_bytes || stride>limit) return false;
    // Existing token/capacity fields are signed int; do not add a new domain.
    if(uint64_t(num_blocks)>uint64_t(0x7fffffff)/page) return false;
    const uint64_t page_bytes=page*row_bytes;
    const uint64_t last_page=uint64_t(num_blocks-1);
    if(last_page>(limit-page_bytes)/stride) return false;
    const uint64_t max_end=last_page*stride+page_bytes;
    return max_end<=limit;
}

__forceinline__ bool quant_decode_fp4_cache_offset32(const Flash_fwd_params &p) {
    // Main is FP8 (512+16); extra is FP4 (256+32). Prove them independently,
    // retaining their 64-bit base pointers even when their sum exceeds 2^31.
    return quant_decode_cache_scope_offset32(
            p.k_ptr,p.num_blocks,p.page_block_size,p.k_batch_stride,512,16)
        && quant_decode_cache_scope_offset32(
            p.extra_k_ptr,p.extra_num_blocks,p.extra_page_block_size,
            p.extra_k_batch_stride,256,32);
}

__forceinline__ bool quant_decode_direct_grid_gate(const Flash_fwd_params &p) {
    const int64_t tasks=int64_t(p.b)+p.num_sm_parts-1;
    return p.b>0 && p.num_sm_parts>0 && tasks>=1 && tasks<=65535;
}

// End-exclusive bounds are per cache base, including every scale-plane byte.
// Check division before multiplication; no host indices or device sync is used.
__forceinline__ bool quant_decode_fp8_scope_u32_span(
        const void *base,int blocks,int page,int64_t stride) {
    constexpr uint64_t limit=uint64_t(1)<<32;
    if(base==nullptr || blocks<=0 || page<=0 || (page&(page-1))!=0 ||
       stride<=0 || uint64_t(stride)>limit-1) return false;
    const uint64_t page_bytes=uint64_t(page)*(512u+16u);
    if(page_bytes>limit || uint64_t(stride)<page_bytes ||
       int64_t(blocks)*page>2147483647LL) return false;
    return uint64_t(blocks-1)<=(limit-page_bytes)/uint64_t(stride);
}
__forceinline__ bool quant_decode_fp8_scope_u32_gate(const Flash_fwd_params &p) {
    return quant_decode_fp8_scope_u32_span(p.k_ptr,p.num_blocks,p.page_block_size,p.k_batch_stride)
        && (p.extra_topk<=0 || quant_decode_fp8_scope_u32_span(
            p.extra_k_ptr,p.extra_num_blocks,p.extra_page_block_size,p.extra_k_batch_stride));
}

template<bool ExtraFP4>
void launch_sparse_quant_decode_wi(Flash_fwd_params &p,hggcStream_t stream) {
    using PairedT=QuantDecodePairedTraits<ExtraFP4>;
    FLASH_ASSERT(sparse_quant_decode_uses_raw_q_layout(
        true, true, is_sm89_or_newer(), p.d, p.d_v, p.h, p.ngroups,
        p.page_block_size, p.extra_page_block_size));
    FLASH_ASSERT(p.topk%64==0);
    FLASH_ASSERT(sparse_quant_decode_raw_q_alignment_supported(
        p.q_ptr, p.q_batch_stride, p.q_row_stride, p.q_head_stride, p.d));
    const int extra_storage=p.extra_topk>=0?p.extra_topk:0;
    const int64_t storage_total=int64_t(p.topk)+extra_storage;
    const bool sq1_lengths=(p.topk_len_ptr!=nullptr || p.extra_topk_len_ptr!=nullptr)
        && p.h==1 && p.ngroups==128 && p.seqlen_q==p.ngroups && p.topk==128
        && extra_storage>=128 && extra_storage%64==0
        && storage_total>=256
        && quant_decode_direct_grid_gate(p);
    bool checked_address_entry=false;
    if(sq1_lengths) {
        if constexpr(ExtraFP4) checked_address_entry=quant_decode_fp4_cache_offset32(p);
        else checked_address_entry=quant_decode_fp8_scope_u32_gate(p);
    }
    // Only short storage enters the L85 full-row proof (at most64 N32 blocks).
    // Larger storage uses the same checked supplier on the persistent grid.
    const bool length_guard_entry=checked_address_entry && storage_total<=2048;
    if(length_guard_entry) {
        if constexpr(ExtraFP4) {
            launch_sparse_quant_decode_wi_main<QuantDecodeSq1LengthsPrefixFP4Traits>(p,stream);
        } else {
            launch_sparse_quant_decode_wi_main<QuantDecodeSq1LengthsPrefixFP8Traits>(p,stream);
        }
    } else if(checked_address_entry) {
        if constexpr(ExtraFP4) {
            launch_sparse_quant_decode_wi_main<QuantDecodeSq1PersistentOffset32FP4Traits>(p,stream);
        } else {
            launch_sparse_quant_decode_wi_main<QuantDecodeSq1PersistentU32FP8Traits>(p,stream);
        }
    } else if constexpr(ExtraFP4) {
        const bool direct=quant_decode_direct_grid_gate(p);
        if(quant_decode_fp4_guard_shape(p)) {
            const bool narrow=quant_decode_fp4_cache_offset32(p);
            if(direct) {
                if(narrow) {
                    if(p.topk==128 && p.extra_topk>0) {
                        launch_sparse_quant_decode_wi_main<QuantDecodeDirectScopeNarrowNativeFP4WG1PublishFirstRemotePVTraits>(p,stream);
                    } else {
                        launch_sparse_quant_decode_wi_main<QuantDecodeDirectScopeNarrowGuardedTraits>(p,stream);
                    }
                } else {
                    launch_sparse_quant_decode_wi_main<QuantDecodeDirectGuardedPairedTraits>(p,stream);
                }
            } else {
                // Preserve I's checked span selection on the complete legacy grid.
                if(narrow) {
                    launch_sparse_quant_decode_wi_main<QuantDecodeScopeNarrowGuardedTraits>(p,stream);
                } else {
                    launch_sparse_quant_decode_wi_main<QuantDecodeGuardedPairedTraits<true>>(p,stream);
                }
            }
        } else {
            if(direct) {
                launch_sparse_quant_decode_wi_main<QuantDecodeDirectPairedTraits<true>>(p,stream);
            } else {
                launch_sparse_quant_decode_wi_main<PairedT>(p,stream);
            }
        }
    } else {
        if(quant_decode_direct_grid_gate(p)) {
            // Each call checks live rows before consuming the partition votes.
            if(quant_decode_fp4_guard_shape(p) && quant_decode_votes_capacity(p)) {
                quant_decode_allvalid_votes_kernel<<<dim3(p.num_sm_parts),256,0,stream>>>(p);
                CHECK_CUDA_KERNEL_LAUNCH();
                if(quant_decode_fp8_scope_u32_gate(p)) {
                    if(p.topk==128 && p.extra_topk>0) {
                        launch_sparse_quant_decode_wi_main<QuantDecodeDirectVotesScopeU32FP8WG1PublishBoundWG0PostP0Traits>(p,stream);
                    } else {
                        launch_sparse_quant_decode_wi_main<QuantDecodeDirectVotesScopeU32PairedTraits>(p,stream);
                    }
                } else {
                    launch_sparse_quant_decode_wi_main<QuantDecodeDirectVotesPairedTraits<false>>(p,stream);
                }
            } else {
                launch_sparse_quant_decode_wi_main<QuantDecodeDirectPairedTraits<false>>(p,stream);
            }
        } else {
            launch_sparse_quant_decode_wi_main<PairedT>(p,stream);
        }
    }
    // One combine after the selected main entry, with the original sink/LSE contract.
    run_flash_mla_combine_kernel<typename PairedT::InputT>(p,stream);
}

void run_sparse_quant_decode_wi_fp8(Flash_fwd_params &p,hggcStream_t stream) {
    launch_sparse_quant_decode_wi<false>(p,stream);
}
void run_sparse_quant_decode_wi_fp4_extra(Flash_fwd_params &p,hggcStream_t stream) {
    launch_sparse_quant_decode_wi<true>(p,stream);
}

} // namespace flashmla::v41prefill
