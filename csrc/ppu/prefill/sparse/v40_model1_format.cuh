#pragma once
// MODEL1 V4.0: 448 E4M3 values + 64 raw BF16 RoPE values, 7 UE8M0 scales.
namespace flashmla::v41prefill {
__forceinline__ __device__ uint32_t v41_prefill_e8m0_scale2_bits(uint8_t raw);
struct V40Fp8RegisterPacket {
    uint32_t x0,y0,z0,w0,x1,y1,z1,w1;
    uint8_t scale_raw0,scale_raw1;
    uint32_t x2,y2,z2,w2;
};
static_assert(sizeof(V40Fp8RegisterPacket)==52);
template<int S,int E,typename T>
__forceinline__ __device__ V40Fp8RegisterPacket v40_prefill_register_load(
    const SparseQuantPrefillParams &p,int tidx,int token_idx,bool valid) {
    static_assert(T::kV40Cache && !T::kFP4Cache && E-S==4 && (S==0 || S==4));
    V40Fp8RegisterPacket packet{};
    if(!valid) return packet;
    const int lane=tidx&7;
    const int page=token_idx>>__builtin_ctz(p.quant_page_size);
    const int off=token_idx&(p.quant_page_size-1);
#if defined(FLASHMLA_V41_QUANT_ADDRESS32)
    const auto *base=reinterpret_cast<const uint8_t*>(p.kv)+uint32_t(page)*uint32_t(p.quant_block_stride);
    const auto *data=base+uint32_t(off)*576;
    const auto *scale=base+uint32_t(p.quant_page_size)*576+uint32_t(off)*8;
#else
    const auto *base=reinterpret_cast<const uint8_t*>(p.kv)+int64_t(page)*p.quant_block_stride;
    const auto *data=base+int64_t(off)*576;
    const auto *scale=base+int64_t(p.quant_page_size)*576+int64_t(off)*8;
#endif
    const uint4 g0=__ldg(reinterpret_cast<const uint4*>(data+S*64+lane*16));
    packet.x0=g0.x;packet.y0=g0.y;packet.z0=g0.z;packet.w0=g0.w;
    packet.scale_raw0=__ldg(scale+S+(lane>>2));
    if constexpr(S==4) {
        if(lane>=4) {
            const auto *rope=data+448+(lane-4)*32;
            const uint4 g1=__ldg(reinterpret_cast<const uint4*>(rope));
            const uint4 g2=__ldg(reinterpret_cast<const uint4*>(rope+16));
            packet.x1=g1.x;packet.y1=g1.y;packet.z1=g1.z;packet.w1=g1.w;
            packet.x2=g2.x;packet.y2=g2.y;packet.z2=g2.z;packet.w2=g2.w;
        } else {
            const uint4 g1=__ldg(reinterpret_cast<const uint4*>(data+(S+2)*64+lane*16));
            packet.x1=g1.x;packet.y1=g1.y;packet.z1=g1.z;packet.w1=g1.w;
            packet.scale_raw1=__ldg(scale+S+2+(lane>>2));
        }
    } else {
        const uint4 g1=__ldg(reinterpret_cast<const uint4*>(data+(S+2)*64+lane*16));
        packet.x1=g1.x;packet.y1=g1.y;packet.z1=g1.z;packet.w1=g1.w;
        packet.scale_raw1=__ldg(scale+S+2+(lane>>2));
    }
    return packet;
}
template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
__forceinline__ __device__ void v40_prefill_register_finish(
    RawK &raw_sK,int tidx,bool valid,const V40Fp8RegisterPacket &packet) {
    static_assert(T::kV40Cache && E-S==4);
    const int row=tidx/8,lane=tidx&7;
    const int paired_base=(S/2)*4096+(row>>4)*2048+(row&15)*64
        +(lane>>2)*1024+(((lane&3)^((row>>1)&3))*16);
    const int first_chunk=paired_base+(row&1)*8;
    const int second_chunk=paired_base+((row&1)^1)*8;
    auto *dst=cute::raw_pointer_cast(raw_sK.data());
    if constexpr(FRONT_DRAIN) {
        cutlass::arch::NamedBarrier::sync(256,
            static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
    }
    #pragma unroll
    for(int group=0;group<2;++group) {
        if(S==4 && group==1 && lane>=4) {
            // Exact BF16 bits including signs/zeros/NaN payload: no FP8 conversion/scale.
            const int pos0=first_chunk+group*4096,pos1=second_chunk+group*4096;
            *reinterpret_cast<uint4*>(dst+pos0)=uint4{packet.x1,packet.y1,packet.z1,packet.w1};
            *reinterpret_cast<uint4*>(dst+pos1)=uint4{packet.x2,packet.y2,packet.z2,packet.w2};
        } else {
            const uint4 packed=group==0?uint4{packet.x0,packet.y0,packet.z0,packet.w0}:
                                       uint4{packet.x1,packet.y1,packet.z1,packet.w1};
            const uint8_t raw=group==0?packet.scale_raw0:packet.scale_raw1;
            const uint32_t scale_bits2=valid?v41_prefill_e8m0_scale2_bits(raw):0;
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
#if defined(FLASHMLA_V41_QUANT_DECODED_STORE_RELEASE)
    __threadfence_block();
    asm volatile("" ::: "memory");
#else
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
#endif
}
template<int S,int E,typename T,bool FRONT_DRAIN,typename RawK>
__forceinline__ __device__ void v40_prefill_load_fp8(
    const SparseQuantPrefillParams &p,RawK &raw_sK,int tidx,int token_idx,bool valid) {
    if constexpr(FRONT_DRAIN) {
        cutlass::arch::NamedBarrier::sync(256,
            static_cast<cutlass::arch::ReservedNamedBarriers>(1+(threadIdx.x>>8)));
    }
    const auto packet=v40_prefill_register_load<S,E,T>(p,tidx,token_idx,valid);
    v40_prefill_register_finish<S,E,T,false>(raw_sK,tidx,valid,packet);
}
} // namespace flashmla::v41prefill
