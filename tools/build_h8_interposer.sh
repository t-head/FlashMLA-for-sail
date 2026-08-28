#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${FLASH_MLA_H8_BUILD_DIR:-"${repo_dir}/build/h8-interposer"}
ppu_sdk=${PPU_SDK:-/usr/local/PPU_SDK}
hgcc=${ppu_sdk}/bin/hgcc
source_file=${repo_dir}/csrc/ppu/prefill/sparse/instantiations/interpose_bf16.cu
object_file=${build_dir}/interpose_bf16.o
output_file=${build_dir}/libflashmla_h8_interpose.so
target_so=${FLASH_MLA_TARGET_SO:-}
expected_symbol=_Z31run_sparse_prefill_fwd_dispatchIN7cutlass10bfloat16_tEEvR19SparsePrefillParams

if [[ ! -x "${hgcc}" ]]; then
    echo "hgcc not found: ${hgcc}" >&2
    exit 2
fi

if [[ -n "${target_so}" ]]; then
    if [[ ! -f "${target_so}" ]]; then
        echo "FLASH_MLA_TARGET_SO does not exist: ${target_so}" >&2
        exit 2
    fi
    if ! nm -D "${target_so}" | awk -v symbol="${expected_symbol}" '$3 == symbol && $2 == "W" { found=1 } END { exit !found }'; then
        echo "target FlashMLA does not expose the expected weak dispatch ABI" >&2
        exit 2
    fi
fi

mkdir -p "${build_dir}"

"${hgcc}" \
    -O3 -std=c++17 -Xcompiler -fPIC \
    -arch=ppu_10 \
    -DSWITCH_TO_HGGCRT -DUSE_CLANG -DUSE_HGGC -DUSE_PPU -DUSE_AIU=1 \
    -DACOMPUTE_VERSION=10000 \
    --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
    -mllvm -ppu-max-vreg-count=256 \
    -mllvm -ppu-sink-matrix-addr=true \
    -mllvm -ppu-max-alloca-byte-size=320 \
    -mllvm -ppu-sink-async-addr=true \
    -mllvm -ppu-sink-load-addr=true \
    -mllvm -ppu-sink-store-addr=true \
    -mllvm -ppu-alloca-half-ldst-simplify=true \
    -mllvm -ppu-force-warpage=true \
    -mllvm -ppu-force-vregrr=true \
    -I"${repo_dir}/csrc" \
    -I"${repo_dir}/csrc/actlize/include" \
    -I"${repo_dir}/csrc/kerutils/include" \
    -I"${repo_dir}/csrc/ppu" \
    -c "${source_file}" -o "${object_file}"

"${hgcc}" -shared -o "${output_file}" "${object_file}" -ldl

if ! nm -D "${output_file}" | awk -v symbol="${expected_symbol}" '$3 == symbol && $2 == "T" { found=1 } END { exit !found }'; then
    echo "interposer did not export the expected strong dispatch ABI" >&2
    exit 3
fi

echo "${output_file}"
