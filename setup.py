import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
)

DISABLE_FP16 = os.getenv("FLASH_MLA_DISABLE_FP16", "FALSE") == "TRUE"
ENABLE_C_DECODE_SPARSE = os.getenv("FLASHMLA_C_ENABLE_DECODE_SPARSE", "TRUE") == "TRUE"
CPP_INFERENCE = 'FLASH_MLA_CPP_INFER_BUILD' in os.environ.keys() and os.environ['FLASH_MLA_CPP_INFER_BUILD'] == "1"

def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def get_sources():
    sources = [
        "csrc/flash_api.cpp",
        "csrc/flash_fwd_split_hdim576_512_bf16_sm80.cu",
        "csrc/flash_fwd_sparse_prefill_hdim576_512_bf16_sm80.cu",
        "csrc/flash_fwd_mla_metadata.cu",
        "csrc/flash_splitkv/mla_combine.cu",
        "csrc/flash_splitkv/splitkv_mla.cu",
    ]

    if not DISABLE_FP16:
        sources.append("csrc/flash_fwd_split_hdim576_512_fp16_sm80.cu")

    return sources


def get_features_args():
    features_args = []
    if DISABLE_FP16:
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    if ENABLE_C_DECODE_SPARSE:
        features_args.append("-DFLASHMLA_C_ENABLE_DECODE_SPARSE")
    if CPP_INFERENCE:
        features_args.append("-DFLASH_MLA_CPP_INFER_BUILD")
    features_args.append("-DFLASH_MLA_STANDALONE_BUILD")
    features_args.append("-DUSE_TS")

    return features_args

this_dir = os.path.dirname(os.path.abspath(__file__))
# subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])
dir_cutlass3 = this_dir +  "/csrc/cutlass3"
# check the existence of cutlass3
if not os.path.exists(dir_cutlass3):
    try:
        repo_cutlass3 = os.path.dirname(this_dir) + "/cutlass3"
        if not os.path.exists(repo_cutlass3):
            raise RuntimeError(
                f"cutlass3 does not exist: cutlass3 must be fetched in advance as:\n"
                f" \"{repo_cutlass3}\" or \"{dir_cutlass3}\""
            )
        else:
            os.symlink(repo_cutlass3, dir_cutlass3)
    except Exception as e:
        raise EnvironmentError("setup dependencies FAILED: " + repr(e))

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")

cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla_cuda",
        sources=get_sources(),
        libraries=['cuda'],
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-DNDEBUG",
                    "-D_USE_MATH_DEFINES",
                    "-Wno-deprecated-declarations",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v,--register-usage-level=10",
                    "-mllvm",
                    "-ppu-max-vreg-count=256",
                    "-mllvm",
                    "-ppu-sink-matrix-addr=true",
                    "-mllvm",
                    "-ppu-max-alloca-byte-size=320",
                    "-mllvm",
                    "-ppu-sink-async-addr=true",
                    "-mllvm",
                    "-ppu-sink-load-addr=true",
                    "-mllvm",
                    "-ppu-sink-store-addr=true",
                    "-mllvm",
                    "-ppu-alloca-half-ldst-simplify=true",
                    "-mllvm",
                    "-ppu-force-warpage=true",
                    "-mllvm",
                    "-ppu-force-vregrr=true",
                    "-DUSE_PPU",
                    "-DUSE_AIU=1",
                    "-DACOMPUTE_VERSION=10000",
                    "-mllvm",
                    "-ppu-simt-branch=false",
                    "-mllvm",
                    "-ppu-disable-licm=true"
                    # "-mllvm",
                    # "-ppu-indvars-instr-sink-ctrl=true
                ]
                + cc_flag
            ) + get_features_args(),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "cutlass3" / "include",
        ],
    )
)


try:
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    rev = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
except Exception as _:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    rev = '+' + date_time_str


setup(
    name="flash_mla",
    version="1.0.0",
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
