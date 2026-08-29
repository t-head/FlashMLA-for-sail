import os
from datetime import datetime
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


DISABLE_FP16 = os.getenv("FLASH_MLA_DISABLE_FP16", "FALSE") == "TRUE"
ENABLE_C_DECODE_SPARSE = (
    os.getenv("FLASHMLA_C_ENABLE_DECODE_SPARSE", "TRUE") == "TRUE"
)
CPP_INFERENCE = os.getenv("FLASH_MLA_CPP_INFER_BUILD") == "1"
SPARSE_PREFILL_ONLY = os.getenv("FLASH_MLA_SPARSE_PREFILL_ONLY") == "1"


def append_hgcc_threads(hgcc_extra_args):
    hgcc_threads = os.getenv("NVCC_THREADS") or "32"
    return hgcc_extra_args + ["--threads", hgcc_threads]


def get_sources():
    if SPARSE_PREFILL_ONLY:
        return [
            "csrc/sparse_prefill_api.cpp",
            "csrc/flash_fwd_sparse_prefill_hdim576_512_bf16_sm80.cu",
        ]
    sources = [
        "csrc/flash_api.cpp",
        "csrc/flash_fwd_split_hdim576_512_bf16_sm80.cu",
        "csrc/flash_fwd_sparse_prefill_hdim576_512_bf16_sm80.cu",
        "csrc/flash_fwd_mla_metadata.cu",
        "csrc/flash_splitkv/mla_combine.cu",
        "csrc/flash_splitkv/splitkv_mla.cu",
        "csrc/flash_splitkv/splitkv_dsa.cu",
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
    if SPARSE_PREFILL_ONLY:
        features_args.append("-DFLASH_MLA_SPARSE_PREFILL_ONLY")
    features_args.extend(["-DFLASH_MLA_STANDALONE_BUILD", "-DUSE_TS"])
    return features_args


this_dir = os.path.dirname(os.path.abspath(__file__))
dir_actlize = os.path.join(this_dir, "csrc", "actlize")
if not os.path.exists(dir_actlize):
    repo_actlize = os.path.join(os.path.dirname(this_dir), "actlize")
    if not os.path.exists(repo_actlize):
        raise EnvironmentError(
            "setup dependencies FAILED: actlize must be fetched at "
            f"{repo_actlize} or {dir_actlize}"
        )
    os.symlink(repo_actlize, dir_actlize)


arch_flags = ["-arch=ppu_10", "-arch=ppu_15"]
cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]
hgcc_args = append_hgcc_threads(
    [
        "-O3",
        "-std=c++17",
        "-DNDEBUG",
        "-D_USE_MATH_DEFINES",
        "-Wno-deprecated-declarations",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
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
    ]
    + arch_flags
)

include_dirs = [
    Path(this_dir) / "csrc",
    Path(this_dir) / "csrc" / "actlize" / "include",
]

# Some runtime images split the compiler-owned CUDA/HGGC headers across
# CUDA_SDK/targets and PPU_SDK/targets.  Adding the entire latter tree changes
# which hggc_runtime.h nvcc force-includes and creates CUDA/HGGC math-header
# conflicts.  A builder may instead provide a narrow compatibility directory
# containing only the missing HGGC datatype headers.
hggc_shim_include = os.getenv("FLASH_MLA_HGGC_SHIM_INCLUDE")
if hggc_shim_include:
    include_dirs.append(Path(hggc_shim_include))


ext_modules = [
    CUDAExtension(
        name="flash_mla_cuda",
        sources=get_sources(),
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": hgcc_args + get_features_args(),
        },
        include_dirs=include_dirs,
    )
]


try:
    rev = "+" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
except Exception:
    rev = "+dev"


def custom_local_scheme(version):
    return "+dev%03d.%s" % (version.distance, version.node[:7])


def custom_version_scheme(version):
    return "2.0.0"


setup(
    name="flash_mla",
    use_scm_version={
        "local_scheme": custom_local_scheme,
        "version_scheme": custom_version_scheme,
    },
    setup_requires=["setuptools-scm==9.2.2"],
    packages=find_packages(include=["flash_mla"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
