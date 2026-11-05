import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
)

DISABLE_FP16 = os.getenv("FLASH_MLA_DISABLE_FP16", "FALSE") == "TRUE"
ENABLE_C_DECODE_SPARSE = os.getenv("FLASHMLA_C_ENABLE_DECODE_SPARSE", "TRUE") == "TRUE"
CPP_INFERENCE = 'FLASH_MLA_CPP_INFER_BUILD' in os.environ.keys() and os.environ['FLASH_MLA_CPP_INFER_BUILD'] == "1"

def append_hgcc_threads(hgcc_extra_args):
    hgcc_threads = os.getenv("NVCC_THREADS") or "32"
    return hgcc_extra_args + ["--threads", hgcc_threads]

def get_sources():
    sources = [
        "csrc/api/api.cpp",
        "csrc/ppu/prefill/sparse/native_fp8.cu",
        "csrc/ppu/prefill/sparse/native_fp8_prefill_ws.cu",
        "csrc/ppu/prefill/sparse/native_fp8_prefill_std.cu",
        "csrc/ppu/decode/sparse/native_fp8.cu",
        "csrc/ppu/decode/sparse/native_fp8_decode_ws.cu",
        "csrc/ppu/decode/sparse/native_fp8_decode_std.cu",
        "csrc/ppu/decode/dense/instantiations/hdim576_512_bf16.cu",
        "csrc/ppu/decode/dense/instantiations/hdim576_512_fp8.cu",
        "csrc/ppu/decode/dense/instantiations/splitkv_mla_bf16.cu",
        "csrc/ppu/prefill/sparse/instantiations/dispatch_bf16.cu",
        "csrc/ppu/decode/sparse/instantiations/hdim576_bf16.cu",
        "csrc/ppu/decode/sparse/instantiations/hdim512_bf16.cu",
        "csrc/ppu/decode/sparse/instantiations/sparse_decode_wg_hs64_bf16.cu",
        "csrc/ppu/decode/sparse/instantiations/sparse_decode_wg_bf16.cu",
        "csrc/ppuxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
        "csrc/ppuxx/decode/combine/instantiations/mla_combine_bf16.cu",
        "csrc/ppu/prefill/sparse/instantiations/wg_bf16_sm80.cu",
        "csrc/ppu/prefill/sparse/instantiations/wg_bf16_sm89.cu",
    ]

    if not DISABLE_FP16:
        sources.append("csrc/ppu/decode/dense/instantiations/hdim576_512_fp16.cu")
        sources.append("csrc/ppu/decode/dense/instantiations/splitkv_mla_fp16.cu")
        sources.append("csrc/ppuxx/decode/combine/instantiations/mla_combine_fp16.cu")
        sources.append("csrc/ppu/prefill/sparse/instantiations/wg_fp16_sm80.cu")
        sources.append("csrc/ppu/prefill/sparse/instantiations/wg_fp16_sm89.cu")

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
# subprocess.run(["git", "submodule", "update", "--init", "csrc/actlize"])  # disabled: no git
dir_actlize = this_dir +  "/csrc/actlize"
# check the existence of actlize
if not os.path.exists(dir_actlize):
    try:
        repo_actlize = os.path.dirname(this_dir) + "/actlize"
        if not os.path.exists(repo_actlize):
            raise RuntimeError(
                f"actlize does not exist: actlize must be fetched in advance as:\n"
                f" \"{repo_actlize}\" or \"{dir_actlize}\""
            )
        else:
            os.symlink(repo_actlize, dir_actlize)
    except Exception as e:
        raise EnvironmentError("setup dependencies FAILED: " + repr(e))

cc_flag = []
cc_flag.append("-arch=ppu_10")
cc_flag.append("-arch=ppu_15")

cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla_cuda",
        sources=get_sources(),
        libraries=['hggc'],
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": append_hgcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-DNDEBUG",
                    "-D_USE_MATH_DEFINES",
                    "-Wno-deprecated-declarations",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--resource-usage",
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
                    "-DACOMPUTE_VERSION=10000"
                ]
                + cc_flag
            ) + get_features_args(),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "actlize" / "include",
            Path(this_dir) / "csrc" / "api",
            Path(this_dir) / "csrc" / "kerutils" / "include",
            Path(this_dir) / "csrc" / "ppu",
        ],
    )
)


FLASHMLA_VERSION = '2.1.0'  # FlashMLA version
RELEASE_VERSION = 'v1.0.0'  # Release version


def get_ppu_sdk_version():
    # Query version macros using the selected SDK compiler, falling back to PATH.
    sdk_root = os.environ.get('PPU_SDK') or os.environ.get('PPU_HOME') or '/usr/local/PPU_SDK'
    compiler = os.path.join(sdk_root, 'bin', 'hgcc')
    if not os.path.isfile(compiler):
        compiler = shutil.which('hgcc')
    if not compiler:
        return (0, 0, 0)
    try:
        output = subprocess.check_output(
            [compiler, '-E', '-x', 'cu', '/dev/stdin', '-o', '/dev/stdout'],
            input=(
                'PPU_VERSION __HGGCCC_VER_MAJOR__ '
                '__HGGCCC_VER_MINOR__ __HGGCCC_VER_BUILD__\n'
            ),
            text=True,
            timeout=10,
        )
        major, minor, build = map(int, output.rsplit('PPU_VERSION', 1)[1].split())
        return (major, minor, build)
    except Exception:
        pass
    return (0, 0, 0)


def get_package_version(ppu_sdk_version):
    try:
        from setuptools_scm import Configuration
        from setuptools_scm.git import parse
    except ImportError as error:
        raise RuntimeError('Missing build dependency: install setuptools-scm first') from error

    release_tag = f'{FLASHMLA_VERSION}+{RELEASE_VERSION}'
    # Match only this release tag; uncommitted changes never affect the version.
    scm = parse(this_dir, Configuration(), describe_command=[
        'git', 'describe', '--tags', '--long', '--match', release_tag,
    ])
    if scm is None or scm.node is None:
        raise RuntimeError('Git commit information unavailable; build from a Git repository with commit history')
    if ppu_sdk_version == (0, 0, 0):
        raise RuntimeError('PPU SDK version unavailable; check hgcc --version for the selected SDK')

    ppu_version = '.'.join(map(str, ppu_sdk_version))
    local_version = f'{RELEASE_VERSION}.ppu{ppu_version}'
    if scm.distance in (None, 0):
        return f'{FLASHMLA_VERSION}+{local_version}'
    return f'{FLASHMLA_VERSION}.dev{scm.distance}+{local_version}.{scm.node[:9]}'


package_version = get_package_version(get_ppu_sdk_version())

setup(
    name="flash_mla",
    version=package_version,
    setup_requires=["setuptools-scm"],
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
