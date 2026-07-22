import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

import torch

DISABLE_FP16 = os.getenv("FLASH_MLA_DISABLE_FP16", "FALSE") == "TRUE"
ENABLE_C_DECODE_SPARSE = os.getenv("FLASHMLA_C_ENABLE_DECODE_SPARSE", "TRUE") == "TRUE"
CPP_INFERENCE = 'FLASH_MLA_CPP_INFER_BUILD' in os.environ.keys() and os.environ['FLASH_MLA_CPP_INFER_BUILD'] == "1"

this_dir = os.path.dirname(os.path.abspath(__file__))


def get_sources():
    sources = [
        "csrc/api/api.cpp",
        "csrc/ppu/decode/dense/instantiations/hdim576_512_bf16.cu",
        "csrc/ppu/decode/dense/instantiations/splitkv_mla_bf16.cu",
        "csrc/ppu/prefill/sparse/instantiations/dispatch_bf16.cu",
        "csrc/ppu/decode/sparse/instantiations/hdim576_bf16.cu",
        "csrc/ppu/decode/sparse/instantiations/hdim512_bf16.cu",
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

# ============================================================================
# PPU HGCC Build Extension — compiles .cu files directly via hgcc
# ============================================================================

class HGCCBuildExtension(build_ext):
    """Custom build extension that uses hgcc to compile .cu/.cpp files for PPU."""

    def build_extensions(self):
        if not os.environ.get("MAX_JOBS"):
            os.environ["MAX_JOBS"] = "32"

        for ext in self.extensions:
            self._build_extension_hgcc(ext)

    def _build_extension_hgcc(self, ext):
        import ninja  # noqa: F401

        ppu_sdk = os.environ.get("PPU_SDK", "")
        hgcc = os.path.join(ppu_sdk, "bin", "hgcc")
        torch_dir = torch.__path__[0]

        sources = [os.path.join(this_dir, s) for s in ext.sources]
        include_dirs = [os.path.abspath(d) for d in ext.include_dirs]

        output_dir = os.path.join(self.build_temp, "hgcc_objs")
        os.makedirs(output_dir, exist_ok=True)

        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        torch_include = os.path.join(torch_dir, "include")
        torch_include_csrc = os.path.join(torch_dir, "include", "torch", "csrc", "api", "include")
        python_include = subprocess.check_output(
            [sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('include'))"]
        ).decode().strip()

        all_includes = include_dirs + [torch_include, torch_include_csrc, python_include]
        include_flags = [f"-I{d}" for d in all_includes]

        feature_flags = get_features_args()

        # hgcc flags for .cu device compilation (pure HGGC, no CUDA)
        hgcc_flags = [
            "-O3", "-std=c++17",
            "-arch=ppu_10",
            "-arch=ppu_15",
            "-Xcompiler", "-fPIC",
            "-DSWITCH_TO_HGGCRT",
            "-DUSE_CLANG", "-DUSE_HGGC", "-DUSE_PPU", "-DUSE_AIU=1",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            f"-DTORCH_EXTENSION_NAME={ext.name}",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-mllvm", "-ppu-max-vreg-count=256",
            "-mllvm", "-ppu-sink-matrix-addr=true",
            "-mllvm", "-ppu-max-alloca-byte-size=320",
            "-mllvm", "-ppu-sink-async-addr=true",
            "-mllvm", "-ppu-sink-load-addr=true",
            "-mllvm", "-ppu-sink-store-addr=true",
            "-mllvm", "-ppu-alloca-half-ldst-simplify=true",
        ] + feature_flags

        # c++ flags for .cpp host compilation
        ppu_sdk_inc = os.path.join(ppu_sdk, "include")
        ppu_targets_inc = os.path.join(ppu_sdk, "targets", "x86_64-linux", "include")
        cxx_flags = [
            "-O3", "-std=c++17", "-fPIC",
            "-DUSE_PPU", "-DUSE_AIU=1",
            "-DTORCH_API_INCLUDE_EXTENSION_H", f"-DTORCH_EXTENSION_NAME={ext.name}",
            "-I" + ppu_sdk_inc,
        ] + feature_flags

        # Build ninja file
        max_jobs = int(os.environ.get("MAX_JOBS", "4"))
        ninja_file = os.path.join(output_dir, "build.ninja")
        obj_files = []

        with open(ninja_file, "w") as f:
            f.write("ninja_required_version = 1.3\n\n")

            f.write(f"rule hgcc_compile\n")
            f.write(f"  command = {hgcc} {' '.join(hgcc_flags)} {' '.join(include_flags)} -c $in -o $out\n")
            f.write(f"  description = HGCC $in\n\n")

            cxx_compiler = "c++"
            cxx_include_flags = [f"-I{d}" for d in all_includes]
            cxx_include_flags += [f"-isystem{ppu_sdk_inc}", f"-isystem{ppu_targets_inc}"]
            f.write(f"rule cxx_compile\n")
            f.write(f"  command = {cxx_compiler} {' '.join(cxx_flags)} {' '.join(cxx_include_flags)} -c $in -o $out\n")
            f.write(f"  description = CXX $in\n\n")

            torch_lib_dir = os.path.join(torch_dir, "lib")
            ppu_lib_dir = os.path.join(ppu_sdk, "lib")
            link_libs = f"-L{torch_lib_dir} -L{ppu_lib_dir} -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lc10 -lc10_cuda -lhggc_wrapper -lhg_wrapper"
            f.write(f"rule link\n")
            f.write(f"  command = {hgcc} -shared -o $out $in {link_libs}\n")
            f.write(f"  description = LINK $out\n\n")

            for src in sources:
                basename = os.path.splitext(os.path.basename(src))[0]
                obj = os.path.join(output_dir, basename + ".o")
                obj_files.append(obj)

                if src.endswith(".cu"):
                    f.write(f"build {obj}: hgcc_compile {src}\n")
                else:
                    f.write(f"build {obj}: cxx_compile {src}\n")

            f.write(f"\nbuild {ext_path}: link {' '.join(obj_files)}\n")
            f.write(f"\ndefault {ext_path}\n")

        print(f"\n[HGCCBuildExtension] Building {ext.name} with {len(sources)} sources, max_jobs={max_jobs}")
        subprocess.check_call(
            ["ninja", "-f", ninja_file, f"-j{max_jobs}"],
        )
        print(f"[HGCCBuildExtension] Built {ext_path}")


# ============================================================================
# Extension module definition
# ============================================================================

dir_actlize = this_dir + "/csrc/actlize"
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


ext_modules = []
ext_modules.append(
    Extension(
        name="flash_mla_cuda",
        sources=get_sources(),
        include_dirs=[
            str(Path(this_dir) / "csrc"),
            str(Path(this_dir) / "csrc" / "actlize" / "include"),
            str(Path(this_dir) / "csrc" / "api"),
            str(Path(this_dir) / "csrc" / "kerutils" / "include"),
            str(Path(this_dir) / "csrc" / "ppu"),
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

def custom_local_scheme(version):
    return '+dev%03d.%s' % (version.distance, version.node[:7])

### update FlashMLA Version
### flash_mla-1.0.0 do not support deepseek-v4, with fixed commit-id #f907f433e
### flash_mla-1.0.1 support deepseek-v4 with API Breaking Changes!
def custom_version_scheme(version):
    return '2.0.0'

setup(
    name="flash_mla",
    use_scm_version={
        "local_scheme": custom_local_scheme,
        "version_scheme": custom_version_scheme,
    },
    setup_requires=["setuptools-scm==9.2.2"],
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": HGCCBuildExtension},
)
