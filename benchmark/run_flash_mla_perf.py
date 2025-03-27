import argparse
import os
import torch
from utils import run_fa_cycle_on_device

device_name = torch.cuda.get_device_name()
USE_PPU = (device_name.lower().find("ppu") != -1)
if not any(k in device_name.lower() for k in ['ppu','nvidia']):
    print("Warning: Unrecognized device name: "+ device_name)

if USE_PPU:
    # os.environ['HGGC_PROFILE_MODE'] = '4'
    os.environ['HGGC_RESET_CACHE'] = '1'
    os.environ['ALIPPU_RESET_CE_MASK'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Performance Testing for FA with format or list.')
    parser.add_argument('--caselist', default=None, type=str, required=False, help='the list of FA cases')
    parser.add_argument('--string', default=None, type=str, required=False, help='the string of FA cases')
    parser.add_argument('--format', default=None, type=str, required=False, help='the string of FA cases')
    parser.add_argument('--output', default="output", type=str, required=False, help='the output storing cycles of FA cases')
    parser.add_argument('--local', default=False, action="store_true", required=False, help='specify if run local')
    parser.add_argument('--backend', default="flash_mla", type=str, required=False, help='specify backend, all, flash_mla, flash_infer, flash_mla_triton')

    args = parser.parse_args()
    fa_cases = list()
    if args.string:
        fa_cases = [args.string]
    elif args.format:
        fa_cases = [args.format]
    elif args.caselist:
        with open(args.caselist, "r") as f:
          lines = f.readlines()
          for line in lines:
              if line.strip() != "" and not line.startswith("#"):
                fa_cases.append(line.strip())
    else:
        print("Must give a string a format or a caselist file!")
        exit(-1)

    if args.backend == "all":
        for backend in ['flash_mla', 'flash_infer', 'flash_mla_triton'] :
            run_fa_cycle_on_device(fa_cases, args.output, "ppu" if USE_PPU else "gpu", args.local, backend)
    else:
        run_fa_cycle_on_device(fa_cases, args.output, "ppu" if USE_PPU else "gpu", args.local, args.backend)