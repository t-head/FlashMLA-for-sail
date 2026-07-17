import argparse
import os
import torch
from utils import run_fa_cycle_on_device, str_to_list, worker, split_list_into_groups, read_cmds_from_file
import multiprocessing as mp

device_name = torch.cuda.get_device_name()
USE_PPU = (device_name.lower().find("ppu") != -1) or (device_name.lower().find("zw") != -1)
if not any(k in device_name.lower() for k in ['ppu', 'zw','nvidia']):
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
    parser.add_argument('--consecutive', default=False, action="store_true", required=False, help='specify if run_flahs_mla.py with caselist')
    parser.add_argument('--backend', default="flash_mla", type=str, required=False, help='specify backend, all, flash_mla, flash_infer, flash_mla_triton')
    parser.add_argument('--mode', default="metrics", type=str, choices=['metrics', 'full'], required=False, help='specify if run full ncu')
    parser.add_argument('--device', default=None, type=str, required=False, help='specify which device to run. 0 means gpu0. 0,3 means gpu0,1,2,3')

    args = parser.parse_args()
    fa_cases = list()
    if args.string:
        fa_cases = [args.string]
    elif args.format:
        fa_cases = [args.format]
    elif args.caselist:
        # Use the shared loader so dsa.list / nested .list / tqdm-progress
        # noise is handled the same way as in run_flash_mla.py.
        fa_cases = read_cmds_from_file(args.caselist)
    else:
        print("Must give a string a format or a caselist file!")
        exit(-1)
    if args.device == None:
        if args.backend == "all":
            for backend in ['flash_mla', 'flash_infer', 'flash_mla_triton'] :
                run_fa_cycle_on_device(fa_cases, args.output, "ppu" if USE_PPU else "gpu", args.local, backend, args)
        else:
            run_fa_cycle_on_device(fa_cases, args.output, "ppu" if USE_PPU else "gpu", args.local, args.backend, args)
    else:
        devices = str_to_list(args.device)
        if len(devices) == 1 or len(devices) > 2:
            num_gpus = devices
        elif len(devices) == 2:
            num_gpus = [i for i in range(devices[0], devices[1] + 1)]
        else:
            num_gpus = [0]
        processes = []
        fa_cases_groups = split_list_into_groups(fa_cases, len(num_gpus))
        for i in range(len(num_gpus)):
            # 创建子进程并传递 GPU ID, 在worker中循环 backend的取值
            p = mp.Process(target=worker, args=(num_gpus[i], fa_cases_groups[i], args.output, "ppu" if USE_PPU else "gpu", args.local, args.backend, args))
            p.start()
            processes.append(p)
