import os
import csv
import subprocess
import re
import time
import torch
device_name = torch.cuda.get_device_name()
dev_name_mapping = {
    "PPU-ZW810E": "810e", "PPU-ZW810": "810", "PPU-ZW610": "610",
    "ZW-M890P": "890P", "ZW-M890L": "890L", "ZW-M530": "530", 
}


def run_cmd(cmd: str, timeout=3600, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    print(f"Run command: {cmd}, timeout: {timeout}")
    ret = subprocess.run(args=cmd, timeout=timeout, shell=True, stdout=stdout, stderr=stderr, encoding="utf-8")
    if stdout:
        for line in ret.stdout.splitlines() + ret.stderr.splitlines():
            print(line)
    if ret.returncode != 0:
        print(f"Run command failed!")
    else:
        print(f"Run command succeed!")
    return ret

def str_to_list(s, type_func=int):
    """Convert a comma-separated string to a list of a specified type."""
    return [type_func(i.strip()) for i in s.split(',')]

def split_list_into_groups(lst, num):
    group_size = len(lst) // num
    remainder = len(lst) % num
    start = 0
    groups = []
    for i in range(num):
        groups.append([])
    for i in range(len(lst)):
        group_idx = i % num
        groups[group_idx].append(lst[i])
    return groups

def worker(gpu_id, fa_cases, output, device, is_local, backend, args):
    # 设置当前进程可见的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process {os.getpid()} is running on GPU {gpu_id}")
    if backend == "all":
        for _backend in ['flash_mla', 'flash_infer', 'flash_mla_triton']:
            run_fa_cycle_on_device(fa_cases, output, device, is_local, _backend, args)
    else:
        run_fa_cycle_on_device(fa_cases, output, device, is_local, backend, args)

# devices = {
#     "name": ["cycle", "tensor core efficiency", "waves"],
#     "gpu":  ["sm__cycles_elapsed.max", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "launch__waves_per_multiprocessor"],
#     "ppu":  ["ce__cycles_elapsed.max", "cu__inst_executed_pipe_tensor_fp16.avg.pct_of_peak_sustained_active", "launch__waves_per_cu"],
# }

def read_cycle_from_nculog(filename):
    kernel_pattern = r"(.*)kernel(.*)Device(.*)"
    duration_pattern = "__time_duration.sum"
    cycles_pattern = "__cycles_elapsed.max"
    tc_pattern = "_tensor_(.*)avg.pct_of_peak_sustained" # cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
    # "dram__throughput.avg.pct_of_peak_sustained_elapsed" or "ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
    l2_pattern = "hit_rate.pct"
    hbm_pattern = "throughput.avg.pct_of_peak_sustained_elapsed"

    kernel_list = []
    duration_list = []
    cycles_list = []
    tc_list = []
    l2_list = []
    hbm_list = []

    with open(filename, newline='') as log_file:
        for line in log_file.read().split("\n"):
            if re.search(kernel_pattern, line, re.IGNORECASE):
                kernel_list.append(line.strip())
            if re.search(cycles_pattern, line):
                cycles_list.append(int(line.strip().split()[-1]))
            if re.search(tc_pattern, line):
                tc_list.append(float(line.strip().split()[-1]))
            if re.search(l2_pattern, line):
                l2_list.append(float(line.strip().split()[-1]))
            if re.search(hbm_pattern, line):
                hbm_list.append(float(line.strip().split()[-1]))
            if re.search(duration_pattern, line): # convert to us
                fwd_time = float(line.strip().split()[-1])
                if "ns" in line: # ppu return ns, gpu return us
                    fwd_time = fwd_time/1000
                duration_list.append(round(fwd_time,4))

    assert len(kernel_list) == len(cycles_list), f"{len(kernel_list)}, {len(cycles_list)}"
    assert len(kernel_list) == len(tc_list), f"{len(kernel_list)}, {len(tc_list)}"
    assert(len(kernel_list) == len(l2_list))
    assert(len(kernel_list) == len(hbm_list))
    assert(len(kernel_list) == len(duration_list))


    op_cycles = dict()
    fwd_cycle = 0
    fwd_cycle_sum = 0 # sum of 3 kernels
    duration = 0
    fwd_tc = 0
    fwd_l2 = 0
    fwd_hbm = 0

    for i in range(len(kernel_list)):
        op = kernel_list[i]
        cycle = cycles_list[i]
        op_cycles[op] = [cycles_list[i], tc_list[i], l2_list[i], hbm_list[i], duration_list[i]]
        assert ("fwd" in op.lower() or "mla" in op.lower())
        if "fwd" in op.lower() or "mla" in op.lower(): # flashmla ppu / triton / flashinfer
            fwd_cycle_sum += cycle
        if "sparse" in op.lower() or "splitkv_mla_kernel" in op.lower():
            # only sum: flash_fwd_splitkv_mla_kernel / flash_sparse_decode_fwd_kernel / flash_sparse_prefill_fwd_kernel
            fwd_cycle = cycles_list[i]
            fwd_tc = tc_list[i]
            fwd_l2 = l2_list[i]
            fwd_hbm = hbm_list[i]
            duration = duration_list[i]

    # calculate statistics data
    if fwd_cycle_sum != 0:
        # fwd unit case
        return fwd_cycle_sum, fwd_tc, fwd_l2, fwd_hbm, fwd_cycle, duration, op_cycles
    else:
        print("Not valid CSV file!")
        return 0, 0, 0, 0, 0, 0, []
        #exit(-1)

def batch_read_cycle_from_nculog(filename, fa_cases):
    '''
    type1: get_mla_metadata_kernel, flash_sparse_decode_fwd_kernel/flash_fwd_splitkv_mla_kernel, mla_combine_kernel
    type2: flash_sparse_prefill_fwd_kernel
    '''
    kernel_pattern = r"(.*)kernel(.*)Device(.*)"
    duration_pattern = "__time_duration.sum"
    cycles_pattern = "__cycles_elapsed.max"
    tc_pattern = "_tensor_(.*)avg.pct_of_peak_sustained" # cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
    # "dram__throughput.avg.pct_of_peak_sustained_elapsed" or "ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
    l2_pattern = "hit_rate.pct"
    hbm_pattern = "throughput.avg.pct_of_peak_sustained_elapsed"

    kernel_list = []
    duration_list = []
    cycles_list = []
    tc_list = []
    l2_list = []
    hbm_list = []

    with open(filename, newline='') as log_file:
        for line in log_file.read().split("\n"):
            if re.search(kernel_pattern, line, re.IGNORECASE):
                kernel_list.append(line.strip())
            if re.search(cycles_pattern, line):
                cycles_list.append(int(line.strip().split()[-1]))
            if re.search(tc_pattern, line):
                tc_list.append(float(line.strip().split()[-1]))
            if re.search(l2_pattern, line):
                l2_list.append(float(line.strip().split()[-1]))
            if re.search(hbm_pattern, line):
                hbm_list.append(float(line.strip().split()[-1]))
            if re.search(duration_pattern, line):
                fwd_time = float(line.strip().split()[-1])
                if "ns" in line: # ppu return ns, gpu return us
                    fwd_time = fwd_time/1000
                duration_list.append(round(fwd_time,4))

    assert(len(kernel_list) == len(cycles_list))
    assert(len(kernel_list) == len(tc_list))
    assert(len(kernel_list) == len(l2_list))
    assert(len(kernel_list) == len(hbm_list))
    assert(len(kernel_list) == len(duration_list))

    output_list = list()
    start_flag, end_flag = False, False

    j = 0
    for i in range(len(kernel_list)):
        op = kernel_list[i]
        cycle = cycles_list[i]
        # op_cycles[op] = [cycles_list[i], tc_list[i], l2_list[i], hbm_list[i], duration_list[i]]
        assert ("fwd" in op.lower() or "mla" in op.lower())
        # find start flag
        if "get_mla_metadata_kernel" in op.lower() or "flash_sparse_prefill_fwd_kernel" in op.lower():
            # new case start
            # op_cycles = dict()
            fwd_cycle = 0
            fwd_cycle_sum = 0 # sum of 3 kernels
            duration = 0
            fwd_tc = 0
            fwd_l2 = 0
            fwd_hbm = 0
            extra_duration = 0
        if "fwd" in op.lower(): # flashmla ppu / triton / flashinfer
            fwd_cycle_sum += cycle
            duration += duration_list[i]
        elif "meta" in op.lower():
            extra_duration = duration_list[i]

        if "sparse" in op.lower() or "splitkv_mla_kernel" in op.lower():
            # only sum: flash_fwd_splitkv_mla_kernel / flash_sparse_decode_fwd_kernel / flash_sparse_prefill_fwd_kernel
            fwd_cycle = cycles_list[i]
            fwd_tc = tc_list[i]
            fwd_l2 = l2_list[i]
            fwd_hbm = hbm_list[i]
            # duration = duration_list[i]
        if "mla_combine_kernel" in op.lower() or "flash_sparse_prefill_fwd_kernel" in op.lower():
            # a case end
            output_list.append({
                "case name": fa_cases[j].strip(),
                "op time in us": duration,
                "pmu cycles": fwd_cycle_sum,
                "pmu flops util": fwd_tc,
                "pmu hbm util": fwd_hbm,
                "pmu l2 util": fwd_l2,
                "get_mla_metadata_kernel": extra_duration,
            })
            j += 1
    # return statistics data
    return output_list

def clean_casename(name):
    _need_replace = ['--', '=', 'format', 'Formatted', '[', ']', ":", "*", " ", ","]
    # for item in _need_replace:
    #     name = name.replace(item, "_")
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    while "__" in name:
        name = name.replace("__", "_")
    return name

def run_fa_cycle_on_device(fa_cases, output_file, dev="gpu", run_local=False, backend="flash_mla", args=None):
    mode = args.mode
    consecutive = args.consecutive
    output_lines = list()
    # headers = ["casename","cycle","tc efficiency", "hbm efficiency", "cmd", "detail"]
    headers = ["casename","cycles(sm__cycles_elapsed.max)","tc_pct(TensorCore效率%)","L2_hit_pct(L2命中率%)","hbm_pct(HBM带宽效率%)","inner_cycles","duration_us(gpu__time_duration.sum)","ncu_cmd","detail"]
    # new_row=["casename"]  metrics.get("name", [])  ["detail"]
    # output_lines.append(new_row)
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    kernel_pattern="-k 'regex:fwd*|mla*'"
    if consecutive:
        timestamp = str(round(time.time() * 1000))
        log_file = f"./logs/gpu_cycles_single_case_{timestamp}.log"
        if not args.caselist:
            print("must give a caselist if run consecutively")
            exit(1)
        # run by caselist
        if mode == "full":
            output_name = clean_casename(case) + backend
            cmd = '{} --set full {} -o {} python ./run_flash_mla.py --backend={} --caselist="{}" \
                2>&1 | tee -a {}'.format("ncu" if dev == "gpu" else "acu", kernel_pattern, output_name, backend, args.caselist, log_file)
        else:
            metrics_string = "gpu__time_duration.sum,sm__cycles_elapsed.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed" if dev=="gpu" else \
                             "ppu__time_duration.sum,ce__cycles_elapsed.max,cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,l2__requests_hit_rate.pct,ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
            cmd = '{} --replay-mode application --clock-control none {} --metrics="{}"  \
                --page=details python ./run_flash_mla.py --backend={} --caselist="{}" \
                2>&1 | tee -a {}'.format("ncu" if dev == "gpu" else "acu", kernel_pattern, metrics_string, backend, args.caselist, log_file)

        ret = run_cmd(cmd)

        if mode != "full":
            if ret.returncode == 0:
                output_list = batch_read_cycle_from_nculog(log_file, fa_cases)
                output_file = f'{os.path.basename(args.caselist).split(".")[0]}_{dev_name_mapping[device_name]}_flashmla_acu_result.csv'
                with open(output_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=output_list[0].keys())
                    writer.writeheader()
                    writer.writerows(output_list)

            else:
                print("ERROR: failed to run cmd, please check!!")
        return
    for case in fa_cases:
        timestamp = str(round(time.time() * 1000))
        log_file = f"./logs/gpu_cycles_single_case_{timestamp}.log"
        cmd = "rm -f "+ log_file
        run_cmd(cmd)
        # gpu
        # metrics = devices.get(dev, [])
        # metrics_string = ', '.join(metrics) if metrics else ""
        if mode == "full":
            output_name = clean_casename(case) + backend
            cmd = '{} --set full {} -o {} python ./run_flash_mla.py --backend={} --format="{}" \
                2>&1 | tee -a {}'.format("ncu" if dev == "gpu" else "acu", kernel_pattern, output_name, backend, case, log_file)
        else:
            metrics_string = "gpu__time_duration.sum,sm__cycles_elapsed.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed" if dev=="gpu" else \
                             "ppu__time_duration.sum,ce__cycles_elapsed.max,cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,l2__requests_hit_rate.pct,ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
            cmd = '{} --clock-control none {} --metrics="{}"  \
                --page=details python ./run_flash_mla.py --backend={} --format="{}" \
                2>&1 | tee -a {}'.format("ncu" if dev == "gpu" else "acu", kernel_pattern, metrics_string, backend, case, log_file)

        ret = run_cmd(cmd)

        if mode != "full":
            if ret.returncode == 0:
                cycles, tc, l2, hbm, inner_cycle, duration, detail = read_cycle_from_nculog(log_file)
                row = [case.replace(",","_"), str(cycles), str(tc), str(l2), str(hbm), str(inner_cycle), str(duration), str(cmd), str(detail)]
                output_lines.append(row)
                csv_path = f"{output_file}_{backend}.csv"
                file_is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
                with open(csv_path, "a") as f:
                    writer = csv.writer(f)
                    if file_is_new:
                        writer.writerow(headers)
                    writer.writerow(row)
                    print("write result succeed")
            else:
                print("ERROR: failed to run cmd, please check!!")
                if len(fa_cases) == 1:
                    exit(-1) # only one case, fail and exit

    output_file = output_file + '_' + backend + '.csv'
    if len(fa_cases) == 1:
        with open("local.log", "w") as f:
            writer = csv.writer(f)
            for row in output_lines:
                writer.writerow(row)
            print("write result to local.log succeed")
    if run_local:
        with open(output_file, "w") as f:
            writer = csv.writer(f)
            for row in output_lines:
                writer.writerow(row)
            print("write result succeed")

def read_cmds_from_file(casefile):
    test_cases = list()
    with open(casefile, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.endswith("list"):
                _case = read_cmds_from_file(line)
                test_cases.extend(_case)
                continue
            # Some caselists (e.g. dsa.list) have `[MLA] --format=...` glued onto
            # the tail of an unrelated stdout line such as a tqdm progress bar.
            # Strip everything before `[MLA]` when present.
            mla_pos = line.find("[MLA]")
            if mla_pos > 0:
                line = line[mla_pos:]
            # Drop progress-bar / log noise that has no test directive.
            if mla_pos < 0 and ("warmup:" in line or "it/s]" in line):
                continue
            test_cases.append(line)
    return test_cases