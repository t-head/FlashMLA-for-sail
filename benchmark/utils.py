import os
import csv
import subprocess
import re
import time

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

def worker(gpu_id, fa_cases, output, device, is_local, backend, mode):
    # 设置当前进程可见的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process {os.getpid()} is running on GPU {gpu_id}")
    if backend == "all":
        for _backend in ['flash_mla', 'flash_infer', 'flash_mla_triton']:
            run_fa_cycle_on_device(fa_cases, output, device, is_local, _backend, mode)
    else:
        run_fa_cycle_on_device(fa_cases, output, device, is_local, backend, mode)

# devices = {
#     "name": ["cycle", "tensor core efficiency", "waves"],
#     "gpu":  ["sm__cycles_active.max", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "launch__waves_per_multiprocessor"],
#     "ppu":  ["ce__cycles_active.max", "cu__inst_executed_pipe_tensor_fp16.avg.pct_of_peak_sustained_active", "launch__waves_per_cu"],
# }

def read_cycle_from_nculog(filename):
    kernel_pattern = r"(.*)kernel(.*)Device(.*)"
    cycles_pattern = "__cycles_active.max"
    tc_pattern = "pct_of_peak_sustained_active"
    # "dram__throughput.avg.pct_of_peak_sustained_elapsed" or "ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
    hbm_pattern = "throughput.avg.pct_of_peak_sustained_elapsed"
    kernel_list = []
    cycles_list = []
    tc_list = []
    hbm_list = []

    with open(filename, newline='') as log_file:
        for line in log_file.read().split("\n"):
            if re.search(kernel_pattern, line, re.IGNORECASE):
                kernel_list.append(line.strip())
            if re.search(cycles_pattern, line):
                cycles_list.append(int(line.strip().split()[-1]))
            if re.search(tc_pattern, line):
                tc_list.append(float(line.strip().split()[-1]))
            if re.search(hbm_pattern, line):
                hbm_list.append(float(line.strip().split()[-1]))

    assert(len(kernel_list) == len(cycles_list))
    assert(len(kernel_list) == len(tc_list))
    assert(len(kernel_list) == len(hbm_list))

    op_cycles = dict()
    fwd_cycle_sum = 0
    fwd_tc_sum = 0
    fwd_hbm_sum = 0

    for i in range(len(kernel_list)):
        op = kernel_list[i]
        cycle = cycles_list[i]
        op_cycles[op] = cycle
        if "fwd" in op.lower() or "mla" in op.lower(): # flashmla ppu / triton / flashinfer
            fwd_cycle_sum += cycle
            fwd_tc_sum += tc_list[i]
            fwd_hbm_sum += hbm_list[i]
    # calculate statistics data
    if fwd_cycle_sum != 0:
        # fwd unit case
        return fwd_cycle_sum, fwd_tc_sum, fwd_hbm_sum, op_cycles
    else:
        print("Not valid CSV file!")
        return 0, 0, 0, []
        #exit(-1)


def clean_casename(name):
    _need_replace = ['--', '=', 'format', 'Formatted', '[', ']', ":", "*", " ", ","]
    # for item in _need_replace:
    #     name = name.replace(item, "_")
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    while "__" in name:
        name = name.replace("__", "_")
    return name

def run_fa_cycle_on_device(fa_cases, output_file, dev="gpu", run_local=False, backend="flash_mla", mode="metrics"):
    output_lines = list()
    headers = ["casename","cycle","tc efficiency", "hbm efficiency", "cmd", "detail"]
    # new_row=["casename"]  metrics.get("name", [])  ["detail"] 
    # output_lines.append(new_row)
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
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
            cmd = '{} --set full -o {} python ./run_flash_mla.py --backend={} --format="{}" \
                2>&1 | tee -a {}'.format("ncu" if dev == "gpu" else "acu", output_name, backend, case, log_file)
        else: 
            metrics_string = "sm__cycles_active.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed" if dev=="gpu" else \
                            "ce__cycles_active.max,cu__inst_executed_pipe_tensor_{}.avg.pct_of_peak_sustained_active,ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed".format("fp16" if "fp16" in case else "bf16")
            cmd = '{} --clock-control none --metrics="{}"  \
                --page=details python ./run_flash_mla.py --backend={} --format="{}" \
                2>&1 | tee -a {}'.format("ncu" if dev == "gpu" else "acu", metrics_string, backend, case, log_file)
        
        ret = run_cmd(cmd)

        if mode != "full":
            if ret.returncode == 0:
                cycle, tc, hbm, detail = read_cycle_from_nculog(log_file)
                row = [case.replace(",","_"), str(cycle), str(tc), str(hbm), str(cmd), str(detail)]
                output_lines.append(row)
                with open(f"{output_file}_{backend}.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                    print("write result succeed")
            else:
                print("ERROR: failed to run cmd, please check!!")
                if len(fa_case) == 1:
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
