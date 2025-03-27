import os
import csv
import subprocess
import re

def run_cmd(cmd: str, timeout=300, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
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


# devices = {
#     "name": ["cycle", "tensor core efficiency", "waves"],
#     "gpu":  ["sm__cycles_active.max", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "launch__waves_per_multiprocessor"],
#     "ppu":  ["ce__cycles_active.max", "cu__inst_executed_pipe_tensor_fp16.avg.pct_of_peak_sustained_active", "launch__waves_per_cu"],
# }

def read_cycle_from_nculog(filename):
    kernel_pattern = r"(.*)kernel(.*)Device(.*)"
    cycles_pattern = "__cycles_active.max"
    tc_pattern = "pct_of_peak_sustained_active"
    kernel_list = []
    cycles_list = []
    tc_list = []

    with open(filename, newline='') as log_file:
        for line in log_file.read().split("\n"):
            if re.search(kernel_pattern, line, re.IGNORECASE):
                kernel_list.append(line.strip())
            if re.search(cycles_pattern, line):
                cycles_list.append(int(line.strip().split()[-1]))
            if re.search(tc_pattern, line):
                tc_list.append(float(line.strip().split()[-1]))

    assert(len(kernel_list) == len(cycles_list))
    assert(len(kernel_list) == len(tc_list))

    op_cycles = dict()
    fwd_cycle_sum = 0
    fwd_tc_sum = 0

    for i in range(len(kernel_list)):
        op = kernel_list[i]
        cycle = cycles_list[i]
        op_cycles[op] = cycle
        if "fwd" in op.lower() or "mla" in op.lower(): # flashmla ppu / triton / flashinfer
            fwd_cycle_sum += cycle
            fwd_tc_sum += tc_list[i]
    # calculate statistics data
    if fwd_cycle_sum != 0:
        # fwd unit case
        return fwd_cycle_sum, fwd_tc_sum, op_cycles
    else:
        print("Not valid CSV file!")
        return 0, 0, []
        #exit(-1)


def run_fa_cycle_on_device(fa_cases, output_file, dev="gpu", run_local=False, backend="flash_mla"):
    output_lines = list()
    headers = ["casename","cycle","tc efficiency", "cmd","detail"]
    # new_row=["casename"]  metrics.get("name", [])  ["detail"] 
    # output_lines.append(new_row)

    for case in fa_cases:
        log_file = "./gpu_cycles_single_case.log"
        cmd = "rm -f "+ log_file
        run_cmd(cmd)
        # gpu
        # metrics = devices.get(dev, [])
        # metrics_string = ', '.join(metrics) if metrics else ""

        metrics_string = "sm__cycles_active.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active" if dev=="gpu" else \
                         "ce__cycles_active.max,cu__inst_executed_pipe_tensor_{}.avg.pct_of_peak_sustained_active".format("fp16" if "fp16" in case else "bf16")
        cmd = '{} --clock-control none --metrics="{}"  \
              --page=details python ./run_flash_mla.py --backend={} --format={} \
              2>&1 | tee -a {}'.format("ncu" if dev == "gpu" else "acu", metrics_string, backend, case, log_file)
        
        # print(cmd)
        # cmd = "ncu --clock-control none --metrics=sm__cycles_active.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,launch__waves_per_multiprocessor "\
        #      "--page=details python ./run_flash_attn.py --format="  case \
        #      " 2>&1 | tee -a "  log_file

        # ppu:
        # cmd = "acu --metrics=ce__cycles_active.max --page=details "\
        #      "python ./run_flash_attn.py --format="  case \
        #      " 2>&1 | tee -a "  log_file

        run_cmd(cmd)

        cycle, tc, detail = read_cycle_from_nculog(log_file)
        output_lines.append([case.replace(",","_"), str(cycle), str(tc), str(cmd), str(detail)])

    # print('output file:')
    # print(output_file)

    # dirname = os.path.dirname(output_file)

    # print('dirname:')
    # print(dirname)

    # cmd = f"mkdir -p {dirname}"
    # print(cmd)

    # run_cmd(cmd)
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
    else:
        # if not os.path.exists(output_file):
        #     with open(output_file, "w", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(headers)
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            for row in output_lines:
                writer.writerow(row)
            print("write result succeed")