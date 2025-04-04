import subprocess

def run_commands(commands):
    for cmd in commands:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Error executing: {cmd}")
            break

compiler = "gcc-12.1.0"
log_rio = "correlation_rio.log"
log_cfsca = "correlation_cfsca.log"
log_cmp = "correlation_cmp.log"
log_apco = "correlation_apco.log"
log_ga="correlation_ga.log"
log_boca="correlation_boca.log"
tuning_time = 4000
flags_file = "/home/qtl/test/flag.txt"
include_path = "\"/home/qtl/PolyBenchC-4.2.1/utilities /home/qtl/PolyBenchC-4.2.1/utilities/polybench.c\""

polybench_programs = [
    "/home/qtl/PolyBenchC-4.2.1/datamining/correlation",
    "/home/qtl/PolyBenchC-4.2.1/datamining/covariance",
    "/home/qtl/PolyBenchC-4.2.1/linear-algebra/blas/symm",
    "/home/qtl/PolyBenchC-4.2.1/linear-algebra/kernels/2mm",
    "/home/qtl/PolyBenchC-4.2.1/linear-algebra/kernels/3mm",
    "/home/qtl/PolyBenchC-4.2.1/linear-algebra/solvers/cholesky",
    "/home/qtl/PolyBenchC-4.2.1/linear-algebra/solvers/lu",
    "/home/qtl/PolyBenchC-4.2.1/medley/nussinov",
    "/home/qtl/PolyBenchC-4.2.1/stencils/heat-3d",
    "/home/qtl/PolyBenchC-4.2.1/stencils/jacobi-2d"
]

cbench_programs = [
    "/home/qtl/cBench-main/automotive_bitcount/src",
    "/home/qtl/cBench-main/automotive_susan_e/src",
    "/home/qtl/cBench-main/automotive_susan_c/src",
    "/home/qtl/cBench-main/automotive_susan_s/src",
    "/home/qtl/cBench-main/consumer_tiff2rgba/src",
    "/home/qtl/cBench-main/consumer_jpeg_c/src",
    "/home/qtl/cBench-main/office_rsynth/src",
    "/home/qtl/cBench-main/security_sha/src",
    "/home/qtl/cBench-main/bzip2e/src",
    "/home/qtl/cBench-main/telecom_adpcm_c/src"
]

polybench_commands = [
    f"python3 main.py --compiler {compiler} --algorithm RIO --program {prog} --log_file {log_rio} --include_path {include_path} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in polybench_programs
] + [
    f"python3 main.py --compiler {compiler} --algorithm GATuner --program {prog} --log_file {log_ga} --include_path {include_path} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in polybench_programs
] + [
    f"python3 main.py --compiler {compiler} --algorithm BOCATuner --program {prog} --log_file {log_boca} --include_path {include_path} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in polybench_programs
]+ [
    f"python3 main.py --compiler {compiler} --algorithm APCOTuner --program {prog} --log_file {log_apco} --include_path {include_path} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in polybench_programs
]+ [
    f"python3 main.py --compiler {compiler} --algorithm CompTuner --program {prog} --log_file {log_cmp} --include_path {include_path} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in polybench_programs
] + [
    f"python3 main.py --compiler {compiler} --algorithm CFSCATuner --program {prog} --log_file {log_cfsca} --include_path {include_path} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in polybench_programs
] 


cbench_commands = [
    f"python3 main.py --compiler {compiler} --algorithm RIO --program {prog} --log_file {log_rio} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in cbench_programs
]+ [
    f"python3 main.py --compiler {compiler} --algorithm GATuner --program {prog} --log_file {log_ga} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in cbench_programs
]+ [
    f"python3 main.py --compiler {compiler} --algorithm BOCATuner --program {prog} --log_file {log_boca} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in cbench_programs
]+ [
    f"python3 main.py --compiler {compiler} --algorithm APCOTuner --program {prog} --log_file {log_apco} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in cbench_programs
] + [
    f"python3 main.py --compiler {compiler} --algorithm CompTuner --program {prog} --log_file {log_cmp} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in cbench_programs
] +[
    f"python3 main.py --compiler {compiler} --algorithm CFSCATuner --program {prog} --log_file {log_cfsca} --tuning_time {tuning_time} --flags_file {flags_file}"
    for prog in cbench_programs
] 

all_commands =polybench_commands+ cbench_commands

run_commands(all_commands)
# import subprocess

# def run_commands(commands):
#     for cmd in commands:
#         print(f"Running: {cmd}")
#         result = subprocess.run(cmd, shell=True)
#         if result.returncode != 0:
#             print(f"Error executing: {cmd}")
#             break

# compiler = "clang-12.0.1"
# log_rio = "correlation_rio.log"
# log_cfsca = "correlation_cfsca.log"
# log_cmp = "correlation_cmp.log"
# log_apco = "correlation_apco.log"
# log_ga="correlation_ga.log"
# log_boca="correlation_boca.log"
# tuning_time = 50
# #flags_file = "/home/qtl/test/flag.txt"
# include_path = "\"/home/qtl/PolyBenchC-4.2.1/utilities /home/qtl/PolyBenchC-4.2.1/utilities/polybench.c\""

# polybench_programs = [
#     #"/home/qtl/PolyBenchC-4.2.1/datamining/correlation",
#     #"/home/qtl/PolyBenchC-4.2.1/datamining/covariance",
#     #"/home/qtl/PolyBenchC-4.2.1/linear-algebra/blas/symm",
#     #"/home/qtl/PolyBenchC-4.2.1/linear-algebra/kernels/2mm",
#     #"/home/qtl/PolyBenchC-4.2.1/linear-algebra/kernels/3mm",
#     #"/home/qtl/PolyBenchC-4.2.1/linear-algebra/solvers/cholesky",
#     "/home/qtl/PolyBenchC-4.2.1/linear-algebra/solvers/lu",
#     "/home/qtl/PolyBenchC-4.2.1/medley/nussinov",
#     "/home/qtl/PolyBenchC-4.2.1/stencils/heat-3d",
#     "/home/qtl/PolyBenchC-4.2.1/stencils/jacobi-2d"
# ]

# cbench_programs = [
#     "/home/qtl/cBench-main/automotive_bitcount/src",
#     "/home/qtl/cBench-main/automotive_susan_e/src",
#     "/home/qtl/cBench-main/automotive_susan_c/src",
#     "/home/qtl/cBench-main/automotive_susan_s/src",
#     "/home/qtl/cBench-main/consumer_tiff2rgba/src",
#     "/home/qtl/cBench-main/consumer_jpeg_c/src",
#     "/home/qtl/cBench-main/office_rsynth/src",
#     "/home/qtl/cBench-main/security_sha/src",
#     "/home/qtl/cBench-main/bzip2e/src",
#     "/home/qtl/cBench-main/telecom_adpcm_c/src"
# ]

# polybench_commands = [
#     f"python3 main.py --compiler {compiler} --algorithm RIO --program {prog} --log_file {log_rio} --include_path {include_path} --tuning_time {tuning_time} "
#     for prog in polybench_programs
# ] + [
#     f"python3 main.py --compiler {compiler} --algorithm GATuner --program {prog} --log_file {log_ga} --include_path {include_path} --tuning_time {tuning_time}"
#     for prog in polybench_programs
# ] + [
#     f"python3 main.py --compiler {compiler} --algorithm BOCATuner --program {prog} --log_file {log_boca} --include_path {include_path} --tuning_time {tuning_time}"
#     for prog in polybench_programs
# ]+ [
#     f"python3 main.py --compiler {compiler} --algorithm CompTuner --program {prog} --log_file {log_cmp} --include_path {include_path} --tuning_time {tuning_time}"
#     for prog in polybench_programs
# ] 

# cbench_commands = [
#     f"python3 main.py --compiler {compiler} --algorithm RIO --program {prog} --log_file {log_rio} --tuning_time {tuning_time}"
#     for prog in cbench_programs
# ]+ [
#     f"python3 main.py --compiler {compiler} --algorithm GATuner --program {prog} --log_file {log_ga} --tuning_time {tuning_time}"
#     for prog in cbench_programs
# ]+ [
#     f"python3 main.py --compiler {compiler} --algorithm BOCATuner --program {prog} --log_file {log_boca} --tuning_time {tuning_time}"
#     for prog in cbench_programs
# ] + [
#     f"python3 main.py --compiler {compiler} --algorithm CompTuner --program {prog} --log_file {log_cmp} --tuning_time {tuning_time}"
#     for prog in cbench_programs
# ] 
# polybench_commands2 = [
#     f"python3 main.py --compiler {compiler} --algorithm CompTuner --program {prog} --log_file {log_cmp} --include_path {include_path} --tuning_time {tuning_time}"
#     for prog in polybench_programs
# ] 

# # all_commands = polybench_commands + cbench_commands
# #all_commands = cbench_commands
# all_commands = polybench_commands2
# run_commands(all_commands)

