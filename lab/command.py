import subprocess

def run_commands(commands):
    for i, cmd in enumerate(commands, start=1):
        print(f"Running command {i}/{len(commands)}: {cmd}")
        process = subprocess.run(cmd, shell=True)
        if process.returncode != 0:
            print(f"Command {i} failed with return code {process.returncode}")
            break

def main():
    base_command = "python3 main.py --compiler gcc-12.1.0 --algorithm CompTuner --tuning_time 50 --flags_file /home/qtl/test/flag.txt"
    include_path = "--include_path '/home/qtl/PolyBenchC-4.2.1/utilities /home/qtl/PolyBenchC-4.2.1/utilities/polybench.c'"
    log_file = "--log_file correlation_cmp.log"
    
    polybench_programs = [
        "datamining/correlation", "datamining/covariance",
        "linear-algebra/blas/symm", "linear-algebra/kernels/2mm", "linear-algebra/kernels/3mm",
        "linear-algebra/solvers/cholesky", "linear-algebra/solvers/lu",
        "medley/nussinov", "stencils/heat-3d", "stencils/jacobi-2d"
    ]
    #"automotive_bitcount/src", "automotive_susan_e/src",
    cbench_programs = [
         "automotive_susan_c/src", "automotive_susan_s/src",
        "consumer_tiff2rgba/src", "consumer_jpeg_c/src", "office_rsynth/src",
        "security_sha/src", "bzip2e/src", "telecom_adpcm_c/src"
    ]
    
    polybench_commands = [
        f"{base_command} --program /home/qtl/PolyBenchC-4.2.1/{prog} {log_file} {include_path}"
        for prog in polybench_programs
    ]
    
    cbench_commands = [
        f"{base_command} --program /home/qtl/cBench-main/{prog} {log_file}"
        for prog in cbench_programs
    ]
    
    #all_commands = polybench_commands + cbench_commands
    all_commands = cbench_commands
    run_commands(all_commands)

if __name__ == "__main__":
    main()
