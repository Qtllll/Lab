import argparse
import time
from CompilerFactory import CompilerFactory  # 你的编译器模块工厂
from AlgorithmFactory import AlgorithmFactory  # 你的算法模块工厂
from OptimizationDatabase import OptimizationDatabase #数据库模块
import ast
import subprocess
def load_flags_from_file(file_path):
    """从用户提供的文件中读取flag"""
    with open(file_path, 'r') as file:
        flags = file.read().strip()
    return [flag.strip() for flag in flags.split(',') if flag.strip()]


def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="Compiler Auto-tuning Platform")
    parser.add_argument("--compiler", required=True, help="Compiler name and version (e.g., gcc11)")
    parser.add_argument("--algorithm", required=True, help="Algorithm to use (e.g., RIO)")
    parser.add_argument("--program", required=True, help="Path to the program to optimize")
    parser.add_argument("--log_file", default="optimization.log", help="Path to the log file")
    parser.add_argument("--include_path", type=str, default='', help="Include paths for compilation (can be empty)")
    parser.add_argument("--tuning_time", type=int, default=5000, help="Time in seconds to run the tuning process")
    parser.add_argument("--flags_file", type=str, default=None, help="Path to a file containing optimization flags")
    args = parser.parse_args()

    # 2. 实例化编译器模块
    print(f"Creating compiler: {args.compiler}")
    # 解析编译器名称和版本号
    compiler_parts = args.compiler.split('-')  # 将 'gcc-11' 分割成 ['gcc', '11']
    
    compiler_name = compiler_parts[0]
    compiler_version = compiler_parts[1] if len(compiler_parts) > 1 else "latest"  # 默认版本为 "latest"
    
    print(f"compiler_name:{compiler_version}")
    #include文件路径
    include_path=args.include_path
    print(f"Include_path: {include_path}")
    compiler = CompilerFactory.create_compiler(compiler_name, compiler_version,include_path)
    
    gcc_path = compiler.get_path()  # 获取编译器路径
    print(f"Compiler path: {gcc_path}")
    
    
    # 3. 实例化数据库模块
    db = OptimizationDatabase()

     # 3. 获取优化flag
    if args.flags_file:
        print(f"Loading flags from file: {args.flags_file}")
        flags = load_flags_from_file(args.flags_file)

    else:
        print(f"Fetching optimization flags from database for {args.compiler} and {args.program}")
        
        # 查询数据库，看是否已有历史数据
        historical_data = db.query_database(compiler_name, compiler_version, args.algorithm, args.program)
        
        if historical_data:
            # 如果数据库中有历史数据，直接返回历史数据中的 flags
            flags = historical_data['flags']
            execute_time = historical_data['execution_time']
            print(f"Found historical flags in database: {flags}")
            print(f"Found historical execute time in database: {execute_time}")
            db.close()
            return
        else:
            # 如果数据库中没有历史数据，则获取编译器的优化参数空间
            print(f"Fetching flags from compiler {args.compiler} version {compiler_version}")
            flags = compiler.get_search_space()['flags']
            # print(f"Optimization flags from compiler: {flags}")
    # else:
    #     print(f"Fetching optimization flags from compiler version")
    #     flags = compiler.get_search_space()['flags']# 获取优化参数空间
    
    print(f"Optimization flags: {flags}")

    # 5. 查找最接近的编译器版本历史数据
    closest_data = db.query_closest_data(compiler_name, args.program, compiler_version, args.algorithm)
    
    if closest_data:
        print(f"Found closest historical data: {closest_data}")
        # 使用 ast.literal_eval 将字符串解析为列表
        historical_flags = ast.literal_eval(closest_data['flags'])
        # print(set(historical_flags))
        # print(set(flags))
        # print(set(flags).issubset(set(historical_flags)))
        print(len(historical_flags))
        
        
        # 判断自定义flags与历史flags是否为子集
        if set(flags).issubset(set(historical_flags)):
            print(f"User flags are a subset of the closest historical flags.")
            # 判断用户flags的长度是否小于历史flags长度
            if len(flags) < len(historical_flags):
            # 如果用户flags长度小于历史flag，直接进行编译
                print(f"User flags are smaller or equal to historical flags. Compiling with user flags.")
                opt_execute_time,opt_flags=execute_optimized_flags(compiler, args.program, flags)
                print(f"opt-flags: {opt_flags}")
                print(f"speed up: {opt_execute_time}")

                # 弹出询问是否满意
                user_feedback = input(f"Do you accept the result with flags: {opt_flags}? (y/n): ")
                if user_feedback.lower() == 'y':
                    print("Saving results to database...")
                    db.log_result(compiler_name, compiler_version, args.algorithm, args.program, opt_flags, opt_execute_time)
                    print("Results saved.")
                else:
                    print("Re-running optimization...")
                    execute_new_optimization(compiler, args.program, flags, args.algorithm, args.log_file, args.tuning_time, db)
            else:
            # 相等长度情况，直接返回数据库内容
                # 如果数据库中有历史数据，直接返回历史数据中的 flags
                print(f"Found closest flags = input flags in database: {historical_flags}")
                print(f"execute time in database: {closest_data['execution_time']}")
                
        elif set(historical_flags).issubset(set(flags)):
            print(f"User flags are larger than historical flags. Optimizing the excess part.")
            excess_flags = [flag for flag in flags if flag not in historical_flags]
            # 使用调优算法优化超出的flags部分
            optimized_excess_best_performance,optimized_excess_flags,best_compile_command = optimize_excess_flags(excess_flags, compiler, args.program, args.algorithm, args.log_file, args.tuning_time)
            print(f"Optimized excess flags: {optimized_excess_flags}")
            execute_optimized_flags2(compiler, args.program, optimized_excess_flags)
         
            # 合并历史 flags 和优化后的超出部分 flags
            final_flags = list(set(historical_flags + optimized_excess_flags))  # 去重合并
            print(f"Final flags after optimization: {final_flags}")

            # 生成新的编译命令，使用最终优化后的 flags
            #final_compile_command1,final_compile_command2 = compiler.get_compile_command(' '.join(final_flags), args.program,opt_level="-O2")
            #print(f"Final compile command: {final_compile_command1}")

            # 使用优化后的最终 flags 进行编译和执行
            total_duration, final_flags_used = execute_optimized_flags2(compiler, args.program, final_flags)
            print(f"Final flags execution speed up: {total_duration}")
            
            # 弹出询问是否满意
            user_feedback = input(f"Do you accept the result with flags: {final_flags}? (y/n): ")
            if user_feedback.lower() == 'y':
                print("Saving results to database...")
                db.log_result(compiler_name, compiler_version, args.algorithm, args.program, final_flags, total_duration)
                print("Results saved.")
            else:
                print("Re-running optimization...")
                execute_new_optimization(compiler, args.program, flags, args.algorithm, args.log_file, args.tuning_time, db)
        else:
            #不为子集关系需要重新跑算法
            print(f"User flags aren't a subset of the closest historical flags. Running optimization.")
            execute_new_optimization(compiler, args.program, flags, args.algorithm, args.log_file, args.tuning_time,db)
    else:
        print(f"No closest historical data found. Running optimization.")
        #这里逻辑存疑
        execute_new_optimization(compiler, args.program, flags, args.algorithm, args.log_file, args.tuning_time,db)
        

    # 6. 关闭数据库
    db.close()

def execute_terminal_command(command):
    """Execute command"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout:
                print("命令输出：")
                print(result.stdout)
        else:
            if result.stderr:
                print("错误输出：")
                print(result.stderr)
    except Exception as e:
        print("执行命令时出现错误：", str(e))

# 执行流程
def execute_new_optimization(compiler, program, flags, algorithm_name, log_file, tuning_time, db):
    # 执行全新的优化
    print(f"Running new optimization with flags: {flags}")
    algorithm = AlgorithmFactory.create_algorithm(algorithm_name, compiler, source_path=program, flags=flags, log_file=log_file)
    best_performance, best_flags, best_command = algorithm.optimize(compiler, program, log_file, tuning_time)
    if db and best_performance is not None and best_flags is not None:
            # 将best_flags从二进制数组转为标志名
            best_flags_names = [flag for flag, flag_value in zip(flags, best_flags) if flag_value == 1]
            # 将优化结果记录到数据库
            db.log_result(
                compiler.get_name(),  # 编译器名称
                compiler.get_version(),  # 编译器版本
                algorithm_name,  # 算法名称
                program,  # 源文件路径
                best_flags_names,  # 转换后的最佳标志
                best_performance  # 执行时间比值
            )
    print(f"Optimization complete. Best performance: {best_performance}")
    print(f"Best flags found: {best_flags}")

def execute_optimized_flags(compiler, program, flags,exec_param=None):
    """使用编译器实例执行编译和运行，记录执行时间"""
    # 清理临时文件
    #clean_temp_files(program)
    # print("execute_optimized_flags":{flags})
    #生成编译优化标志
    opt = ' '.join(flag if bit else flag.replace("-f", "-fno-", 1) for flag, bit in zip(flags, [1] * len(flags)))
    # opt = ' '.join(flag for flag in flags)
    print(opt)

    # 生成编译命令
    compile_command1,compile_command2 = compiler.get_compile_command(opt, program,opt_level="-O2")
    
    #print(f"Compile command: {compile_command}")

    # 编译
    #compile_start = time.time()
    compiler.compile(compile_command1)
    compiler.compile(compile_command2)
    #compile_end = time.time()
    #print(f"compile result:{compile_result}")
    # 如果编译失败，直接返回
    # if not compile_result:
    #     #log_message = f"Compilation failed for flags: {flags}"
    #     #log_results(log_file, log_message)
    #     print(f"Compilation failed for flags: {flags}")
    #     return None, None

    # 执行程序并记录执行时间
    exec_start = time.time()
    compiler.execute(exec_param)
    exec_end = time.time()
    #print(f"exec result:{exec_result}")

    # 清理中间生成的文件
    execute_terminal_command("rm -rf *.o *.I *.s a.out")

    # 计算编译和执行时间
    #compile_duration = compile_end - compile_start
    exec_duration = exec_end - exec_start
    
    baseline_compile_command1,baseline_compile_command2 = compiler.get_compile_command("", program, opt_level="-O3")
    compiler.compile(baseline_compile_command1)
    compiler.compile(baseline_compile_command2)
    
    o3_time_start = time.time()
    compiler.execute(exec_param)
    o3_time_end = time.time()
    
    execute_terminal_command("rm -rf *.o *.I *.s a.out")
    
    time_o3_c = o3_time_end - o3_time_start  # 基线运行时间

    speedup = time_o3_c / exec_duration if exec_duration > 0 else float('inf')
    #op_str = "iteration:{} speedup:{}".format(str(k_iter), str(speedup))
    #write_log(op_str, LOG_FILE)
    return speedup, flags

    
    
    # # 返回编译执行时间
    # return total_duration,flags

def execute_optimized_flags2(compiler, program, flags, exec_param=None):
    """使用编译器实例执行编译和运行，记录执行时间"""
    
    #print(f'function:{compile_command1}')
    # 编译
    # 构造 opt 字符串
    opt = ''
    for i in range(len(flags)):
        if flags[i]:
            opt += flags[i] + ' '
        else:
            negated_flag_name = flags[i].replace("-f", "-fno-", 1)
            opt += negated_flag_name + ' '
    
    # ---------------------------
    # 候选方案：使用 -O2 加上 opt 标志
    # ---------------------------
    candidate_compile_command1,candidate_compile_command2 = compiler.get_compile_command(opt, program, opt_level="-O2")
    compiler.compile(candidate_compile_command1)
    compiler.compile(candidate_compile_command2)
    
    
    # print(f"compile result:{compile_result}")
    # # 如果编译失败，直接返回
    # if not compile_result:
    #     #log_message = f"Compilation failed for flags: {flags}"
    #     #log_results(log_file, log_message)
    #     print(f"Compilation failed for flags: {flags}")
    #     return None, None

    # 执行程序并记录执行时间
    exec_start = time.time()
    compiler.execute(exec_param)
    exec_end = time.time()
    #print(f"exec result:{exec_result}")
    # 清理中间生成的文件
    execute_terminal_command("rm -rf *.o *.I *.s a.out")

    # 计算编译和执行时间
    #compile_duration = compile_end - compile_start
    exec_duration = exec_end - exec_start
    
    baseline_compile_command1,baseline_compile_command2 = compiler.get_compile_command("", program, opt_level="-O3")
    compiler.compile(baseline_compile_command1)
    compiler.compile(baseline_compile_command2)
    
    o3_time_start = time.time()
    compiler.execute(exec_param)
    o3_time_end = time.time()
    
    execute_terminal_command("rm -rf *.o *.I *.s a.out")
    
    time_o3_c = o3_time_end - o3_time_start  # 基线运行时间

    speedup = time_o3_c / exec_duration if exec_duration > 0 else float('inf')
    return speedup, flags

    


def optimize_excess_flags(excess_flags, compiler, program, algorithm_name,log_file, tuning_time):
    """优化用户flag中超出的部分"""
    print(f"Optimizing excess flags: {excess_flags}")
    algorithm = AlgorithmFactory.create_algorithm(algorithm_name, compiler, source_path=program, flags=excess_flags, log_file=log_file)
    optimized_excess_best_performance,optimized_excess_flags,best_compile_command = algorithm.optimize(compiler, program, log_file, tuning_time)
    #flag if bit else flag.replace("-f", "-fno-", 1) for flag, bit in zip(self.flags, seq)
    best_flags_names = [flag if bit else flag.replace("-f", "-fno-", 1) for flag, bit in zip(excess_flags, optimized_excess_flags)]
    print(f"optimized_excess_best_performance: {optimized_excess_best_performance}, Optimized excess flags: {best_flags_names}")
    return optimized_excess_best_performance, best_flags_names,best_compile_command



if __name__ == "__main__":
    main()
