import random
import time
import subprocess
from AlgorithmInterface import AlgorithmInterface

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

class RandomIterativeOptimization(AlgorithmInterface):
    def __init__(self, compiler, source_path, flags,exec_param=None):
        self.compiler = compiler  # 编译器实例
        self.source_path = source_path  # 源代码路径
        self.flags = flags  # 优化标志空间
        #self.search_space = flags #初始化名字空间
        self.exec_param = exec_param  # 可执行文件参数
        #self.db = db  # 数据库实例

        self.best_performance = None  # 初始化最佳性能
        self.best_flags = None  # 初始化最佳标志配置

    def generate_random_conf(self, x):
        """生成0-1映射的优化标志配置"""
        comb = bin(x).replace('0b', '')
        comb = '0' * (len(self.flags) - len(comb)) + comb
        return [int(s) for s in comb]

    def optimize(self, compiler, source_file, log_file, tuning_time):
        """执行随机迭代优化"""
        ts = []  # 记录时间消耗
        res = []  # 记录优化结果
        seqs = []  # 记录不同的标志组合
        time_zero = time.time()
        best_compile_command = None  # 记录最佳表现对应的编译命令
        ts.append(0)
        it = 0
        while ts[-1] < tuning_time:  # 运行限制由 tuning_time 参数控制
            x = random.randint(0, 2 ** len(self.flags) - 1)
            seq = self.generate_random_conf(x)
            opt = ' '.join(flag if bit else flag.replace("-f", "-fno-", 1) for flag, bit in zip(self.flags, seq))

            # 使用 compiler 实例编译代码
            compile_command1,compile_command2 = self.compiler.get_compile_command(opt, source_file,opt_level="-O2")
            # print(f"Compile command: {compile_command}")
            self.compiler.compile(compile_command1)
            self.compiler.compile(compile_command2)

            # 执行生成的可执行文件
            exec_time_start = time.time()
            self.compiler.execute(self.exec_param)
            exec_time_end = time.time()

            # 清理中间生成的文件
            execute_terminal_command("rm -rf *.o *.I *.s a.out")
    

            # 记录执行时间
            time_c = exec_time_end - exec_time_start

            # 使用 -O3 优化进行基准对比
            
            #self.compiler.compile(self.compiler.get_compile_command("", source_file, opt_level="-O3"))
            baseline_compile_command1,baseline_compile_command2 = compiler.get_compile_command("", source_file, opt_level="-O3")
            compiler.compile(baseline_compile_command1)
            compiler.compile(baseline_compile_command2)

            o3_time_start = time.time()
            self.compiler.execute(self.exec_param)
            o3_time_end = time.time()

            execute_terminal_command("rm -rf *.o *.I *.s a.out")

            time_o3_c = o3_time_end - o3_time_start

            # 比较优化结果
            res.append(time_o3_c / time_c)
            ts.append(time.time() - time_zero)
            seqs.append(seq)

            # 更新最佳性能和最佳标志
            best_per = max(res)
            best_seq = seqs[res.index(best_per)]
            if self.best_performance is None or best_per > self.best_performance:
                self.best_performance = best_per
                self.best_flags = best_seq
                best_compile_command = opt  # 更新最佳表现对应的编译命令
                log_message = (
                "Iteration {}: Best Performance: {}, Best Sequence: {}".format(it, self.best_performance, self.best_flags)
                )
                self.log_results(log_file, log_message)
                #log_results(log_file,"Iteration {}: Best Performance: {}, Best Sequence: {}".format(it, self.best_performance, self.best_flags))
            # 记录日志
            it+=1
       
        
        return self.best_performance, self.best_flags, best_compile_command

    def log_results(self, log_file, message):
        """记录优化结果"""
        with open(log_file, 'a') as log:
            log.write(message + '\n')
