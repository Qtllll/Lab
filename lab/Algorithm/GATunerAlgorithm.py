import argparse
import subprocess
import random
import time
import os
from AlgorithmInterface import AlgorithmInterface

def write_log(ss, file):
    """ Write to log """
    with open(file, 'a') as log:
        log.write(ss + '\n')



# def get_objective_score(compiler,independent, k_iter, source_file, LOG_FILE, all_flags,exec_param):
#     """ Obtain the speedup """
#     #构造opt
#     opt = ''
#     for i in range(len(independent)):
#         if independent[i]:
#             opt = opt + all_flags[i] + ' '
#         else:
#             negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
#             opt = opt + negated_flag_name + ' '
#     exec_time_start = time.time()
#     # 使用 compiler 实例编译代码
#     compile_command = compiler.get_compile_command(opt, source_file)
#             # print(f"Compile command: {compile_command}")
#     compiler.compile(compile_command)
#             # 执行生成的可执行文件
#     compiler.execute(exec_param)
#     exec_time_end = time.time()

#     # 记录执行时间
#     time_c = exec_time_end - exec_time_start
    

#     # 使用 -O3 优化进行基准对比
#     o3_time_start = time.time()
#     compiler.compile(compiler.get_compile_command("-O3", source_file))
#     compiler.execute(exec_param)
#     o3_time_end = time.time()

#     time_o3_c = o3_time_end - o3_time_start
#     #记录的统一
#     op_str = "iteration:{} speedup:{}".format(str(k_iter), str(time_o3_c /time_c))
#     #write_log(op_str, LOG_FILE)
#     return (time_o3_c /time_c), opt

def execute_terminal_command(command):
    """ Execute command """
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

# def get_objective_score(compiler, independent, k_iter, source_file, LOG_FILE, all_flags, exec_param):
#     """ Obtain the speedup """
#     # 构造 opt 字符串
#     opt = ''
#     for i in range(len(independent)):
#         if independent[i]:
#             opt += all_flags[i] + ' '
#         else:
#             negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
#             opt += negated_flag_name + ' '
    
#     # ---------------------------
#     # 候选方案：使用 -O2 加上 opt 标志
#     # ---------------------------
#     candidate_compile_command = compiler.get_compile_command(opt, source_file, opt_level="-O2")
#     compiler.compile(candidate_compile_command)
    
#     exec_time_start = time.time()
#     compiler.execute(exec_param)
#     exec_time_end = time.time()
    
#     # 清理中间生成的文件
#     execute_terminal_command("rm -rf *.o *.I *.s a.out")
    
#     time_c = exec_time_end - exec_time_start  # 候选方案运行时间

#     # ---------------------------
#     # 基线方案：使用 -O3（不加 opt 标志）
#     # ---------------------------
#     baseline_compile_command = compiler.get_compile_command("", source_file, opt_level="-O3")
#     compiler.compile(baseline_compile_command)
    
#     o3_time_start = time.time()
#     compiler.execute(exec_param)
#     o3_time_end = time.time()
    
#     execute_terminal_command("rm -rf *.o *.I *.s a.out")
    
#     time_o3_c = o3_time_end - o3_time_start  # 基线运行时间

#     speedup = time_o3_c / time_c if time_c > 0 else float('inf')
#     op_str = "iteration:{} speedup:{}".format(str(k_iter), str(speedup))
#     write_log(op_str, LOG_FILE)
#     return speedup, opt
def get_objective_score(compiler, independent, k_iter, source_file, LOG_FILE, all_flags, exec_param):
    """ Obtain the speedup """
    # 构造 opt 字符串
    opt = ''
    for i in range(len(independent)):
        if independent[i]:
            opt += all_flags[i] + ' '
        else:
            negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
            opt += negated_flag_name + ' '
    
    # ---------------------------
    # 候选方案：使用 -O2 加上 opt 标志
    # ---------------------------
    candidate_compile_command1,candidate_compile_command2 = compiler.get_compile_command(opt, source_file, opt_level="-O2")
    compiler.compile(candidate_compile_command1)
    compiler.compile(candidate_compile_command2)
    
    exec_time_start = time.time()
    compiler.execute(exec_param)
    exec_time_end = time.time()
    
    # 清理中间生成的文件
    execute_terminal_command("rm -rf *.o *.I *.s a.out")
    
    time_c = exec_time_end - exec_time_start  # 候选方案运行时间

    # ---------------------------
    # 基线方案：使用 -O3（不加 opt 标志）
    # ---------------------------
    baseline_compile_command1,baseline_compile_command2 = compiler.get_compile_command("", source_file, opt_level="-O3")
    compiler.compile(baseline_compile_command1)
    compiler.compile(baseline_compile_command2)
    
    o3_time_start = time.time()
    compiler.execute(exec_param)
    o3_time_end = time.time()
    
    execute_terminal_command("rm -rf *.o *.I *.s a.out")
    
    time_o3_c = o3_time_end - o3_time_start  # 基线运行时间

    speedup = time_o3_c / time_c if time_c > 0 else float('inf')
    op_str = "iteration:{} speedup:{}".format(str(k_iter), str(speedup))
    #write_log(op_str, LOG_FILE)
    return speedup, opt

class GATunerAlgorithm(AlgorithmInterface):
  def __init__(self, compiler, source_path, flags, log_file, exec_param=None):
    
    self.tuner = GA(
            options=flags,
            get_objective_score=get_objective_score,
            source_path=source_path,#源代码
            gcc_path=compiler.get_path(),
            exec_param=exec_param,
            log_file=log_file,#不一定要有
            compiler=compiler,
      )
  def optimize(self, compiler, source_file, log_file, tuning_time):
      best_performance, best_flags, best_compile_command = self.tuner.run(tuning_time)
      return best_performance, best_flags, best_compile_command

  def log_results(self):
        # You can implement this to log the final results of the optimization
      pass

class GA:
    def __init__(self, options, get_objective_score, source_path, gcc_path, exec_param, log_file,compiler):
        self.SOURCE_PATH = source_path
        self.GCC_PATH = gcc_path
        self.EXEC_PARAM = exec_param
        self.LOG_FILE = log_file
        self.compiler = compiler

        self.options  = options
        self.get_objective_score = get_objective_score
        self.begin = time.time()
        geneinfo = []
        for i in range(4):
            x = random.randint(0, 2 ** len(self.options))
            geneinfo.append(self.generate_random_conf(x))
        fitness = []
        #self.dep = []

        #initial combinations
        for x in geneinfo:
            tmp = self.get_objective_score(self.compiler,x,100086,self.SOURCE_PATH, LOG_FILE=self.LOG_FILE, all_flags = self.options,exec_param=self.EXEC_PARAM)[0]
            fitness.append(-tmp)
        
        #sort by speedup
        self.pop = [(x, fitness[i]) for i, x in enumerate(geneinfo)]
        self.pop = sorted(self.pop, key=lambda x:x[1])

        self.best = self.selectBest(self.pop)
        write_log("Iteration {}: Best Performance: {}, Best Sequence: {}".format(0, -self.best[1], self.best[0]), self.LOG_FILE)
        #self.dep.append(1.0/self.best[1])
        self.end = time.time() - self.begin

        
    
    def generate_random_conf(self, x):
        """
        Generation 0-1 mapping for disable-enable options
        """

        comb = bin(x).replace('0b', '')
        comb = '0' * (len(self.options) - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf
    

    def selectBest(self, pop):
        """
        select best flag combinations
        """
        return pop[0]
        
    def selection(self, inds, k):
        """
        select preserved flag combinations
        """
        s_inds = sorted(inds, key=lambda x:x[1])
        return s_inds[:int(k)]

    def crossoperate(self, offspring):
        """
        cross flag combinations
        """
        dim = len(self.options)
        geninfo1 = offspring[0][0]
        geninfo2 = offspring[1][0]
        pos = random.randrange(1, dim)

        newoff = []
        for i in range(dim):
            if i>=pos:
                newoff.append(geninfo2[i])
            else:
                newoff.append(geninfo1[i])
        return newoff

    def mutation(self, crossoff):
        """
        mutate flag combinations
        """
        dim = len(self.options)
        pos = random.randrange(1, dim)
        crossoff[pos] = 1 - crossoff[pos]
        return crossoff

    def run(self,tuning_time):
        ts = []
        ts.append(self.end)
        time_zero = time.time()

        best_compile_command = None
        

        prev_best = self.best  # 记录上一次的最优解

        iteration = 1  # Initialize the iteration counter

        while ts[-1] < tuning_time:
            selectpop = self.selection(self.pop, 0.5 * 2)
            nextoff = []
            while len(nextoff) != 2:
                offspring = [random.choice(selectpop) for i in range(2)]
                crossoff = self.crossoperate(offspring)
                muteoff = self.mutation(crossoff)
                fit_muteoff, opt = self.get_objective_score(self.compiler,muteoff,100086, self.SOURCE_PATH,  LOG_FILE=self.LOG_FILE, all_flags = self.options, exec_param=self.EXEC_PARAM)
                nextoff.append((muteoff, -fit_muteoff))

                 # Update best performance, flags, and compile command
                # if best_performance is None or fit_muteoff < best_performance:
                #     best_performance = fit_muteoff
                #     best_flags = muteoff
                #     best_compile_command = opt
                #     write_log("Iteration {}: Best Performance: {}, Best Sequence: {}".format(iteration, fit_muteoff, muteoff), self.LOG_FILE)
            self.pop = nextoff       
            #从大到小排
            self.pop = sorted(self.pop, key=lambda x:x[1])
            new_best = self.selectBest(self.pop)
            # self.best = self.selectBest(self.pop)
            #print(new_best[1])
            #print(-new_best[1])
            if new_best[1] < prev_best[1]:
                self.best = new_best
                prev_best = new_best  # 更新记录的最优解
                best_compile_command = opt
                write_log("Iteration {}: Best Performance: {}, Best Sequence: {}".format(iteration, -self.best[1], self.best[0]), self.LOG_FILE)

            ts.append(time.time() - time_zero + self.end)
            #self.dep.append(1.0/self.best[1])
            #ss = '{}: best-per {}, best-seq {}'.format(str(round(ts[-1])), str(self.best[1]), str(self.best[0]))
            #write_log(str(ss),self.LOG_FILE)
            
            iteration += 1  # Increment the iteration counter

        #返回值修改！
        return -self.best[1], self.best[0], best_compile_command