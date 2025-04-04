import argparse
import subprocess
import random
import time
import os
import re
import glob
import numpy as np
from AlgorithmInterface import AlgorithmInterface

def write_log(ss, file):
    """ Write to log """
    with open(file, 'a') as log:
        log.write(ss + '\n')

# ------------------------------
# 辅助函数：执行终端命令
# ------------------------------
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

# ------------------------------
# 辅助函数：性能评估（获得加速比）越大越好
# ------------------------------
# def get_objective_score(independent, k_iter, SOURCE_PATH, GCC_PATH, INCLUDE_PATH, EXEC_PARAM, LOG_FILE, all_flags):
#     """Obtain the speedup"""
#     # 构造编译参数字符串
#     opt = ''
#     for i in range(len(independent)):
#         if independent[i]:
#             opt = opt + all_flags[i] + ' '
#         else:
#             negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
#             opt = opt + negated_flag_name + ' '
    
#     # 候选方案编译、链接、运行（使用 -O2）
#     command = f"{GCC_PATH} -O2 {opt} -c {INCLUDE_PATH} {SOURCE_PATH}/*.c"
#     execute_terminal_command(command)
#     command2 = f"{GCC_PATH} -o a.out -O2 {opt} -lm *.o"
#     execute_terminal_command(command2)

#     time_start = time.time()
#     command3 = f"./a.out {EXEC_PARAM}"
#     execute_terminal_command(command3)
#     time_end = time.time()
#     cmd4 = 'rm -rf *.o *.I *.s a.out'
#     execute_terminal_command(cmd4)
#     time_c = time_end - time_start   # 候选方案执行时间

#     # 基线方案编译、链接、运行（使用 -O3）
#     time_o3 = time.time()
#     command = f"{GCC_PATH} -O3 -c {INCLUDE_PATH} {SOURCE_PATH}/*.c"
#     execute_terminal_command(command)
#     command2 = f"{GCC_PATH} -o a.out -O3 -lm *.o"
#     execute_terminal_command(command2)
#     command3 = f"./a.out {EXEC_PARAM}"
#     execute_terminal_command(command3)
#     time_o3_end = time.time()
#     cmd4 = 'rm -rf *.o *.I *.s a.out'
#     execute_terminal_command(cmd4)
#     time_o3_c = time_o3_end - time_o3   # 基线执行时间

#     op_str = "iteration:{} speedup:{}".format(str(k_iter), str(time_o3_c / time_c))
#     write_log(op_str, LOG_FILE)
#     return (time_o3_c / time_c)
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

# ------------------------------
# 特征提取相关函数
# ------------------------------
# 定义各类优化选项
loop_flags = ['-floop-interchange', '-floop-unroll-and-jam', '-fpeel-loops', '-fsplit-loops',
              '-funswitch-loops', '-fmove-loop-invariants', '-ftree-loop-distribute-patterns',
              '-ftree-loop-distribution', '-ftree-loop-im', '-ftree-loop-optimize', '-ftree-loop-vectorize',
              '-fversion-loops-for-strides', '-fsplit-paths']
branch_flags = ['-fthread-jumps', '-fif-conversion', '-fif-conversion2', '-fhoist-adjacent-loads',
                '-fprintf-return-value', '-ftree-tail-merge', '-fguess-branch-probability', '-fcse-follow-jumps',
                '-ftree-dominator-opts', '-freorder-blocks', '-freorder-blocks-and-partition', '-ftree-ch']
function_flags = ['-fipa-sra', '-ftree-pta', '-ftree-builtin-call-dce', '-fshrink-wrap',
                  '-freorder-functions', '-fcaller-saves', '-fdefer-pop', '-fdevirtualize',
                  '-fdevirtualize-speculatively', '-ffunction-cse', '-findirect-inlining',
                  '-finline-functions', '-finline-functions-called-once', '-finline-small-functions',
                  '-fipa-cp-clone', '-fipa-icf-functions', '-fipa-modref', '-fipa-profile', '-fipa-pure-const',
                  '-fipa-ra', '-fpartial-inlining']
static_variable_flags = ['-fipa-reference', '-fipa-reference-addressable', '-ftoplevel-reorder',
                         '-fipa-icf-variables', '-ftree-bit-ccp', '-ftree-ccp', '-ftree-coalesce-vars']
pointer_flags = ['-fisolate-erroneous-paths-dereference', '-fomit-frame-pointer', '-ftree-vrp']
string_flags = ['-foptimize-strlen']
float_flags = ['-fsigned-zeros']

def obtain_c_code(file_path):
    """
    从 source_path 中读取所有 .c 文件（排除 loop-wrap.c），合并为一个字符串
    """
    c_code = ""
    pattern = os.path.join(file_path, '*.c')
    for file in glob.glob(pattern):
        filename = os.path.basename(file)
        if filename != 'loop-wrap.c':
            with open(file, 'r') as f:
                c_code += f.read() + "\n"
    return c_code

def remove_commentsandinclude_from_c_code(c_code):
    """
    去除代码中的注释和 #include 部分，返回纯净代码
    """
    c_code = re.sub(r'/\*.*?\*/', '', c_code, flags=re.DOTALL)
    c_code = re.sub(r'//.*', '', c_code)
    c_code = re.sub(r'".*?"', lambda x: x.group(0) if '/*' not in x.group(0) else '', c_code)
    c_code = re.sub(r'#include\s*<.*?>', '', c_code)
    c_code = re.sub(r'#include\s*".*?"', '', c_code)
    return "\n".join([line for line in c_code.split('\n') if line.strip() != ''])

def contain_loop(code):
    """统计代码中循环结构数量"""
    for_loop_pattern = r"for\s*\(([^)]+)\)\s*{?"
    while_loop_pattern = r"while\s*\(([^)]+)\)\s*{?"
    do_while_loop_pattern = r"do\s*{?[^}]*}\s*while\s*\(([^)]+)\)"
    for_loops = re.findall(for_loop_pattern, code)
    while_loops = re.findall(while_loop_pattern, code)
    do_while_loops = re.findall(do_while_loop_pattern, code)
    matches = list(set(for_loops + while_loops + do_while_loops))
    return len(matches)

def contain_branch(code):
    """统计代码中分支结构数量"""
    pattern = r'if\s*\(.*?\)|else\s*if\s*\(.*?\)|else|switch\s*\(.*?\)'
    matches = list(set(re.findall(pattern, code)))
    return len(matches)

def contain_function(code):
    """统计代码中函数调用或声明数量（排除 main 及常见类型）"""
    function_call_pattern = r'\b\w+\s*\([^)]*\)'
    function_declaration_pattern = r'\b\w+\s+\w+\s*\([^)]*\)'
    function_calls = re.findall(function_call_pattern, code)
    function_declarations = re.findall(function_declaration_pattern, code)
    function_calls_names = set([re.match(r'\b\w+', call).group() for call in function_calls])
    function_declarations_names = set([re.match(r'\b\w+\s+(\w+)', decl).group(1) for decl in re.findall(function_declaration_pattern, code)])
    matched_functions = function_calls_names.intersection(function_declarations_names)
    for name in ['main', 'int', 'float', 'double', 'string', 'long']:
        matched_functions.discard(name)
    return len(matched_functions)

def contain_static_variable(code):
    """统计代码中 static 变量声明数量"""
    pattern = r'\bstatic\s+\w+\s+\w+\s*=?\s*[^;]*'
    matches = re.findall(pattern, code)
    return len(matches)

def contain_pointer(code):
    """统计代码中指针声明数量"""
    pattern = r'\b([_a-zA-Z][_a-zA-Z0-9]*\s+\*+\s*[_a-zA-Z][_a-zA-Z0-9]*\s*);'
    matches = list(set(re.findall(pattern, code)))
    return len(matches)

def contain_string(code):
    """统计代码中字符串函数调用数量"""
    pattern = r'\b(str(?:len|cpy|ncpy|cat|ncat|cmp|ncmp|chr|rchr|str|tok|dup|ncpy))\b'
    matches = re.findall(pattern, code)
    return len(matches)

def contain_float_calculation(code):
    """统计代码中浮点数常量数量"""
    float_pattern = r'[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?'
    matches = re.findall(float_pattern, code)
    return len(matches)

def compute_feature_weights(code):
    """
    计算各特征出现次数并归一化，返回字典{'loop': weight, 'branch': weight, ...}
    """
    weights = {
        'loop': contain_loop(code),
        'branch': contain_branch(code),
        'function': contain_function(code),
        'static_var': contain_static_variable(code),
        'pointer': contain_pointer(code),
        'string': contain_string(code),
        'float': contain_float_calculation(code),
    }
    total = sum(weights.values()) + 1e-6  # 加上一个小数防止除零错误
    normalized_weights = {k: v / total for k, v in weights.items()}  # 归一化
    return normalized_weights

def read_flags_from_file(file_path):
    """
    从文件中读取候选编译选项（逗号分隔），返回列表
    """
    with open(file_path, 'r') as file:
        flags = file.read().strip()
    return [flag.strip() for flag in flags.split(',') if flag.strip()]

class APCOTunerAlgorithm(AlgorithmInterface):
  def __init__(self, compiler, source_path, flags, log_file, exec_param=None):
    
    self.tuner = APCO(
            source_path=source_path,#源代码
            gcc_path=compiler.get_path(),
            # include_path=?,
            exec_param=exec_param,
            log_file=log_file,#不一定要有
            all_flags=flags,
            compiler=compiler,
      )
  def optimize(self, compiler, source_file, log_file, tuning_time):
      best_performance, best_flags, best_compile_command = self.tuner.search(tuning_time)
      return best_performance, best_flags, best_compile_command

  def log_results(self):
        # You can implement this to log the final results of the optimization
      pass
  

class APCO:
    def __init__(self, source_path, gcc_path, exec_param, log_file, all_flags,compiler):
        self.source_path = source_path
        #self.iterations = iterations

         # 全局参数（请根据实际环境修改）
        self.GCC_PATH = gcc_path
        
        self.EXEC_PARAM = exec_param  # 程序执行参数
        self.LOG_FILE = log_file
        self.compiler=compiler
        

        # 读取所有候选优化标志
        self.all_flags = all_flags
        # 将所有标志按类别组织
        self.flag_categories = {
            'loop': loop_flags,
            'branch': branch_flags,
            'function': function_flags,
            'static_var': static_variable_flags,
            'pointer': pointer_flags,
            'string': string_flags,
            'float': float_flags
        }
        
        # 读取代码、去除注释，计算特征权重
        code = obtain_c_code(self.source_path)
        cleaned_code = remove_commentsandinclude_from_c_code(code)
        self.feature_weights = compute_feature_weights(cleaned_code)
        """
        self.feature_weights可能的样子
        {'loop': 0.4, 'branch': 0.3, 'function': 0.1, 'pointer': 0.2, ...}
        """
        # 根据各类别权重为候选选项初始化采样概率
        self.probabilities = {}
        for category, flags in self.flag_categories.items():
            weight = self.feature_weights.get(category, 0)
            """
            计算该类别的优化标志采样概率，weight * 2 乘2随便取的让权重更明显。
            使用 np.clip(x, min, max) 限制概率范围在 [0.3, 1.0]，避免过低或过高。
            """
            p = np.clip(weight * 2, 0.3, 1.0)
            for flag in flags:
                if flag in self.all_flags:
                    self.probabilities[flag] = p
        for flag in self.all_flags:
            if flag not in self.probabilities:
                self.probabilities[flag] = 0.5

        self.best_candidate = None
        self.best_perf = 0  # 性能指标越大越好
        self.history = []  # 保存 (候选组合, 性能) 记录

       

    def generate_candidate(self):
        """
        修改部分：
        原来返回选中flag的列表，现在返回一个布尔列表，长度等于 self.all_flags，
        每个位置True表示启用该flag，False表示禁用。
        """
        candidate = [random.random() < self.probabilities.get(flag, 0.5) for flag in self.all_flags]
        return candidate

    def evaluate_candidate(self, candidate, iter):
        """
        根据候选组合构造编译参数，并编译、链接、运行程序：
          1. 对每个 all_flags，如果候选中对应位置为True，则直接使用，否则构造否定选项。
          2. 使用 gcc -O2 编译、链接、运行候选方案，记录执行时间。
          3. 使用 gcc -O3 作为基线方案，同样编译、链接、运行，记录执行时间。
          4. 计算加速比 speedup = (baseline_time / candidate_time)，返回该指标。
        """
        # candidate现在是布尔列表，生成独立1/0列表
        independent = [1 if flag_enabled else 0 for flag_enabled in candidate]
        speedup,opt = get_objective_score(
            compiler=self.compiler,
            independent=independent,
            k_iter=iter,
            source_file=self.source_path,
            LOG_FILE=self.LOG_FILE,
            all_flags=self.all_flags,
            exec_param=self.EXEC_PARAM,
        )
        return speedup,opt

    def update_probabilities_detailed(self, prev_candidate, prev_perf, curr_candidate, curr_perf):
      """
    更细粒度地更新采样概率：
      对于每个选项：
      1. 如果状态发生变化，按照原先逻辑处理：
         - 从禁用变为启用：
             如果性能提升（curr_perf > prev_perf），上调该选项概率；
             否则下调该选项概率。
         - 从启用变为禁用：
             如果性能提升，降低该选项概率；
             否则提高该选项概率。
      2. 如果状态未改变：
         - 如果整体性能提升，则增加该选项的采样概率；
         - 如果整体性能下降，则降低该选项的采样概率；
         - 如果性能无变化，则不做调整。
      """
      delta = 0.05  # 固定步长
      for i, flag in enumerate(self.all_flags):
          if prev_candidate is None:
              continue
          if curr_candidate[i] != prev_candidate[i]:
              # 状态发生变化的情况
               # 从禁用变为启用
              if not prev_candidate[i] and curr_candidate[i]:
                  if curr_perf > prev_perf:
                      self.probabilities[flag] = min(self.probabilities.get(flag, 0.5) + delta, 1.0)
                  else:
                      self.probabilities[flag] = max(self.probabilities.get(flag, 0.5) - delta, 0.1)
              # 从启用变为禁用
              elif prev_candidate[i] and not curr_candidate[i]:
                  if curr_perf > prev_perf:
                      self.probabilities[flag] = max(self.probabilities.get(flag, 0.5) - delta, 0.1)
                  else:
                      self.probabilities[flag] = min(self.probabilities.get(flag, 0.5) + delta, 1.0)
          else:
              # 状态未改变的情况
              if curr_perf > prev_perf:
                  self.probabilities[flag] = min(self.probabilities.get(flag, 0.5) + delta, 1.0)
              elif curr_perf < prev_perf:
                  self.probabilities[flag] = max(self.probabilities.get(flag, 0.5) - delta, 0.1)
              # 如果性能无变化，则不更新该选项概率
    def search(self, tuning_time):
      """
      进行基于时间的搜索，生成候选组合、评估、更新采样概率，返回最佳候选组合和编译命令。
      """
      ts = []
      time_zero = time.time()
      ts.append(0)

      prev_candidate = None
      prev_perf = self.best_perf  # 初始最佳性能

      best_compile_command = None  # 记录最佳编译命令

      iteration = 0
      while ts[-1] < tuning_time:
          candidate = self.generate_candidate()  # 生成候选 flag 组合（布尔列表）
          perf, compile_command = self.evaluate_candidate(candidate, iteration)  # 评估性能 & 获取编译命令

          self.history.append((candidate, perf))
          self.update_probabilities_detailed(prev_candidate, prev_perf, candidate, perf)

          if perf > self.best_perf:
              self.best_perf = perf
              self.best_candidate = candidate
              best_compile_command = compile_command  # 记录最佳编译命令
              write_log("Iteration {}: Best Performance: {}, Best Sequence: {}".format(iteration, self.best_perf, self.best_candidate), self.LOG_FILE)
              #print(self.best_perf)
          prev_candidate = candidate
          prev_perf = perf

          iteration += 1
          ts.append(time.time() - time_zero)

    # 最终输出最佳候选方案
      #best_flags = [flag for idx, flag in enumerate(self.all_flags) if self.best_candidate[idx]]
      best_flags = [1 if flag else 0 for flag in self.best_candidate]
      print(best_flags)
      #print(f"Final best candidate flags: {best_flags}, Best perf: {self.best_perf}")

      return self.best_perf, best_flags, best_compile_command


    # def search(self,tuning_time):
    #     """
    #     进行多次迭代搜索，生成候选组合、评估、更新采样概率，返回最佳候选组合。
    #     修改部分：在输出时将布尔列表转换为候选 flags 的字符串形式，
    #     如果对应位置为True，则显示该 flag，否则显示否定形式。
    #     """
    #     prev_candidate = None
    #     prev_perf = self.best_perf  # 初始最佳性能
    #     best_compile_command = None
        
    #     for i in range(tuning_time):
    #         candidate = self.generate_candidate()  # 布尔列表
    #         perf, opt = self.evaluate_candidate(candidate, i)
    #         self.history.append((candidate, perf))
    #         self.update_probabilities_detailed(prev_candidate, prev_perf, candidate, perf)
    #         if perf > self.best_perf:
    #             self.best_perf = perf
    #             self.best_candidate = candidate
    #             best_compile_command = opt
    #         prev_candidate = candidate
    #         prev_perf = perf
           
    #     # 最终输出最佳候选方案中启用的选项
    #     best_flags = [flag for idx, flag in enumerate(self.all_flags) if self.best_candidate[idx]]
    #     print(f"Final best candidate flags:{best_flags}, Best perf{self.best_perf}")
        
    #     return self.best_perf, best_flags, best_compile_command