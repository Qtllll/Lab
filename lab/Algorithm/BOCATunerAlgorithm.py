import argparse
import os,sys,json
import random, time, copy,subprocess
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import norm
from AlgorithmInterface import AlgorithmInterface

def write_log(ss, file):
    """ Write to log """
    with open(file, 'a') as log:
        log.write(ss + '\n')

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
#     # print(f"Compile command: {compile_command}")
#     compiler.compile(compile_command)
#     # 执行生成的可执行文件
#     compiler.execute(exec_param)
#     execute_terminal_command("rm -rf *.o *.I *.s a.out")
#     exec_time_end = time.time()
#     # 记录执行时间
#     time_c = exec_time_end - exec_time_start
    

#     # 使用 -O3 优化进行基准对比
#     o3_time_start = time.time()
#     compiler.compile(compiler.get_compile_command("-O3", source_file))
#     compiler.execute(exec_param)
#     execute_terminal_command("rm -rf *.o *.I *.s a.out")
#     o3_time_end = time.time()

#     time_o3_c = o3_time_end - o3_time_start
#     #记录的统一
#     op_str = "iteration:{} speedup:{}".format(str(k_iter), str(time_o3_c /time_c))
#     #write_log(op_str, LOG_FILE)
#     return (time_o3_c /time_c), opt
# 
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


class get_exchange(object):
    def __init__(self, incumbent):
        self.incumbent = incumbent  # fix values of impactful opts

    def to_next(self, opt_ids, l):
        """
        Flip selected less-impactful opt, then fix impactful optimization
        """
        ans = [0] * l
        for f in opt_ids:
            ans[f] = 1
        for f in self.incumbent:
            ans[f[0]] = f[1]
        return ans
class BOCAAlgorithm(AlgorithmInterface):
  def __init__(self, compiler, source_path, flags, log_file, exec_param=None):
    
    self.tuner = BOCA(
            s_dim=len(flags),
            get_objective_score=get_objective_score,
            seed=456,no_decay=False,
            source_path=source_path,#源代码
            gcc_path=compiler.get_path(),
            exec_param=exec_param,
            log_file=log_file,#不一定要有
            flags=flags,
            compiler=compiler,
            fnum=8, decay=0.5, scale=10, offset=20,
            selection_strategy=['boca', 'local'][0], initial_sample_size=2
      )
  def optimize(self, compiler, source_file, log_file, tuning_time):
      best_performance, best_flags, best_compile_command = self.tuner.run(tuning_time)
      return best_performance, best_flags, best_compile_command

  def log_results(self):
        # You can implement this to log the final results of the optimization
      pass

class BOCA:
    def __init__(self, s_dim, get_objective_score, seed ,no_decay,
                 source_path, gcc_path, exec_param, log_file, flags,compiler,
                 fnum=8, decay=0.5, scale=10, offset=20,
                 selection_strategy=['boca', 'local'][0], initial_sample_size=2):
        self.s_dim = s_dim
        self.get_objective_score = get_objective_score
        self.seed = seed

        self.fnum = fnum  # FNUM, number of impactful option
        if no_decay:
            self.decay = False
        else:
            self.decay = decay  # DECAY
            self.scale = scale  # SCALE
            self.offset = offset  # OFFSET
        self.rnum0 = 2**8  # base-number of less-impactful option-sequences, will decay

        self.selection_strategy = selection_strategy 
        self.initial_sample_size = initial_sample_size

        self.SOURCE_PATH = source_path
        self.GCC_PATH = gcc_path
        #self.INCLUDE_PATH = include_path
        self.EXEC_PARAM = exec_param
        self.LOG_FILE = log_file
        self.all_flags = flags
        self.compiler = compiler

    def generate_random_conf(self, x):
        """
        Generation 0-1 mapping for disable-enable options
        """

        comb = bin(x).replace('0b', '')
        comb = '0' * (self.s_dim - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def get_ei(self, preds, eta):
        """
        Compute Expected Improvements. (eta is global best indep)
        """
        preds = np.array(preds).transpose(1, 0)
        m = np.mean(preds, axis=1)
        s = np.std(preds, axis=1)
        # print('m:' + str(m))
        # print('s:' + str(s))

        def calculate_f(eta, m, s):
            z = (eta - m) / s
            return (eta - m)*norm.cdf(z) + s * norm.pdf(z)
        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f(eta, m, s)

        return f

    def boca_search(self, model, eta, rnum):
        """
        Get 2**fnum * rnum candidate optimization sequences,
        then compute Expected Improvement.

        :return: 2**fnum  * rnum-size list of [EI, seq]
        """
        options = model.feature_importances_
        begin = time.time()
        opt_sort = [[i, x] for i, x in enumerate(options)]
        opt_selected = sorted(opt_sort, key=lambda x: x[1], reverse=True)[:self.fnum]
        opt_ids = [x[0] for x in opt_sort]
        neighborhood_iterators = []

        for i in range(2**self.fnum):  # search all combinations of impactful optimization
            comb = bin(i).replace('0b', '')
            comb = '0' * (self.fnum - len(comb)) + comb  # fnum-size 0-1 string
            inc = []  # list of tuple: (opt_k's idx, enable/disable)
            for k,s in enumerate(comb):
                if s == '1':
                    inc.append((opt_selected[k][0], 1))
                else:
                    inc.append((opt_selected[k][0], 0))
            neighborhood_iterators.append(get_exchange(inc))
        #print('get impactful opt seq, using ' + str(time.time() - begin)+' s.')
        b2 = time.time()
        neighbors = []  # candidate seq
        for i, inc in enumerate(neighborhood_iterators):
            for _ in range(1 + rnum):
                flip_n = random.randint(0, self.s_dim)
                selected_opt_ids = random.sample(opt_ids, flip_n)
                neighbor_iter = neighborhood_iterators[i].to_next(selected_opt_ids, self.s_dim)
                neighbors.append(neighbor_iter)
        #print('get '+str(len(neighbors))+' candidate seq, using '+str(time.time()-b2))

        preds = []
        estimators = model.estimators_
        b3 = time.time()
        for e in estimators:
            preds.append(e.predict(np.array(neighbors)))
        acq_val_incumbent = self.get_ei(preds, eta)
        #print('get EI, using '+str(time.time() - b3)+' s.')

        return [[i,a] for a, i in zip(acq_val_incumbent, neighbors)]

    def get_training_sequence(self, training_indep, training_dep, eta, rnum):
        model = RandomForestRegressor(random_state=self.seed)
        model.fit(np.array(training_indep), np.array(training_dep))

        # get candidate seqs and corresponding EI
        begin = time.time()
        if self.selection_strategy == 'local':
            # print('local search')
            estimators = model.estimators_
            preds = []
            for e in estimators:
                preds.append(e.predict(training_indep))
            train_ei = self.get_ei(preds, eta)
            configs_previous_runs = [(x, train_ei[i]) for i, x in enumerate(training_indep)]
            configs_previous_runs_sorted = sorted(configs_previous_runs, key=lambda x: x[1], reverse=True)[:10]
            merged_predicted_objectives = self.local_search(model, eta, configs_previous_runs_sorted)
        else:
            # print('boca search')
            merged_predicted_objectives = self.boca_search(model, eta, rnum)
        merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
        end = time.time()
        #print('search time: ' + str(end-begin))
        #print('num: ' + str(len(merged_predicted_objectives)))

        # return unique seq in candidate set with highest EI
        begin = time.time()
        for x in merged_predicted_objectives:
            if x[0] not in training_indep:
                #print('get unique seq, using ' + str(time.time() - begin)+' s.')
                return x[0], x[1], len(merged_predicted_objectives)
    
   
    def run(self,tuning_time):
        """
        Run BOCA algorithm

        :return:
        """
        best_compile_command = None
        training_indep = []
        ts = []  # time spend
        begin = time.time()
        # randomly sample initial training instances
        while len(training_indep) < self.initial_sample_size:
            x = random.randint(0, 2**self.s_dim)
            initial_training_instance = self.generate_random_conf(x)
            # print(x, 2**self.s_dim,initial_training_instance)

            if initial_training_instance not in training_indep:
                training_indep.append(initial_training_instance)
                ts.append(time.time() - begin)

        training_dep = [self.get_objective_score(self.compiler, indep, 0,self.SOURCE_PATH, LOG_FILE=self.LOG_FILE, all_flags = self.all_flags, exec_param = self.EXEC_PARAM)[0] for indep in training_indep]
        write_log(str(training_dep), self.LOG_FILE)
        steps = 2
        merge = zip(training_indep, training_dep)
        merge_sort = [[indep, dep] for indep, dep in merge]
        merge_sort = sorted(merge_sort, key=lambda m: abs(m[1]), reverse=True)
        global_best_dep = merge_sort[0][1]  # best objective score
        global_best_indep = merge_sort[0][0]  # corresponding indep
        if self.decay:
            sigma = -self.scale ** 2 / (2 * math.log(self.decay))  # sigma = - scale^2 / 2*log(decay)
        else:
            sigma = None
        
        it=0

        while ts[-1] < tuning_time:
            steps += 1
            if self.decay:
                rnum = int(self.rnum0) * math.exp(-max(0, len(training_indep) - self.offset) ** 2 / (2*sigma**2))  # decay
            else:
                rnum = int(self.rnum0)
            rnum = int(rnum)
            # get best optimimzation sequence
            best_solution, _, num = self.get_training_sequence(training_indep, training_dep, global_best_dep, rnum)
            ts.append(time.time()-begin)
            
            # add to training set, record time spent, score for this sequence
            training_indep.append(best_solution)
            best_result, opt = self.get_objective_score(self.compiler, best_solution, k_iter=(self.initial_sample_size+steps),source_file=self.SOURCE_PATH, LOG_FILE=self.LOG_FILE, all_flags = self.all_flags, exec_param=self.EXEC_PARAM)
            training_dep.append(best_result)
            if abs(best_result) > abs(global_best_dep):
                global_best_dep = best_result
                global_best_indep = best_solution
                #最佳编译命令
                best_compile_command=opt
                write_log("Iteration {}: Best Performance: {}, Best Sequence: {}".format(it, global_best_dep, global_best_indep), self.LOG_FILE)

            ss = '{}: best {}, cur-best {}, independent-number {} , solution {}'.format(str(round(ts[-1])),
                                                                                    str(global_best_dep),
                                                                                    str(best_result),
                                                                                    str(num),
                                                                                    str(best_solution))
            it+=1
            write_log(ss, self.LOG_FILE)
        write_log(str(global_best_indep)+'\n=======================\n', self.LOG_FILE)
        #返回最好的情况
        return global_best_dep, global_best_indep, best_compile_command