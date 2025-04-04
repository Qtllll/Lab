import os,sys
import random, time, copy,subprocess, argparse
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import norm
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
#     # 执行生成的可执行文件
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
#     write_log(op_str, LOG_FILE)
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
#    # write_log(op_str, LOG_FILE)
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




ts_tem = []  # time consumption

class CompTunerAlgorithm(AlgorithmInterface):
  def __init__(self, compiler, source_path, flags, log_file, exec_param=None):
    
    self.tuner = compTuner(
            dim=len(flags),
            c1=2, c2=2, w=0.6,
            get_objective_score=get_objective_score,
            random=456,
            source_path=source_path,#源代码
            gcc_path=compiler.get_path(),
            exec_param=exec_param,
            log_file=log_file,#不一定要有
            flags=flags,
            compiler=compiler,
      )
  def optimize(self, compiler, source_file, log_file, tuning_time):
      best_performance, best_flags, best_compile_command = self.tuner.run(tuning_time)
      return best_performance, best_flags, best_compile_command

  def log_results(self):
        # You can implement this to log the final results of the optimization
      pass


class compTuner:
    def __init__(self, dim, c1, c2, w, get_objective_score, random, source_path, gcc_path, exec_param, log_file, flags,compiler):
        """
        :param dim: number of compiler flags
        :param c1: parameter of pso process
        :param c2: parameter of pso process
        :param w: parameter of pso process
        :param get_objective_score: obtain true speedup
        :param random: random parameter
        :param source_path: program's path
        :param gcc_path: gcc's path
        :param exec_param: exection paramter
        :param log_file: record results
        :param flags: all flags
        :param compiler: compiler
        """
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.dim = dim
        self.V = []
        self.pbest = [] # best vector of each particle
        self.gbest = [] # best performance of each particle
        self.p_fit = [] # best vector of all particles
        self.fit = 0 # best performance of all particles
        self.get_objective_score = get_objective_score 
        self.random = random
        self.SOURCE_PATH = source_path
        self.GCC_PATH = gcc_path
        self.EXEC_PARAM = exec_param
        self.LOG_FILE = log_file
        self.all_flags = flags
        self.compiler = compiler

    def generate_random_conf(self, x):
        """
        :param x: random generate number
        :return: the binary sequence for x
        """
        comb = bin(x).replace('0b', '')
        comb = '0' * (self.dim - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def get_ei(self, preds, eta):
        """
        :param preds: sequences' speedup for EI
        :param eta: global best speedup
        :return: the EI for a sequence
        """
        preds = np.array(preds).transpose(1, 0)
        m = np.mean(preds, axis=1)
        s = np.std(preds, axis=1)

        def calculate_f(eta, m, s):
            z = (eta - m) / s
            return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f(eta, m, s)
        return f
    
    def get_ei_predict(self, model, now_best, wait_for_train):
        """
        :param model: RandomForest Model
        :param now_best: global best speedup
        :param wait_for_train: sequences set
        :return: the sequences' EI and the sequences
        """
        preds = []
        estimators = model.estimators_
        for e in estimators:
            preds.append(e.predict(np.array(wait_for_train)))
        acq_val_incumbent = self.get_ei(preds, now_best)
        return [[i, a] for a, i in zip(acq_val_incumbent, wait_for_train)]

    def runtime_predict(self, model, wait_for_train):
        """
        :param model: RandomForest Model
        :param wait_for_train: sequences Set
        :return: the speedup of sequences Set
        """
        estimators = model.estimators_
        sum_of_predictions = np.zeros(len(wait_for_train))
        for tree in estimators:
            predictions = tree.predict(wait_for_train)
            sum_of_predictions += predictions
        a = []
        average_prediction = sum_of_predictions / len(estimators)
        for i in range(len(wait_for_train)):
            x = [wait_for_train[i], average_prediction[i]]
            a.append(x)
        return a
    
    def getPrecision(self, model, seq):
        """
        :param model: RandomForest Model
        :param seq: sequence
        :return: the precision of a sequence and true speedup
        """
        true_running= self.get_objective_score(self.compiler, seq, k_iter=100086, source_file=self.SOURCE_PATH,LOG_FILE=self.LOG_FILE, all_flags = self.all_flags, exec_param=self.EXEC_PARAM)[0]
        estimators = model.estimators_
        res = []
        for e in estimators:
            tmp = e.predict(np.array(seq).reshape(1, -1))
            res.append(tmp)
        acc_predict = np.mean(res)
        return abs(true_running - acc_predict) / true_running, true_running
    
    def selectByDistribution(self, merged_predicted_objectives):
        """
        :param merged_predicted_objectives: the sequences' EI and the sequences
        :return: the selected sequence
        """
        diffs = [abs(perf - merged_predicted_objectives[0][1]) for seq, perf in merged_predicted_objectives]
        diffs_sum = sum(diffs)
        probabilities = [diff / diffs_sum for diff in diffs]
        index = list(range(len(diffs)))
        idx = np.random.choice(index, p=probabilities)
        return idx
    
    def build_RF_by_CompTuner(self):
        """
        :return: model, initial_indep, initial_dep
        """
        inital_indep = []
        # randomly sample initial training instances
        time_begin = time.time()
        while len(inital_indep) < 2:
            x = random.randint(0, 2 ** self.dim - 1)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in inital_indep:
                inital_indep.append(initial_training_instance)
        initial_dep = [self.get_objective_score(self.compiler, indep, k_iter=0, source_file=self.SOURCE_PATH, LOG_FILE=self.LOG_FILE, all_flags = self.all_flags, exec_param=self.EXEC_PARAM)[0] for indep in inital_indep] # initialization
        ts_tem.append(time.time() - time_begin)
        ss = '{}: best_seq {}, best_per {}'.format(str(round(ts_tem[-1])), str(max(initial_dep)), str(inital_indep[initial_dep.index(max(initial_dep))]))
        write_log(ss, self.LOG_FILE)
        all_acc = []
        model = RandomForestRegressor(random_state=self.random)
        model.fit(np.array(inital_indep), np.array(initial_dep))
        rec_size = 2
        while rec_size < 50:
            global_best = max(initial_dep)
            estimators = model.estimators_
            neighbors = []
            while len(neighbors) < 30000:
                x = random.randint(0, 2 ** self.dim - 1)
                x = self.generate_random_conf(x)
                if x not in neighbors:
                    neighbors.append(x)
            pred = []
            for e in estimators:
                pred.append(e.predict(np.array(neighbors)))
            acq_val_incumbent = self.get_ei(pred, global_best)
            ei_for_current = [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]
            merged_predicted_objectives = sorted(ei_for_current, key=lambda x: x[1], reverse=True)
            acc = 0
            flag = False
            for x in merged_predicted_objectives:
                if flag:
                    break
                if x[0] not in inital_indep:
                    inital_indep.append(x[0])
                    acc, lable = self.getPrecision(model, x[0])
                    initial_dep.append(lable)
                    all_acc.append(acc)
                    flag = True
            rec_size += 1
            if acc > 0.05:
                indx = self.selectByDistribution(merged_predicted_objectives)
                while merged_predicted_objectives[indx][0] in inital_indep:
                    indx = self.selectByDistribution(merged_predicted_objectives)
                inital_indep.append(merged_predicted_objectives[indx][0])
                acc, label = self.getPrecision(model, merged_predicted_objectives[int(indx)][0])
                initial_dep.append(label)
                all_acc.append(acc)
                rec_size += 1
            ts_tem.append(time.time() - time_begin)
            ss = '{}: best_seq {}, best_per {}'.format(str(round(ts_tem[-1])), str(max(initial_dep)), str(inital_indep[initial_dep.index(max(initial_dep))]))
            write_log(ss, self.LOG_FILE)
            model = RandomForestRegressor(random_state=self.random)
            model.fit(np.array(inital_indep), np.array(initial_dep))
            if rec_size > 50 and np.mean(all_acc) < 0.04:
                break
        return model, inital_indep, initial_dep
    
    def getDistance(self, seq1, seq2):
        """
        :param seq1: compared sequence
        :param seq2: compared sequence
        :return: obtaining the diversity of two sequences
        """
        t1 = np.array(seq1)
        t2 = np.array(seq2)
        s1_norm = np.linalg.norm(t1)
        s2_norm = np.linalg.norm(t2)
        cos = np.dot(t1, t2) / (s1_norm * s2_norm)
        return cos
    
    def init_v(self, n, d, V_max, V_min):
        """
        :param n: number of particles
        :param d: number of compiler flags
        :return: particle's initial velocity vector
        """
        v = []
        for i in range(n):
            vi = []
            for j in range(d):
                a = random.random() * (V_max - V_min) + V_min
                vi.append(a)
            v.append(vi)
        return v
    
    def update_v(self, v, x, m, n, pbest, g, w, c1, c2, vmax, vmin):
        """
        :param v: particle's velocity vector
        :param x: particle's position vector
        :param m: number of partical
        :param n: number of compiler flags
        :param pbest: each particle's best position vector
        :param g: all particles' best position vector
        :param w: weight parameter
        :param c1: control parameter
        :param c2: control parameter
        :param vmax: max V
        :param vmin: min V
        :return: each particle's new velocity vector
        """
        for i in range(m):
            a = random.random()
            b = random.random()
            for j in range(n):
                v[i][j] = w * v[i][j] + c1 * a * (pbest[i][j] - x[i][j]) + c2 * b * (g[j] - x[i][j])
                if v[i][j] < vmin:
                    v[i][j] = vmin
                if v[i][j] > vmax:
                    v[i][j] = vmax
        return v
    
    def run(self,tuning_time):
        write_log("Build model",self.LOG_FILE)
        best_compile_command = None
        ts = []
        model, inital_indep, inital_dep = self.build_RF_by_CompTuner()
        begin = time.time()
        self.V = self.init_v(len(inital_indep), len(inital_indep[0]), 10, -10)
        self.fit = 0
        self.pbest = list(inital_indep)
        self.p_fit = list(inital_dep)
        for i in range(len(inital_dep)):
            tmp = inital_dep[i]
            if tmp > self.fit:
                self.fit = tmp
                self.gbest = inital_indep[i]
        end = time.time() + ts_tem[-1]
        ts.append(end - begin)
        ss = '{}: best {}, cur-best-seq {}'.format(str(round(end - begin)), str(self.fit), str(self.gbest))
        write_log(ss, self.LOG_FILE)
        t = 0
        write_log("model complete!",self.LOG_FILE)
        while ts[-1] < tuning_time:
            if t == 0:
                self.V = self.update_v(self.V, inital_indep, len(inital_indep), len(inital_indep[0]), self.pbest, self.gbest, self.w, self.c1, self.c2, 10, -10)
                for i in range(len(inital_indep)):
                    for j in range(len(inital_indep[0])):
                        a = random.random()
                        if 1.0 / (1 + math.exp(-self.V[i][j])) > a:
                            inital_indep[i][j] = 1
                        else:
                            inital_indep[i][j] = 0
                t = t + 1
            else:
                merged_predicted_objectives = self.runtime_predict(model, inital_indep)
                for i in range(len(merged_predicted_objectives)):
                    if merged_predicted_objectives[i][1] > self.p_fit[i]:
                        self.p_fit[i] = merged_predicted_objectives[i][1]
                        self.pbest[i] = merged_predicted_objectives[i][0]
                sort_merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
                current_best_seq = sort_merged_predicted_objectives[0][0]
                temp, opt= self.get_objective_score(self.compiler, current_best_seq, 1000086, source_file=self.SOURCE_PATH, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags, exec_param=self.EXEC_PARAM)
                if  temp > self.fit:
                    self.gbest = current_best_seq
                    self.fit = temp
                    #最佳编译命令
                    best_compile_command=opt
                    write_log("Iteration {}: Best Performance: {}, Best Sequence: {}".format(t, self.fit, self.gbest), self.LOG_FILE)
                    self.V = self.update_v(self.V, inital_indep, len(inital_indep), len(inital_indep[0]), self.pbest,
                                           self.gbest, self.w, self.c1, self.c2, 10, -10)
                    for i in range(len(inital_indep)):
                        for j in range(len(inital_indep[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-self.V[i][j])) > a:
                                inital_indep[i][j] = 1
                            else:
                                inital_indep[i][j] = 0
                else:
                    """
                    Different update
                    """
                    avg_dis = 0.0
                    for i in range(1, len(merged_predicted_objectives)):
                        avg_dis = avg_dis + self.getDistance(merged_predicted_objectives[i][0], current_best_seq)
                    
                    avg_dis = avg_dis / (len(inital_indep) - 1)
                    
                    better_seed_indep = []
                    worse_seed_indep = []
                    better_seed_seq = []
                    worse_seed_seq = []
                    better_seed_pbest = []
                    worse_seed_pbest = []
                    better_seed_V = []
                    worse_seed_V = []
        
                    for i in range(0, len(merged_predicted_objectives)):
                        if self.getDistance(merged_predicted_objectives[i][0], current_best_seq) > avg_dis:
                            worse_seed_indep.append(i)
                            worse_seed_seq.append(merged_predicted_objectives[i][0])
                            worse_seed_pbest.append(self.pbest[i])
                            worse_seed_V.append(self.V[i])
                        else:
                            better_seed_indep.append(i)
                            better_seed_seq.append(merged_predicted_objectives[i][0])
                            better_seed_pbest.append(self.pbest[i])
                            better_seed_V.append(self.V[i])
                    """
                    update better particles
                    """
                    V_for_better = self.update_v(better_seed_V, better_seed_seq, len(better_seed_seq),
                                                 len(better_seed_seq[0]), better_seed_pbest, self.gbest
                                                 , self.w, 2 * self.c1, self.c2, 10, -10)
                    for i in range(len(better_seed_seq)):
                        for j in range(len(better_seed_seq[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-V_for_better[i][j])) > a:
                                better_seed_seq[i][j] = 1
                            else:
                                better_seed_seq[i][j] = 0
                    """
                    update worse particles
                    """
                    V_for_worse = self.update_v(worse_seed_V, worse_seed_seq, len(worse_seed_seq),
                                                len(worse_seed_seq[0]), worse_seed_pbest, self.gbest
                                                , self.w, self.c1, 2 * self.c2, 10, -10)
                    for i in range(len(worse_seed_seq)):
                        for j in range(len(worse_seed_seq[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-V_for_worse[i][j])) > a:
                                worse_seed_seq[i][j] = 1
                            else:
                                worse_seed_seq[i][j] = 0
                    for i in range(len(better_seed_seq)):
                        inital_indep[better_seed_indep[i]] = better_seed_seq[i]
                    for i in range(len(worse_seed_seq)):
                        inital_indep[worse_seed_indep[i]] = worse_seed_seq[i]
                t = t + 1

            ts.append(time.time() - begin + ts_tem[-1])
            ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(ts[-1])), str(self.fit), str(self.gbest))
            write_log(ss, self.LOG_FILE)
            if (time.time() + ts_tem[-1] - begin) > 6000:
                break
                # Return best performance and best flag combination after loop ends
        return self.fit, self.gbest, best_compile_command
