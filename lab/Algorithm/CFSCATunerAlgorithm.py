import os, sys, random, time, copy, subprocess, itertools, math, argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from Algorithm.getrelated import get_related_flags, obtain_c_code, remove_commentsandinclude_from_c_code, read_flags_from_file
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
#     #write_log(op_str, LOG_FILE)
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
time_tem = []

class CFSCAAlgorithm(AlgorithmInterface):
  def __init__(self, compiler, source_path, flags, log_file, exec_param=None):
    
    #先运行generate.py
    code = obtain_c_code(source_path)
    new_code = remove_commentsandinclude_from_c_code(code)
    related_flag_indice = get_related_flags(new_code, flags)

    #print(related_flag_indice)

    if related_flag_indice is not None:
        related_flags_list = [int(x) for x in related_flag_indice]
    else:
        related_flags_list = []
    #print(related_flags_list)

    self.tuner = CFSCA(
            dim=len(flags),
            get_objective_score=get_objective_score,
            seed=456,
            related_flags=related_flags_list,
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

class CFSCA:
    def __init__(self, dim, get_objective_score, seed, related_flags, source_path, gcc_path, exec_param, log_file, flags, compiler):
        """
        :param dim: number of compiler flags
        :param get_objective_score: obtain true speedup
        :param random: random parameter
        :param related_flags: program related flags for the target program
        :param source_path: program's path
        :param gcc_path: gcc's path
        :param include_path: header file for program
        :param exec_param: exection paramter
        :param log_file: record results
        :param flags: all flags
        """
        self.dim = dim
        self.get_objective_score = get_objective_score
        self.seed = seed
        self.related = related_flags
        self.critical = []
        self.global_best_per = 0.0
        self.global_best_seq = []
        self.random = random
        self.SOURCE_PATH = source_path
        self.GCC_PATH = gcc_path
        #self.INCLUDE_PATH = include_path
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
        :return: the EI of a sequence
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
        :param wait_for_train: sequences Set
        :return: the sequences' EI
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
        :param wait_for_train: sequences set
        :return: the speedup of sequences set
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
        :param seq: sequence for prediction
        :return: the precision of a sequence and true speedup
        """
        true_running = self.get_objective_score(self.compiler,seq, 100086, self.SOURCE_PATH,self.LOG_FILE, self.all_flags,self.EXEC_PARAM)[0]
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
        # sequences = [seq for seq, per in merged_predicted_objectives]
        diffs = [abs(perf - merged_predicted_objectives[0][1]) for seq, perf in merged_predicted_objectives]
        diffs_sum = sum(diffs)
        probabilities = [diff / diffs_sum for diff in diffs]
        index = list(range(len(diffs)))
        idx = np.random.choice(index, p=probabilities)
        return idx
    
    def build_RF_by_CompTuner(self):
        """
        :return: model, inital_indep, inital_dep
        """
        inital_indep = []
        time_begin = time.time()
        # randomly sample initial training instances
        while len(inital_indep) < 2:
            x = random.randint(0, 2 ** self.dim - 1)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in inital_indep:
                inital_indep.append(initial_training_instance)
        inital_dep = [self.get_objective_score(self.compiler,indep, 0, self.SOURCE_PATH,self.LOG_FILE, self.all_flags,self.EXEC_PARAM)[0] for indep in inital_indep]
                
        all_acc = []
        time_tem.append(time.time() - time_begin)
        model = RandomForestRegressor(random_state=self.seed)
        model.fit(np.array(inital_indep), np.array(inital_dep))
        rec_size = 2
        
        while rec_size < 11:
            model = RandomForestRegressor(random_state=self.seed)
            model.fit(np.array(inital_indep), np.array(inital_dep))
            global_best = max(inital_dep)
            estimators = model.estimators_
            if all_acc:
                all_acc = sorted(all_acc)
            neighbors = []
            for i in range(30000):
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
                    inital_dep.append(lable)
                    all_acc.append(acc)
                    flag = True
            rec_size += 1

            if acc > 0.05:
                indx = self.selectByDistribution(merged_predicted_objectives)
                while merged_predicted_objectives[int(indx)][0] in inital_indep:
                    indx = self.selectByDistribution(merged_predicted_objectives)
                inital_indep.append(merged_predicted_objectives[int(indx)][0])
                acc, label = self.getPrecision(model, merged_predicted_objectives[int(indx)][0])
                inital_dep.append(label)
                all_acc.append(acc)
                rec_size += 1
            time_tem.append(time.time() - time_begin)
            ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(max(inital_dep)), str(inital_indep[inital_dep.index(max(inital_dep))]))
            write_log(ss, self.LOG_FILE)
        self.global_best_per = max(inital_dep)
        self.global_best_seq = inital_indep[inital_dep.index(max(inital_dep))]
        return model, inital_indep, inital_dep
    
    def get_critical_flags(self, model, inital_indep, inital_dep):
        """
        :param: model: RandomForest Model
        :param: inital_indep: selected sequences
        :param: inital_dep: selected sequences' performance
        :return: critical_flags_idx, new_model
        """
        candidate_seq = []
        candidate_per = []
        inital_indep_temp = copy.deepcopy(inital_indep)
        inital_dep_temp = copy.deepcopy(inital_dep)
        while len(candidate_seq) < 30000:
            x = random.randint(0, 2 ** self.dim - 1)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in candidate_seq:
                candidate_seq.append(initial_training_instance)
        begin = time.time()
        all_per = self.runtime_predict(model,candidate_seq)
        candidate_per = [all[1] for all in all_per]
        pos_seq = [0] * len(self.related)    
        now_best = max(candidate_per)
        now_best_seq = candidate_seq[candidate_per.index(now_best)]    
        now_best = self.get_objective_score(self.compiler,now_best_seq,100086, self.SOURCE_PATH,self.LOG_FILE, self.all_flags,self.EXEC_PARAM)[0]
        inital_indep_temp.append(now_best_seq)
        inital_dep_temp.append(now_best)
        model_new = RandomForestRegressor(random_state=self.seed)
        model_new.fit(np.array(inital_indep_temp), np.array(inital_dep_temp))
        before_time = time_tem[-1]
        time_tem.append(time.time() - begin + before_time)
        if self.global_best_per < now_best:
            self.global_best_per = now_best
            self.global_best_seq = now_best_seq
        ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
        #(ss, self.LOG_FILE)

        for idx in range(len(self.related)):
            new_candidate = []
            for j in range(len(candidate_seq)):
                seq = copy.deepcopy(candidate_seq[j])
                seq[self.related[idx]] = 1 - seq[self.related[idx]]
                new_candidate.append(seq)
            new_per = [all[1] for all in self.runtime_predict(model_new,new_candidate)]
            new_seq = [all[0] for all in self.runtime_predict(model_new,new_candidate)]
            new_best_seq = new_seq[new_per.index(max(new_per))]  
            new_best = self.get_objective_score(self.compiler,now_best_seq,100086, self.SOURCE_PATH,self.LOG_FILE, self.all_flags,self.EXEC_PARAM)[0]
            if new_best > self.global_best_per:
                self.global_best_per = new_best
                self.global_best_seq = new_best_seq

            for l in range(len(new_candidate)):
                if (candidate_per[l] > new_per[l] and new_candidate[l][self.related[idx]] == 1) or (candidate_per[l] < new_per[l] and new_candidate[l][self.related[idx]] == 0):
                    pos_seq[idx] -= 1
                else:
                    pos_seq[idx] += 1
            inital_indep_temp.append(new_best_seq)
            inital_dep_temp.append(new_best)
            model_new = RandomForestRegressor(random_state=self.seed)
            model_new.fit(np.array(inital_indep_temp), np.array(inital_dep_temp))
            time_tem.append(time.time() - begin + before_time)
            
            ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
            write_log(ss, self.LOG_FILE)

        sort_pos = sorted(enumerate(pos_seq), key=lambda x: x[1], reverse=True)
        critical_flag_idx = []
        for i in range(10):
            critical_flag_idx.append(self.related[sort_pos[i][0]])
        return critical_flag_idx, model_new
    
    def searchBycritical(self, critical_flag):
        """
        :param: critical_flag: idx of critical flag
        :return: the bias generation sequences
        """
        permutations = list(itertools.product([0, 1], repeat=10))
        seqs = []
        while len(seqs) < 1024 * 40:
            x = random.randint(0, 2 ** self.dim - 1)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in seqs:
                seqs.append(initial_training_instance)
        for i in range(len(permutations)):
            for idx in range(len(critical_flag)):
                for offset in range(0, 1024 * 40, 1024):
                    seqs[i + offset][critical_flag[idx]] = permutations[i][idx]
        return seqs
    
    def run(self,tuning_time):
        write_log("Build model",self.LOG_FILE)
        begin_all = time.time()
        """
        build model and get data set
        """
        best_compile_command = None
        model, inital_indep, inital_dep = self.build_RF_by_CompTuner()
        critical_flag, model_new = self.get_critical_flags(model, inital_indep, inital_dep)
        all_before = time_tem[-1]
        begin_all = time.time()
        it=0
        print(self.global_best_per)
        write_log("model complete!",self.LOG_FILE)
        while (time_tem[-1] < tuning_time):
            seq = self.searchBycritical(critical_flag)
            result = self.runtime_predict(model_new, seq)
            sorted_result = sorted(result, key=lambda x: x[1], reverse=True)  
            true_reslut , opt= self.get_objective_score(self.compiler,sorted_result[0][0],0, self.SOURCE_PATH,self.LOG_FILE, self.all_flags,self.EXEC_PARAM)
            if true_reslut > self.global_best_per:
                self.global_best_per = true_reslut
                self.global_best_seq = sorted_result[0][0]
                best_compile_command = opt
                write_log("Iteration {}: Best Performance: {}, Best Sequence: {}".format(it, self.global_best_per, self.global_best_seq), self.LOG_FILE)

            time_tem.append(time.time() - begin_all + all_before)
            #write_log("Iteration {}: Best Performance: {}, Best Sequence: {}".format(it, self.global_best_per, self.global_best_seq), self.LOG_FILE)
            # ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
            # write_log(ss, self.LOG_FILE)
            it+=1
        # best_result = self.get_objective_score(self.compiler,self.global_best_seq, 0,self.SOURCE_PATH,self.LOG_FILE, self.all_flags,self.EXEC_PARAM)[0]
        # ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(time_tem[-1])), str(best_result), str(self.global_best_seq))
        return self.global_best_per, self.global_best_seq, best_compile_command