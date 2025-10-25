# 配置手册

## 统一命名配置gcc环境

以gcc-12.1.0版本为例

```bash
https://ftp.gnu.org/gnu/gcc/gcc-12.1.0/gcc-12.1.0.tar.gz
sudo apt install build-essential libgmp-dev libmpfr-dev libmpc-dev
sudo apt install flex bison
tar -xvzf gcc-12.1.0.tar.gz
mkdir gcc-12.1.0-build
cd gcc-12.1.0-build
export PATH=/opt/gcc-12.1.0/bin:$PATH
source ~/.bashrc
sudo ln -s /opt/gcc-12.1.0/bin/gcc /usr/bin/gcc-12.1.0
gcc-12.1.0 --version
```


## 统一命名配置llvm环境

以clang-12.0.1版本为例

```bash
sudo apt install clang-12 llvm-12
sudo ln -s /opt/llvm-12.0.1/bin/clang /usr/bin/clang-12.0.1
```


## 代码克隆本地

```bash
git clone git@github.com:Qtllll/Lab.git
cd Lab/lab #进入代码目录
```

## 安装项目依赖库

```
pip install scikit-learn
```

## 运行参数

本编译器自动调优平台通过命令行接口实现功能调用，用户需通过指定参数配置优化任务。程序入口为main.py，基础命令格式为：

```bash
python3 main.py --compiler <编译器版本> --algorithm <优化算法> --program <程序路径>
```

`--compiler`:用于指定编译器类型及版本（如GCC-12.1.0或Clang-12.0.1）

`--algorithm`:选择优化算法（可选RIO、CompTuner、BOCATuner、GATuner、CFSCATuner、APCOTuner）

`--program`:需填写待优化的C/C++源代码文件路径

附加参数配置：

`--log_file`:可自定义日志输出路径（默认生成optimization.log）

`--include_path`:用于指定外部头文件目录

`--tuning_time`:可设定优化任务最大持续时间（单位秒，默认5000秒）

`--flags_file`:允许用户加载预定义的优化选项文件（如flags.txt），以限定优化参数搜索空间。

**基础优化任务运行测试用例：** 可以直接指定编译器、算法与目标程序即可启动。

例如对example.c程序使用GCC-12.1.0编译器和RIO算法时：

```bash
python3 main.py --compiler gcc-12.1.0 --algorithm RIO --program example.c
```



**复杂编译环境运行测试用例：** 若程序依赖第三方头文件库，可通过`--include_path`添加包含路径：

```bash 
python3 main.py \
  --compiler gcc-12.1.0 \
  --algorithm CFSCATuner \
  --program /home/qtl/Lab/lab/polybench-code/datamining/correlation \
  --include_path "/home/qtl/Lab/lab/polybench-code/utilities /home/qtl/Lab/lab/polybench-code/utilities/polybench.c"

```

**自定义优化范围运行测试用例：** 需限定特定优化选项时，可预先在`flags.txt`中定义候选参数集合，通过`--flags_file`加载以缩小搜索范围：

```bash
python3 main.py \
  --compiler gcc-12.1.0 \
  --algorithm CFSCATuner \
  --program /home/qtl/test/polybench-code/datamining/correlation \
  --log_file correlation_cfsca.log \
  --include_path "/home/qtl/Lab/lab/polybench-code/utilities /home/qtl/Lab/lab/polybench-code/utilities/polybench.c" \
  --tuning_time 50 \
  --flags_file /home/qtl/test/flag.txt
```

**查询数据库数据：**系统将自动记录优化结果至SQLite数据库（默认文件名为`optimization_results.db`），存储字段涵盖编译器版本、目标程序名称、最优编译参数集合、程序实际运行时间及相对于-O3基准的加速比提升百分比。

例如，当用户优化test.c程序后，可通过终端执行命令 ,直接查询该程序的历史优化记录: 

```bash
sqlite3 optimization_results.db “SELECT * FROM optimization_results WHERE program_path=’test.c’;”
```

