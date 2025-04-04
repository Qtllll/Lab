from CompilerInterface import CompilerInterface
import subprocess
import shutil

class LlvmCompiler(CompilerInterface):
    def __init__(self, version, include_path=None):
        super().__init__(version)
        self.llvm_name = "clang"
        self.version = version
        self.llvm_path = self._find_llvm_path(version)
        if not self.llvm_path:
            raise ValueError(f"LLVM version {version} not found on the system.")
        self.include_path = include_path  # Include path, defaults to None

    def get_name(self):
        return self.llvm_name
    
    def get_version(self):
        return self.version

    def get_path(self):
        return self.llvm_path

    def _find_llvm_path(self, version):
        """动态查找系统中指定版本的 LLVM 路径"""
        llvm_command = f"clang-{version}"
        llvm_path = shutil.which(llvm_command)  # 查找命令路径
        return llvm_path

    def get_search_space(self):
        """动态获取 LLVM 支持的优化参数"""
        result = self._execute_llvm_help()
        if result["returncode"] != 0:
            raise ValueError(f"Failed to get optimizers: {result['error']}")

        # 解析 clang --help 输出的优化选项
        search_space = self._parse_optimizers_help(result["output"])
        return search_space

    def _execute_llvm_help(self):
        """执行 `clang --help` 获取帮助选项"""
        cmd = f"{self.llvm_path} --help"
        return self._execute(cmd)

    
    
    # def _parse_optimizers_help(self, output):
    #   """解析 `clang --help` 的输出，提取优化选项"""
    #   optimizers = {
    #       "opt_level": [0, 1, 2, 3],  # 默认优化级别
    #       "flags": []
    #   }

    #   lines = output.splitlines()
    #   in_flags_section = False

    #   for line in lines:
    #       line = line.strip()

    #       # 查找优化标志部分
    #       if line.startswith("The following options control optimizations:"):
    #           in_flags_section = True
    #           continue

    #       # # 查找 -O 开头的优化标志
    #       # if line.startswith("-O"):
    #       #     optimizers["opt_level"].append(line.split()[0])

    #       # 查找以 -f 开头的优化标志
    #       elif line.startswith("-f"):
    #         flag = line.split(" ")[0]  # 提取标志并去掉后面的描述部分
    #         if "=" not in flag:  # 不需要带有参数的标志
    #             optimizers["flags"].append(flag)

    #   return optimizers
    def _parse_optimizers_help(self, output):
        """解析 `clang --help` 的输出，提取优化选项"""
        optimizers = {
            "opt_level": ["-O0", "-O1", "-O2", "-O3", "-Ofast"],  # 预定义优化级别
            "flags": []
        }

        lines = output.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("-f") or line.startswith("-m"):  # 解析 -f 和 -m 选项
                flag = line.split(" ")[0]
                if "=" not in flag:  # 过滤掉带参数的选项
                    optimizers["flags"].append(flag)

        return optimizers

    def compile(self, command):
        """编译源文件并返回是否成功及错误信息"""
        result = self._execute(command)
        return result["returncode"] == 0, result.get("error", "")


    # def compile(self, command):
    #     """编译源文件"""
    #     result = self._execute(command)
    #     return result["returncode"] == 0

    # def execute(self, exec_param=None):
    #     """执行编译后的程序"""
    #     cmd = f"./a.out {exec_param if exec_param else '1125000'}"
    #     return self._execute(cmd)
    def execute(self, exec_param=""):
        """执行编译后的程序"""
        cmd = f"./a.out {exec_param}".strip()
        return self._execute(cmd)


    # def get_compile_command(self, opt_flags, source_file):
    #     """生成编译命令"""
    #     compile_cmd = f"{self.llvm_path} {source_file} -o a.out {opt_flags} -lm"
    #     if self.include_path:
    #         compile_cmd = f"{self.llvm_path} -I{self.include_path} {source_file} -o a.out {opt_flags} -lm"
    #     return compile_cmd
    
    # def get_compile_command(self, opt_flags, source_file, opt_level=None):
    #     """
    #     生成编译命令。可指定优化级别 opt_level（例如 "-O2" 或 "-O3"），
    #     并将 include_path（如果有）加入命令中。
    #     """
    #     level_option = opt_level if opt_level else ""
    #     # 若设置了 include_path，则添加 -I 路径
    #     if self.include_path:
    #         compile_cmd = f"{self.llvm_path} {level_option} -I{self.include_path} {source_file}/*.c -o a.out {opt_flags} -lm"
    #     else:
    #         compile_cmd = f"{self.llvm_path} {level_option} {source_file}/*.c -o a.out {opt_flags} -lm"
    #     return compile_cmd
    def get_compile_command(self, opt_flags, source_file, opt_level=None):
        """
        生成编译命令。可指定优化级别 opt_level（例如 "-O2" 或 "-O3"），
        并将 include_path（如果有）加入命令中。
        """
        level_option = opt_level if opt_level else ""
        include_option = f"-I{self.include_path}" if self.include_path else ""
    
        # 如果 source_file 是目录，假设目录下有多个 C 文件
        # if source_file.endswith("/"):
        #     source_file = f"{source_file}*.c"

        # compile_cmd = f"{self.llvm_path} {level_option} {include_option} {source_file} {opt_flags} -o a.out -lm"
        # Step 1: 生成 .o 目标文件（编译）
        command1 = f"{self.llvm_path} {level_option} {opt_flags} {include_option} -c {source_file}/*.c"

        # Step 2: 链接所有 .o 文件生成 a.out
        command2 = f"{self.llvm_path} -o a.out {level_option} {opt_flags} -lm *.o"

        return command1, command2
        #return compile_cmd


    def _execute(self, cmd):
        """执行命令并捕获输出"""
        try:
            completed = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if completed.returncode != 0:
                return {"returncode": completed.returncode, "error": completed.stderr.strip()}
            return {"returncode": completed.returncode, "output": completed.stdout.strip()}
        except Exception as e:
            print(f"Exception occurred while executing command: {cmd}")
            print(str(e))
            return {"returncode": 1, "error": str(e)}
