from CompilerInterface import CompilerInterface
import os  # 用于获取文件大小
import subprocess

import shutil  # 用于检查命令是否存在


class GccCompiler(CompilerInterface):
    def __init__(self, version, include_path=None):
        super().__init__(version)
       
        # 动态获取 GCC 路径
        self.gcc_name="gcc"
        self.version=version
        self.gcc_path = self._find_gcc_path(version)
        if not self.gcc_path:
            raise ValueError(f"GCC version {version} not found on the system.")
        self.include_path = include_path  # Include path, defaults to None

    def get_name(self):
        return self.gcc_name
    
    def get_version(self):
        return self.version

    def get_path(self):
        return self.gcc_path

    def _find_gcc_path(self, version):
        """
        动态查找系统中指定版本的 GCC 路径。
        """
        gcc_command = f"gcc-{version}"  # 示例：gcc-11
        gcc_path = shutil.which(gcc_command)  # 检查命令是否存在并返回路径
        return gcc_path

    def get_search_space(self):
        """
        动态获取 GCC 支持的优化参数空间。
        使用 `gcc --help=optimizers` 获取支持的优化选项。
        """
        # 获取优化参数空间
        result = self._execute_gcc_help()
        if result["returncode"] != 0:
            raise ValueError(f"Failed to get optimizers: {result['error']}")

        # 打印获取的 GCC 命令输出
        #print("GCC Command Output:\n", result["output"])

        # 解析 gcc --help=optimizers 的输出
        search_space = self._parse_optimizers_help(result["output"])

        # 打印解析后的优化空间
        # print("Parsed Search Space:")
        # print(search_space)
    
        return search_space

    
    def _execute_gcc_help(self):
        """
        执行 `gcc --help=optimizers` 获取优化选项的输出
        """
        cmd = f"{self.gcc_path} --help=optimizers"
        return self._execute(cmd)
    
    
    def _parse_optimizers_help(self, output):
        """
        解析 `gcc --help=optimizers` 的输出，提取优化选项。
        """
        optimizers = {
            "opt_level": [0, 1, 2, 3],
            "flags": []
        }

        # 解析输出中的内容
        lines = output.splitlines()
        in_flags_section = False

        for line in lines:
            line = line.strip()

            # 开始优化标志部分
            if line.startswith("The following options control optimizations:"):
                in_flags_section = True
                continue  # Skip this line

            # 只处理 -f 开头的选项
            if in_flags_section and line.startswith("-f"):
                # 提取 -f 开头的选项并去掉后面的解释部分
                flag = line.split(" ")[0]

                # 如果标志包含等号（=），则跳过
                if "=" not in flag:
                    optimizers["flags"].append(flag)

        return optimizers
    # def _parse_optimizers_help(self, output):
    #     optimizers = {
    #         "opt_level": [0, 1, 2, 3],
    #         "flags": []
    #     }

    #     for line in output.splitlines():
    #         line = line.strip()
    #         if line.startswith("-f"):
    #             flag = line.split(" ")[0]  # 只取第一个空格前的部分
    #             optimizers["flags"].append(flag)  # 不过滤 `=`

    #     return optimizers


    def compile(self, command):
        
        result = self._execute(command)
        return result["returncode"] == 0
    
    
    # def execute(self, exec_param=None):
    #     cmd = f"./a.out {exec_param if exec_param else '1125000'}"
    #     return self._execute(cmd)
    def execute(self, exec_param=""):
        """执行编译后的程序"""
        cmd = f"./a.out {exec_param}".strip()
        return self._execute(cmd)

    
    
    # def get_compile_command(self, opt_flags, source_file):
    #     """
    #     生成编译命令。
    #     这里将包含优化级别和其它标志，生成完整的编译命令行。
    #     """
    #     #compile_cmd = f"{self.gcc_path} {source_file} -o a.out {opt_flags} -lm"
    #     compile_cmd= f"{self.gcc_path} -O2 {opt_flags} -c  {source_file}/*.c"

    #     # 如果提供了 include_path，则将其添加到编译命令中
    #     if self.include_path:
    #         compile_cmd = f"{self.gcc_path} -I{self.include_path} {source_file} -o a.out {opt_flags} -lm"

    #     return compile_cmd
    def get_compile_command(self, opt_flags, source_file, opt_level=None):
        """
        生成编译命令。可指定优化级别 opt_level（例如 "-O2" 或 "-O3"），
        并将 include_path（如果有）加入命令中。
        """
        level_option = opt_level if opt_level else ""
        # 若设置了 include_path，则添加 -I 路径
        if self.include_path:
            #compile_cmd = f"{self.gcc_path} {level_option} -I{self.include_path} {source_file}/*.c -o a.out {opt_flags} -lm"
            # Step 1: 生成 .o 目标文件（编译）
            command1 = f"{self.gcc_path} {level_option} {opt_flags} -c -I{self.include_path} {source_file}/*.c"

            # Step 2: 链接所有 .o 文件生成 a.out
            command2 = f"{self.gcc_path} -o a.out {level_option} {opt_flags} -lm *.o"
        else:
            #compile_cmd = f"{self.gcc_path} {level_option} {source_file}/*.c -o a.out {opt_flags} -lm"
            # Step 1: 生成 .o 目标文件（编译）
            command1 = f"{self.gcc_path} {level_option} {opt_flags} -c {source_file}/*.c"

            # Step 2: 链接所有 .o 文件生成 a.out
            command2 = f"{self.gcc_path} -o a.out {level_option} {opt_flags} -lm *.o"
        return command1,command2
    
    def _execute(self, cmd):
        try:
            completed = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if completed.returncode != 0:
                return {"returncode": completed.returncode, "error": completed.stderr.strip()}
            return {"returncode": completed.returncode, "output": completed.stdout.strip()}
        except Exception as e:
            print(f"Exception occurred while executing command: {cmd}")
            print(str(e))
            return {"returncode": 1, "error": str(e)}
        

   


