from Compiler.llvm_compiler import LlvmCompiler  # 确保这行导入没有问题

from Compiler.gcc_compiler import GccCompiler



class CompilerFactory:
    @staticmethod
    def create_compiler(compiler_name, version,include_path=None):
        if compiler_name == "gcc":
            return GccCompiler(version,include_path)
        elif compiler_name == "clang":
            return LlvmCompiler(version, include_path)  # 返回 LlvmCompiler 实例
        else:
            raise ValueError(f"Unsupported compiler: {compiler_name}")
        

