from abc import ABC, abstractmethod

class AlgorithmInterface(ABC):
    @abstractmethod
    def optimize(self, compiler, source_file, log_file):
        """执行优化过程"""
        pass


    @abstractmethod
    def log_results(self, log_file, message):
        """记录优化过程中的日志"""
        pass
