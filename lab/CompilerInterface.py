from abc import ABC, abstractmethod

class CompilerInterface(ABC):
    def __init__(self, version):
        self.version = version

    @abstractmethod
    def get_search_space(self):
        """返回该版本编译器的可用优化参数"""
        pass
