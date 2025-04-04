from Algorithm.RandomIterativeOptimization import RandomIterativeOptimization
from Algorithm.CompTunerAlgorithm import CompTunerAlgorithm
from Algorithm.BOCATunerAlgorithm import BOCAAlgorithm
from Algorithm.GATunerAlgorithm import GATunerAlgorithm
from Algorithm.CFSCATunerAlgorithm import CFSCAAlgorithm
from Algorithm.APCOTunerAlgorithm import APCOTunerAlgorithm
class AlgorithmFactory:
    @staticmethod
    def create_algorithm(algorithm_name,compiler, source_path, flags, log_file):
        if algorithm_name == "RIO":
            return RandomIterativeOptimization(compiler, source_path, flags)
        elif algorithm_name == "CompTuner":
            return CompTunerAlgorithm(compiler, source_path, flags, log_file)
        elif algorithm_name == "BOCATuner":
            return BOCAAlgorithm(compiler, source_path, flags, log_file)
        elif algorithm_name == "GATuner":
            return GATunerAlgorithm(compiler, source_path, flags, log_file)
        elif algorithm_name == "CFSCATuner":
            return CFSCAAlgorithm(compiler, source_path, flags, log_file)
        elif algorithm_name == "APCOTuner":
            return APCOTunerAlgorithm(compiler, source_path, flags, log_file)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")
