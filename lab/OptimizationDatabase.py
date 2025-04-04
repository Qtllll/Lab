import sqlite3
from datetime import datetime

def weighted_version_distance(version1, version2):
    """
    计算两个版本号之间的加权距离
    :param version1: 第一个版本号（字符串格式）
    :param version2: 第二个版本号（字符串格式）
    :return: 两个版本号之间的加权距离（整数值）
    """
    # 将版本号分割成各个部分，并将它们转换成整数列表
    v1_parts = [int(i) for i in version1.split('.')]
    v2_parts = [int(i) for i in version2.split('.')]
    
    # 对齐版本号长度，短的版本号补充 0
    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts.extend([0] * (max_len - len(v1_parts)))  # 补充 0
    v2_parts.extend([0] * (max_len - len(v2_parts)))  # 补充 0
    
    # 设定不同部分的权重
    weights = [100, 10, 1]  # 主版本：100，次版本：10，修订版：1
    distance = 0
    
    # 计算加权差异
    for i in range(len(v1_parts)):
        distance += abs(v1_parts[i] - v2_parts[i]) * weights[i]
    
    return distance

class OptimizationDatabase:
    def __init__(self, db_file="optimization_results.db"):
        """初始化数据库连接和表格创建"""
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """创建数据库表格"""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            compiler_name TEXT NOT NULL,
            compiler_version TEXT NOT NULL,
            algorithm_name TEXT NOT NULL,
            program_path TEXT NOT NULL,
            optimization_flags TEXT NOT NULL,
            execution_time REAL NOT NULL
        );
        """)
        self.conn.commit()

    def query_database(self, compiler_name, compiler_version,algorithm_name, program_path):
        """
        查询数据库，根据编译器名称、版本和程序路径获取历史优化记录。
        :param program_path: 程序路径
        :param compiler_version: 编译器版本
        :return: 满足条件的历史记录（字典形式），没有结果时返回 None
        """
        query = """
        SELECT optimization_flags, execution_time 
        FROM optimization_results 
        WHERE compiler_name = ? 
        AND compiler_version = ? 
        AND algorithm_name = ?
        AND program_path = ?
        """
        
        self.cursor.execute(query, (compiler_name, compiler_version, algorithm_name, program_path))
        result = self.cursor.fetchone()
        
        if result:
            # 返回查询结果（flags 和执行时间）
            return {'flags': result[0], 'execution_time': result[1]}
        else:
            # 没有找到匹配的记录
            return None
        
    def query_closest_data(self, compiler_name, program_path, compiler_version, algorithm_name):
        """
        查询数据库中与给定编译器名称、程序路径最接近的历史记录（按版本号接近）
        :param compiler_name: 编译器名称
        :param program_path: 程序路径
        :param compiler_version: 给定的编译器版本
        :return: 最接近的历史记录（字典形式），没有结果时返回 None
        """
        query = """
        SELECT compiler_version, optimization_flags, execution_time 
        FROM optimization_results 
        WHERE compiler_name = ? 
        AND program_path = ?
        AND algorithm_name = ?
        """
    
        self.cursor.execute(query, (compiler_name, program_path, algorithm_name))
        results = self.cursor.fetchall()

        closest_data = None
        closest_version_distance = float('inf')  # 存储最小版本差距
    
        for result in results:
            db_version = result[0]
            if db_version:  # 确保版本数据存在
                try:
                    # 计算版本之间的加权距离
                    dist = weighted_version_distance(compiler_version, db_version)
                    if dist < closest_version_distance:
                        closest_version_distance = dist
                        closest_data = {'compiler_version': db_version, 'flags': result[1], 'execution_time': result[2]}
                except Exception as e:
                    print(f"Error comparing versions: {e}")
    
        return closest_data


    def log_result(self, compiler_name, compiler_version, algorithm_name, program_path, flags, execution_time):
        """记录每次优化的结果"""
        self.cursor.execute("""
        INSERT INTO optimization_results (compiler_name, compiler_version, algorithm_name, program_path, optimization_flags, execution_time)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (compiler_name, compiler_version, algorithm_name, program_path, str(flags), execution_time))
        self.conn.commit()

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


