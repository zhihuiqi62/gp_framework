import numpy as np
import operator
import random
import time
from typing import List, Dict, Any, Callable, Optional

# ================= 基础函数集（增强版） =================
def add(x, y): 
    """安全加法（自动广播）"""
    return np.add(x, y)

def sub(x, y): 
    """安全减法（自动广播）"""
    return np.subtract(x, y)

def mul(x, y): 
    """安全乘法（自动广播）"""
    return np.multiply(x, y)

def div(x, y): 
    """安全除法（处理除零和广播）"""
    y_safe = np.where(np.abs(y) < 1e-10, 1e-10, y)  # 严格处理接近零的值
    return np.divide(x, y_safe)

def abs_(x): 
    """绝对值"""
    return np.abs(x)

def sqrt(x): 
    """安全平方根（处理负数）"""
    return np.sqrt(np.abs(x))

def log(x): 
    """安全对数（处理零和负数）"""
    return np.log(np.abs(x) + 1e-8)  # 加偏移避免零

def inv(x): 
    """安全倒数（处理接近零的值）"""
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)  # 严格阈值
    return 1.0 / x_safe

def rank(x, d=10):
    """
    滚动窗口排名归一化（防止未来函数泄漏）
    d: 窗口长度，默认10。不满d期填nan
    """
    # 如果输入是标量，直接返回
    if np.isscalar(x):
        return x
    # 转换输入为numpy数组
    x = np.asarray(x)
    # 如果输入为空数组，直接返回
    if x.size == 0:
        return x
    # 将窗口长度d四舍五入为整数
    d = int(round(d))
    # 确保窗口长度至少为1
    d = max(1, d)
    # 如果是一维数组
    if x.ndim == 1:
        # 创建与x同形状的全nan数组用于存放结果
        res = np.full_like(x, np.nan, dtype=np.float64)
        # 从第d-1个元素开始，遍历每个位置
        for i in range(d-1, len(x)):
            # 取当前窗口的d个元素
            window = x[i-d+1:i+1]
            # 对窗口内元素进行排名归一化
            ranks = window.argsort().argsort() / (len(window) - 1 + 1e-8)
            # 将当前时刻的排名结果赋值给res
            res[i] = ranks[-1]
        # 返回结果
        return res
    else:
        # 如果是二维数组，创建同形状的全nan数组
        res = np.full_like(x, np.nan, dtype=np.float64)
        # 遍历每一列
        for j in range(x.shape[1]):
            # 取第j列
            col = x[:, j]
            # 从第d-1个元素开始，遍历每个位置
            for i in range(d-1, len(col)):
                # 取当前窗口的d个元素
                window = col[i-d+1:i+1]
                # 对窗口内元素进行排名归一化
                ranks = window.argsort().argsort() / (len(window) - 1 + 1e-8)
                # 将当前时刻的排名结果赋值给res
                res[i, j] = ranks[-1]
        # 返回结果
        return res

def delay(x, d):
    """延迟函数（确保d为正整数，处理边界）"""
    d = int(round(d))  # 确保整数
    d = max(1, d)  # 延迟至少1期
    if np.isscalar(x):
        return x
    x = np.asarray(x)
    # 滚动后填充NaN（原代码用0填充可能不合理）
    rolled = np.roll(x, d, axis=0) if x.ndim >= 1 else x
    # 前d期填充NaN（更合理的延迟处理）
    if x.ndim == 1:
        rolled[:d] = np.nan
    else:
        rolled[:d, :] = np.nan
    return rolled

def correlation(x, y, d):
    """滚动窗口相关系数（防止未来函数泄漏）"""
    d = int(round(d))
    d = max(1, min(d, 100))
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0 or np.isscalar(x) or np.isscalar(y):
        return np.nan
    
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            # 取当前窗口的d个元素
            x_window = x[i-d+1:i+1]
            y_window = y[i-d+1:i+1]
            if len(x_window) >= 2:
                try:
                    corr = np.corrcoef(x_window, y_window)[0, 1]
                    res[i] = corr if not np.isnan(corr) else np.nan
                except:
                    res[i] = np.nan
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                x_window = x[i-d+1:i+1, j]
                y_window = y[i-d+1:i+1, j]
                if len(x_window) >= 2:
                    try:
                        corr = np.corrcoef(x_window, y_window)[0, 1]
                        res[i, j] = corr if not np.isnan(corr) else np.nan
                    except:
                        res[i, j] = np.nan
        return res

def covariance(x, y, d):
    """滚动窗口协方差（防止未来函数泄漏）"""
    d = int(round(d))
    d = max(1, min(d, 100))
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0 or np.isscalar(x) or np.isscalar(y):
        return np.nan
    
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            # 取当前窗口的d个元素
            x_window = x[i-d+1:i+1]
            y_window = y[i-d+1:i+1]
            if len(x_window) >= 2:
                try:
                    cov = np.cov(x_window, y_window)[0, 1]
                    res[i] = cov if not np.isnan(cov) else np.nan
                except:
                    res[i] = np.nan
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                x_window = x[i-d+1:i+1, j]
                y_window = y[i-d+1:i+1, j]
                if len(x_window) >= 2:
                    try:
                        cov = np.cov(x_window, y_window)[0, 1]
                        res[i, j] = cov if not np.isnan(cov) else np.nan
                    except:
                        res[i, j] = np.nan
        return res

def scale(x, d=10): 
    """滚动窗口缩放至绝对值和为1（防止未来函数泄漏）"""
    if np.isscalar(x):
        return x
    x = np.asarray(x)
    if x.size == 0:
        return x
    d = int(round(d))
    d = max(1, d)
    
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            # 取当前窗口的d个元素
            window = x[i-d+1:i+1]
            # 计算窗口内绝对值和
            s = np.sum(np.abs(window))
            s = s if s > 1e-10 else 1e-10  # 避免除零
            # 缩放当前值
            res[i] = x[i] / s
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            col = x[:, j]
            for i in range(d-1, len(col)):
                # 取当前窗口的d个元素
                window = col[i-d+1:i+1]
                # 计算窗口内绝对值和
                s = np.sum(np.abs(window))
                s = s if s > 1e-10 else 1e-10  # 避免除零
                # 缩放当前值
                res[i, j] = col[i] / s
        return res

def delta(x, d):
    """当前值与延迟d期的差值（复用delay函数）"""
    return sub(x, delay(x, d))

def signedpower(x): 
    """带符号二次幂运算 sign(x) * (abs(x) ** 2)"""
    return np.sign(x) * (np.abs(x) ** 2)

def decay_linear(x, d):
    """滚动窗口线性加权移动平均（防止未来函数泄漏）"""
    d = int(round(d))
    d = max(1, min(d, 100))
    if np.isscalar(x):
        return x
    x = np.asarray(x)
    w = np.arange(1, d + 1, dtype=np.float64)
    w /= w.sum()  # 归一化权重
    
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            res[i] = np.dot(window, w)
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                window = x[i-d+1:i+1, j]
                res[i, j] = np.dot(window, w)
        return res

# 时间序列函数（滚动窗口处理，防止未来函数泄漏）
def ts_min(x, d): 
    """滚动窗口最小值"""
    d = int(round(d))
    d = max(1, min(d, 100))
    x = np.asarray(x)
    if np.isscalar(x):
        return x
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            res[i] = np.min(window)
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                window = x[i-d+1:i+1, j]
                res[i, j] = np.min(window)
        return res

def ts_max(x, d):
    """滚动窗口最大值"""
    d = int(round(d))
    d = max(1, min(d, 100))
    x = np.asarray(x)
    if np.isscalar(x):
        return x
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            res[i] = np.max(window)
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                window = x[i-d+1:i+1, j]
                res[i, j] = np.max(window)
        return res

def ts_argmin(x, d):
    """滚动窗口最小值位置"""
    d = int(round(d))
    d = max(1, min(d, 100))
    x = np.asarray(x)
    if np.isscalar(x):
        return 0
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            res[i] = np.argmin(window)
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                window = x[i-d+1:i+1, j]
                res[i, j] = np.argmin(window)
        return res

def ts_argmax(x, d):
    """滚动窗口最大值位置"""
    d = int(round(d))
    d = max(1, min(d, 100))
    x = np.asarray(x)
    if np.isscalar(x):
        return 0
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            res[i] = np.argmax(window)
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                window = x[i-d+1:i+1, j]
                res[i, j] = np.argmax(window)
        return res

def ts_rank(x, d):
    """滚动窗口排名"""
    d = int(round(d))
    d = max(1, min(d, 100))
    x = np.asarray(x)
    if np.isscalar(x):
        return x
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            # 计算当前值在窗口中的排名
            current_value = x[i]
            rank_position = np.sum(window < current_value)
            normalized_rank = rank_position / (len(window) + 1e-8)
            res[i] = normalized_rank
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                window = x[i-d+1:i+1, j]
                current_value = x[i, j]
                rank_position = np.sum(window < current_value)
                normalized_rank = rank_position / (len(window) + 1e-8)
                res[i, j] = normalized_rank
        return res

def ts_sum(x, d):
    """滚动窗口求和"""
    d = int(round(d))
    d = max(1, min(d, 100))
    x = np.asarray(x)
    if np.isscalar(x):
        return x
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            res[i] = np.sum(window)
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                window = x[i-d+1:i+1, j]
                res[i, j] = np.sum(window)
        return res

def ts_product(x, d):
    """滚动窗口乘积"""
    d = int(round(d))
    d = max(1, min(d, 100))
    x = np.asarray(x)
    if np.isscalar(x):
        return x
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            res[i] = np.prod(window)
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                window = x[i-d+1:i+1, j]
                res[i, j] = np.prod(window)
        return res

def ts_stddev(x, d):
    """滚动窗口标准差"""
    d = int(round(d))
    d = max(1, min(d, 100))
    x = np.asarray(x)
    if np.isscalar(x):
        return 0.0
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            res[i] = np.std(window, ddof=1)
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                window = x[i-d+1:i+1, j]
                res[i, j] = np.std(window, ddof=1)
        return res

def ts_corr(x, y, d):
    return correlation(x, y, d)  # 复用优化后的correlation

def ts_cov(x, y, d):
    return covariance(x, y, d)  # 复用优化后的covariance

ts_prod = ts_product  # 保持一致性

def ts_zscore(x, d):
    """滚动窗口Z-score"""
    d = int(round(d))
    d = max(1, min(d, 100))
    x = np.asarray(x)
    if np.isscalar(x):
        return 0.0
    if x.ndim == 1:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for i in range(d-1, len(x)):
            window = x[i-d+1:i+1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            res[i] = mean / (std + 1e-8)
        return res
    else:
        res = np.full_like(x, np.nan, dtype=np.float64)
        for j in range(x.shape[1]):
            for i in range(d-1, x.shape[0]):
                window = x[i-d+1:i+1, j]
                mean = np.mean(window)
                std = np.std(window, ddof=1)
                res[i, j] = mean / (std + 1e-8)
        return res

# 排名操作函数（增强鲁棒性）
def rank_sub(x, y, d=10):
    """滚动窗口排名差值（防止未来函数泄漏）"""
    rx = rank(x, d)
    ry = rank(y, d)
    try:
        return rx - ry
    except:
        # 形状不匹配时返回0数组
        return np.zeros_like(rx) if hasattr(rx, 'size') and rx.size > 0 else 0

def rank_div(x, y, d=10):
    """滚动窗口排名比值（防止未来函数泄漏）"""
    rx = rank(x, d)
    ry = rank(y, d)
    try:
        ry_safe = np.where(ry < 1e-10, 1e-10, ry)  # 避免除零
        return rx / ry_safe
    except:
        return np.ones_like(rx) if hasattr(rx, 'size') and rx.size > 0 else 1

def sigmoid(x):
    """安全sigmoid（避免溢出）"""
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))  # 截断极端值


class FunctionSet:
    """函数集管理（支持动态增删）"""
    def __init__(self):
        self.functions = {
            'add': (add, 2),
            'sub': (sub, 2),
            'mul': (mul, 2),
            'div': (div, 2),
            'abs': (abs_, 1),
            'sqrt': (sqrt, 1),
            'log': (log, 1),
            'inv': (inv, 1),
            'rank': (rank, 2),
            'delay': (delay, 2),
            'correlation': (correlation, 3),
            'covariance': (covariance, 3),
            'scale': (scale, 2),
            'delta': (delta, 2),
            'signedpower': (signedpower, 1),
            'decay_linear': (decay_linear, 2),
            'ts_min': (ts_min, 2),
            'ts_max': (ts_max, 2),
            'ts_argmin': (ts_argmin, 2),
            'ts_argmax': (ts_argmax, 2),
            'ts_rank': (ts_rank, 2),
            'ts_sum': (ts_sum, 2),
            'ts_product': (ts_product, 2),
            'ts_stddev': (ts_stddev, 2),
            'ts_corr': (ts_corr, 3),
            'ts_cov': (ts_cov, 3),
            'ts_prod': (ts_prod, 2),
            'ts_zscore': (ts_zscore, 2),
            'rank_sub': (rank_sub, 3),
            'rank_div': (rank_div, 3),
            'sigmoid': (sigmoid, 1),
        }
    
    def add_function(self, name: str, func: Callable, arity: int):
        self.functions[name] = (func, arity)
    
    def get(self, name: str):
        return self.functions[name]
    
    def all(self):
        return self.functions.copy()
    
    def remove_function(self, name: str):
        if name in self.functions:
            del self.functions[name]


class Node:
    """表达式树节点（增强评估稳定性）"""
    def __init__(self, name: str, children: List['Node'] = None, value: Any = None):
        self.name = name
        self.children = children or []
        self.value = value  # 变量名（str）或常数（number）

    def evaluate(self, X, function_set):
        if self.value is not None:
            if isinstance(self.value, str):  # variable
                if isinstance(X, dict):
                    arr = X.get(self.value, 0)
                    return arr
                elif hasattr(X, '__getitem__'):
                    try:
                        return X[self.value]
                    except Exception:
                        return 0
                else:
                    return 0
            else:  # constant
                if isinstance(X, dict) and len(X) > 0:
                    arr = next(iter(X.values()))
                    return np.full_like(arr, self.value, dtype=np.float64)
                return self.value
        func, arity = function_set.get(self.name)
        args = [child.evaluate(X, function_set) for child in self.children]
        try:
            result = func(*args)
            # 兜底：如果返回None或不是np.ndarray/float，返回全nan
            if result is None or (isinstance(result, float) and np.isnan(result)):
                if isinstance(X, dict) and len(X) > 0:
                    arr = next(iter(X.values()))
                    return np.full_like(arr, np.nan, dtype=np.float64)
                return np.nan
            # 如果是标量，直接返回
            if np.isscalar(result):
                return result
            # 如果是0维数组，转为float
            if isinstance(result, np.ndarray) and result.shape == ():
                return float(result)
            # 如果是object类型，转为float64
            if isinstance(result, np.ndarray) and result.dtype == object:
                try:
                    return result.astype(np.float64)
                except Exception:
                    arr = next(iter(X.values())) if isinstance(X, dict) and len(X) > 0 else 0
                    return np.full_like(arr, np.nan, dtype=np.float64)
            return result
        except Exception:
            if isinstance(X, dict) and len(X) > 0:
                arr = next(iter(X.values()))
                return np.full_like(arr, np.nan, dtype=np.float64)
            return np.nan
    
    def to_str(self) -> str:
        """转为字符串表达式"""
        if self.value is not None:
            return str(self.value)
        return f"{self.name}({', '.join([c.to_str() for c in self.children])})"

    def depth(self) -> int:
        """计算节点深度（用于控制树复杂度）"""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def size(self) -> int:
        """计算节点数量（用于控制树大小）"""
        return 1 + sum(child.size() for child in self.children)


class GeneticProgrammer:
    """遗传规划主类（优化参数和搜索策略）"""
    def __init__(
        self,
        generations: int = 30,  # 增加进化代数
        population_size: int = 100,  # 增加种群规模
        n_components: int = 5,
        hall_of_fame: int = 10,  # 精英保留数量
        function_set: FunctionSet = None,
        parsimony_coefficient: float = 0.001,  # 简约系数（惩罚复杂树）
        tournament_size: int = 7,  # 锦标赛规模
        random_state: int = 42,
        init_depth: tuple = (2, 6),  # 初始树深度范围（更深）
        const_range: Optional[tuple] = (-2, 2),  # 常数范围扩大
        ts_window: int = 30,  # 时间窗口范围扩大
        # 遗传操作概率（增强多样性）
        p_crossover: float = 0.6,
        p_subtree_mutation: float = 0.25,
        p_hoist_mutation: float = 0.05,
        p_point_mutation: float = 0.08,
        p_point_replace: float = 0.3,  # 点突变概率提高
        variable_names: List[str] = None
    ):
        self.generations = generations
        self.population_size = population_size
        self.n_components = n_components
        self.hall_of_fame = hall_of_fame
        self.function_set = function_set or FunctionSet()
        self.parsimony_coefficient = parsimony_coefficient
        self.tournament_size = tournament_size
        self.random_state = np.random.RandomState(random_state)
        self.init_depth = init_depth
        self.const_range = const_range
        self.ts_window = ts_window
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.variable_names = variable_names or ['X']
        self.best_programs_ = []  # 最优程序存储
    
    def fit(
        self,
        fitness_func: Callable,
        fitness_args: tuple = (),
        fitness_kwargs: dict = None,
        progress_callback: Callable = None
    ):
        fitness_kwargs = fitness_kwargs or {}
        
        # 如果没有传入进度回调，使用默认的进度显示
        if progress_callback is None:
            start_time = time.time()
            
            def default_progress_callback(gen, avg_len, avg_fit, best_len, best_fit):
                """默认进度显示函数"""
                if gen == 0:
                    print('开始遗传规划因子挖掘...')
                    print('=' * 90)
                    print('|    Population    |           Best Individual            |       Factor Quality       |')
                    print('-' * 90)
                    print(' Gen   AvgLen    AvgFit    BestLen    BestFit     Time Left')
                    print('-' * 90)
                
                # 处理-inf显示
                avg_fit_str = f"{avg_fit:7.4f}" if avg_fit != -np.inf else "   -inf"
                best_fit_str = f"{best_fit:7.4f}" if best_fit != -np.inf else "   -inf"
                
                # 计算剩余时间
                elapsed = time.time() - start_time
                if gen > 0:
                    time_per_gen = elapsed / (gen + 1)
                    time_left = (self.generations - gen - 1) * time_per_gen
                    time_str = f"{time_left/60:.2f}m" if time_left > 60 else f"{time_left:.2f}s"
                else:
                    time_str = "?"
                
                print(f"Gen {gen:3d} | AvgLen: {avg_len:6.1f} | AvgFit: {avg_fit_str} | "
                      f"BestLen: {best_len:4d} | BestFit: {best_fit_str} | Time: {time_str}")
                
                if gen == self.generations - 1:
                    print('-' * 90)
            
            progress_callback = default_progress_callback
        
        # 初始化种群
        population = [self._random_program() for _ in range(self.population_size)]
        
        for gen in range(self.generations):
            # 评估适应度（带简约惩罚）
            raw_fitness = self._evaluate_population(population, fitness_func, fitness_args, fitness_kwargs)
            # 对复杂树施加惩罚（鼓励简约）
            fitnesses = [
                raw_fitness[i] - self.parsimony_coefficient * population[i].size()
                for i in range(len(population))
            ]
            
            # 统计信息
            avg_fit = np.mean(fitnesses)
            best_idx = np.argmax(fitnesses)
            best_fit = fitnesses[best_idx]
            avg_size = np.mean([p.size() for p in population])
            best_size = population[best_idx].size()
            
            # 进度回调
            if progress_callback:
                progress_callback(gen, avg_size, avg_fit, best_size, best_fit)
            
            # 精英保留
            elite_indices = np.argsort(fitnesses)[-self.hall_of_fame:]
            elites = [population[i] for i in elite_indices]
            
            # 生成下一代
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent = self._tournament(population, fitnesses)
                child = self._mutate_or_crossover(parent, population)
                # 限制树最大深度（避免指数爆炸）
                if child.depth() > 10:
                    child = self._hoist_mutation(child)  # 过深则剪枝
                new_population.append(child)
            population = new_population
        
        # 最终筛选最优程序
        final_fitness = self._evaluate_population(population, fitness_func, fitness_args, fitness_kwargs)
        best_indices = np.argsort(final_fitness)[-self.n_components:]
        self.best_programs_ = [population[i] for i in best_indices]
        return self

    def _random_program(self, method='half_and_half'):
        variables = self.variable_names
        min_depth, max_depth = self.init_depth
        depth = self.random_state.randint(min_depth, max_depth + 1)
        if method == 'grow':
            def grow(d):
                if d == 0 or (d > 0 and self.random_state.rand() < 0.5):
                    if self.const_range is not None and self.random_state.rand() < 0.5:
                        return Node(name=self.random_state.choice(variables), value=self.random_state.choice(variables))
                    elif self.const_range is not None:
                        return Node(name='const', value=self.random_state.uniform(*self.const_range))
                    else:
                        return Node(name=self.random_state.choice(variables), value=self.random_state.choice(variables))
                else:
                    items = list(self.function_set.all().items())
                    idx = self.random_state.randint(0, len(items))
                    func_name, (func, arity) = items[idx]
                    # 特殊处理时间窗口参数（确保是合理整数）
                    if arity == 2 and func_name.startswith('ts_'):
                        children = [grow(d - 1)]
                        window_size = self.random_state.randint(1, self.ts_window + 1)
                        children.append(Node(name='const', value=window_size))
                    elif arity == 3 and func_name.startswith('ts_'):
                        children = [grow(d - 1), grow(d - 1)]
                        window_size = self.random_state.randint(1, self.ts_window + 1)
                        children.append(Node(name='const', value=window_size))
                    elif arity == 2 and func_name in ['delay', 'delta', 'decay_linear']:
                        children = [grow(d - 1)]
                        window_size = self.random_state.randint(1, self.ts_window + 1)
                        children.append(Node(name='const', value=window_size))
                    elif arity == 3 and func_name in ['correlation', 'covariance']:
                        children = [grow(d - 1), grow(d - 1)]
                        window_size = self.random_state.randint(1, self.ts_window + 1)
                        children.append(Node(name='const', value=window_size))
                    else:
                        children = [grow(d - 1) for _ in range(arity)]
                    return Node(name=func_name, children=children)
            return grow(depth)
        elif method == 'full':
            def full(d):
                if d == 0:
                    if self.const_range is not None and self.random_state.rand() < 0.5:
                        return Node(name=self.random_state.choice(variables), value=self.random_state.choice(variables))
                    elif self.const_range is not None:
                        return Node(name='const', value=self.random_state.uniform(*self.const_range))
                    else:
                        return Node(name=self.random_state.choice(variables), value=self.random_state.choice(variables))
                else:
                    items = list(self.function_set.all().items())
                    idx = self.random_state.randint(0, len(items))
                    func_name, (func, arity) = items[idx]
                    # 特殊处理时间窗口参数（确保是合理整数）
                    if arity == 2 and func_name.startswith('ts_'):
                        children = [full(d - 1)]
                        window_size = self.random_state.randint(1, self.ts_window + 1)
                        children.append(Node(name='const', value=window_size))
                    elif arity == 3 and func_name.startswith('ts_'):
                        children = [full(d - 1), full(d - 1)]
                        window_size = self.random_state.randint(1, self.ts_window + 1)
                        children.append(Node(name='const', value=window_size))
                    elif arity == 2 and func_name in ['delay', 'delta', 'decay_linear']:
                        children = [full(d - 1)]
                        window_size = self.random_state.randint(1, self.ts_window + 1)
                        children.append(Node(name='const', value=window_size))
                    elif arity == 3 and func_name in ['correlation', 'covariance']:
                        children = [full(d - 1), full(d - 1)]
                        window_size = self.random_state.randint(1, self.ts_window + 1)
                        children.append(Node(name='const', value=window_size))
                    else:
                        children = [full(d - 1) for _ in range(arity)]
                    return Node(name=func_name, children=children)
            return full(depth)
        else:  # half_and_half
            if self.random_state.rand() < 0.5:
                return self._random_program(method='grow')
            else:
                return self._random_program(method='full')

    def _evaluate_population(self, population, fitness_func, args, kwargs):
        """评估种群适应度（支持并行，此处简化为串行）"""
        return [fitness_func(prog, *args, **kwargs) for prog in population]

    def _tournament(self, population, fitnesses):
        """锦标赛选择"""
        idxs = self.random_state.choice(len(population), self.tournament_size, replace=False)
        best = idxs[np.argmax([fitnesses[i] for i in idxs])]
        return population[best]

    def _mutate_or_crossover(self, parent, population):
        """遗传操作：交叉或变异"""
        op = self.random_state.rand()
        if op < self.p_crossover:
            # 交叉：与随机个体交换子树
            mate_idx = self.random_state.randint(0, len(population))
            mate = population[mate_idx]
            return self._crossover(parent, mate)
        elif op < self.p_crossover + self.p_subtree_mutation:
            # 子树变异：替换随机子树
            return self._subtree_mutation(parent)
        elif op < self.p_crossover + self.p_subtree_mutation + self.p_hoist_mutation:
            # 提升变异：将子节点提升为根
            return self._hoist_mutation(parent)
        elif op < self.p_crossover + self.p_subtree_mutation + self.p_hoist_mutation + self.p_point_mutation:
            # 点变异：替换单个节点
            return self._point_mutation(parent)
        else:
            # 复制当前节点
            return self._copy_tree(parent)

    def _copy_tree(self, node: Node) -> Node:
        """深拷贝树"""
        return Node(
            name=node.name,
            value=node.value,
            children=[self._copy_tree(c) for c in node.children]
        )
    
    def _all_nodes(self, node: Node) -> List[Node]:
        """获取所有节点列表"""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._all_nodes(child))
        return nodes

    def _crossover(self, parent1, parent2):
        """交叉操作"""
        t1 = self._copy_tree(parent1)
        t2 = self._copy_tree(parent2)
        nodes1 = self._all_nodes(t1)
        nodes2 = self._all_nodes(t2)
        if not nodes1 or not nodes2:
            return t1
        n1_idx = self.random_state.randint(0, len(nodes1))
        n2_idx = self.random_state.randint(0, len(nodes2))
        n1 = nodes1[n1_idx]
        n2 = nodes2[n2_idx]
        n1.name, n1.value, n1.children = n2.name, n2.value, [self._copy_tree(c) for c in n2.children]
        return t1

    def _subtree_mutation(self, node: Node) -> Node:
        """子树变异"""
        t = self._copy_tree(node)
        nodes = self._all_nodes(t)
        n_idx = self.random_state.randint(0, len(nodes))
        n = nodes[n_idx]
        depth = self.random_state.randint(*self.init_depth)
        new_subtree = self._random_program_with_depth(depth)
        # 添加空值检查
        if new_subtree is not None:
            n.name, n.value, n.children = new_subtree.name, new_subtree.value, new_subtree.children
        return t

    def _random_program_with_depth(self, depth: int) -> Node:
        variables = self.variable_names
        def build(current_depth):
            if current_depth >= depth:
                if self.random_state.rand() < 0.6:
                    var_name = self.random_state.choice(variables)
                    return Node(name=var_name, value=var_name)
                else:
                    if self.const_range is not None:
                        return Node(name='const', value=self.random_state.uniform(*self.const_range))
                    else:
                        # 如果禁止常数，就生成变量
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
            functions = list(self.function_set.all().items())
            idx = self.random_state.randint(0, len(functions))
            func_name, (_, arity) = functions[idx]
            # 特殊处理时间窗口参数（确保是合理整数）
            if arity == 2 and func_name.startswith('ts_'):
                children = [build(current_depth + 1)]
                window_size = self.random_state.randint(1, self.ts_window + 1)
                children.append(Node(name='const', value=window_size))
                return Node(name=func_name, children=children)
            elif arity == 3 and func_name.startswith('ts_'):
                children = [build(current_depth + 1), build(current_depth + 1)]
                window_size = self.random_state.randint(1, self.ts_window + 1)
                children.append(Node(name='const', value=window_size))
                return Node(name=func_name, children=children)
            elif arity == 2 and func_name in ['delay', 'delta', 'decay_linear']:
                children = [build(current_depth + 1)]
                window_size = self.random_state.randint(1, self.ts_window + 1)
                children.append(Node(name='const', value=window_size))
                return Node(name=func_name, children=children)
            elif arity == 3 and func_name in ['correlation', 'covariance']:
                children = [build(current_depth + 1), build(current_depth + 1)]
                window_size = self.random_state.randint(1, self.ts_window + 1)
                children.append(Node(name='const', value=window_size))
                return Node(name=func_name, children=children)
            else:
                children = [build(current_depth + 1) for _ in range(arity)]
                return Node(name=func_name, children=children)
        result = build(0)
        # 确保返回有效的Node对象
        if result is None:
            # 如果build函数返回None，创建一个默认的变量节点
            var_name = self.random_state.choice(variables)
            return Node(name=var_name, value=var_name)
        return result

    def _hoist_mutation(self, node: Node) -> Node:
        """提升变异（剪枝）"""
        t = self._copy_tree(node)
        nodes = self._all_nodes(t)
        if len(nodes) <= 1:
            return t
        sub_nodes = nodes[1:]
        n_idx = self.random_state.randint(0, len(sub_nodes))
        n = sub_nodes[n_idx]
        return self._copy_tree(n)

    def _point_mutation(self, node: Node) -> Node:
        """点变异"""
        t = self._copy_tree(node)
        def mutate(n: Node):
            if self.random_state.rand() < self.p_point_replace:
                if n.value is not None:
                    # 变量/常数突变
                    if self.random_state.rand() < 0.5:
                        # 变为其他变量
                        n.name = self.random_state.choice(self.variable_names)
                        n.value = n.name
                    else:
                        # 变为新常数
                        n.name = 'const'
                        if self.const_range is not None:
                            n.value = self.random_state.uniform(*self.const_range)
                        else:
                            # 如果禁止常数，就变为变量
                            n.name = self.random_state.choice(self.variable_names)
                            n.value = n.name
                else:
                    # 函数突变（保持参数个数）
                    current_arity = len(n.children)
                    # 筛选同参数个数的函数
                    candidates = [
                        (name, func) for name, (func, arity) in self.function_set.all().items()
                        if arity == current_arity
                    ]
                    if candidates:
                        idx = self.random_state.randint(0, len(candidates))
                        func_name, _ = candidates[idx]
                    n.name = func_name
            # 递归突变子节点
            for c in n.children:
                mutate(c)
        mutate(t)
        return t