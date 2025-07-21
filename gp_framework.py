"""
遗传编程(Genetic Programming)框架
=================================

这是一个用于金融因子挖掘的遗传编程框架，包含以下主要组件：

1. 基础函数集：包含数学运算、时间序列分析、排名等函数
2. 函数集管理类：动态管理可用的函数
3. 表达式树节点类：表示因子表达式的树形结构
4. 遗传编程主类：实现进化算法的核心逻辑

主要特性：
- 防止未来函数泄漏的滚动窗口计算
- 安全的数值计算（处理除零、溢出等异常）
- 支持多种遗传操作（交叉、变异、选择）
- 可扩展的函数集
- 树复杂度控制和简约惩罚

作者：AI助手
用途：股票因子挖掘和量化投资研究
"""

import numpy as np
import operator
import random
import time
from typing import List, Dict, Any, Callable, Optional

# ================= 基础函数集（增强版） =================
# 这些函数是构建因子表达式的基本组件，都经过安全性增强处理

def add(x, y): 
    """
    安全加法运算（自动广播）
    
    参数:
        x, y: 数值或numpy数组
    
    返回:
        x + y 的结果，支持广播
    """
    return np.add(x, y)

def sub(x, y): 
    """
    安全减法运算（自动广播）
    
    参数:
        x, y: 数值或numpy数组
    
    返回:
        x - y 的结果，支持广播
    """
    return np.subtract(x, y)

def mul(x, y): 
    """
    安全乘法运算（自动广播）
    
    参数:
        x, y: 数值或numpy数组
    
    返回:
        x * y 的结果，支持广播
    """
    return np.multiply(x, y)

def div(x, y): 
    """
    安全除法运算（处理除零和广播）
    
    当除数接近零时，将其替换为小的非零值以避免除零错误
    
    参数:
        x: 被除数
        y: 除数
    
    返回:
        x / y 的结果，除数为零时用小值替代
    """
    y_safe = np.where(np.abs(y) < 1e-10, 1e-10, y)  # 严格处理接近零的值
    return np.divide(x, y_safe)

def abs_(x): 
    """
    绝对值函数
    
    参数:
        x: 输入数值或数组
    
    返回:
        |x| 的绝对值
    """
    return np.abs(x)

def sqrt(x): 
    """
    安全平方根函数（处理负数）
    
    对负数取绝对值后再开方，避免复数结果
    
    参数:
        x: 输入数值或数组
    
    返回:
        sqrt(|x|) 的结果
    """
    return np.sqrt(np.abs(x))

def log(x): 
    """
    安全对数函数（处理零和负数）
    
    对输入取绝对值并加小偏移量，避免log(0)错误
    
    参数:
        x: 输入数值或数组
    
    返回:
        log(|x| + 1e-8) 的结果
    """
    return np.log(np.abs(x) + 1e-8)  # 加偏移避免零

def inv(x): 
    """
    安全倒数函数（处理接近零的值）
    
    当输入接近零时，用小值替代以避免除零
    
    参数:
        x: 输入数值或数组
    
    返回:
        1/x 的结果，x接近零时用小值替代
    """
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)  # 严格阈值
    return 1.0 / x_safe

def rank(x, d=10):
    """
    滚动窗口排名归一化（防止未来函数泄漏）
    
    在指定窗口长度内计算当前值的排名，并归一化到[0,1]区间
    这是量化研究中常用的因子标准化方法
    
    参数:
        x: 输入时间序列数据
        d: 窗口长度，默认10期。不满d期的位置填nan
    
    返回:
        归一化排名序列，前d-1期为nan
    
    注意:
        - 使用滚动窗口避免未来函数泄漏
        - 排名归一化有助于因子标准化
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
        # 遍历每一列（通常每列代表一只股票）
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
    """
    延迟函数（确保d为正整数，处理边界）
    
    将时间序列向后移动d期，常用于构建滞后因子
    
    参数:
        x: 输入时间序列
        d: 延迟期数，必须为正整数
    
    返回:
        延迟d期的序列，前d期填充NaN
    
    示例:
        delay([1,2,3,4,5], 2) -> [nan, nan, 1, 2, 3]
    """
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
    """
    滚动窗口相关系数计算（防止未来函数泄漏）
    
    计算两个时间序列在滚动窗口内的皮尔逊相关系数
    
    参数:
        x, y: 两个输入时间序列
        d: 滚动窗口长度
    
    返回:
        滚动相关系数序列，前d-1期为nan
    
    注意:
        - 相关系数范围为[-1, 1]
        - 窗口长度至少需要2个观测值
    """
    d = int(round(d))
    d = max(1, min(d, 100))  # 限制最大窗口长度
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
    """
    滚动窗口协方差计算（防止未来函数泄漏）
    
    计算两个时间序列在滚动窗口内的协方差
    
    参数:
        x, y: 两个输入时间序列
        d: 滚动窗口长度
    
    返回:
        滚动协方差序列，前d-1期为nan
    
    注意:
        协方差反映两个变量的线性关系强度和方向
    """
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
    """
    滚动窗口缩放至绝对值和为1（防止未来函数泄漏）
    
    将当前值除以滚动窗口内所有值的绝对值和，实现标准化
    
    参数:
        x: 输入时间序列
        d: 滚动窗口长度，默认10
    
    返回:
        标准化后的序列，前d-1期为nan
    
    用途:
        因子标准化，使因子值在合理范围内
    """
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
    """
    当前值与延迟d期的差值（复用delay函数）
    
    计算 x[t] - x[t-d]，常用于构建动量类因子
    
    参数:
        x: 输入时间序列
        d: 延迟期数
    
    返回:
        差值序列
    
    示例:
        delta([1,2,3,4,5], 2) -> [nan, nan, 2, 2, 2]  # 3-1=2, 4-2=2, 5-3=2
    """
    return sub(x, delay(x, d))

def signedpower(x): 
    """
    带符号二次幂运算 sign(x) * (abs(x) ** 2)
    
    保持原始符号的同时放大数值差异
    
    参数:
        x: 输入数值或数组
    
    返回:
        带符号的二次幂结果
    
    特点:
        - 正数保持正，负数保持负
        - 放大绝对值大的数，压缩绝对值小的数
    """
    return np.sign(x) * (np.abs(x) ** 2)

def decay_linear(x, d):
    """
    滚动窗口线性加权移动平均（防止未来函数泄漏）
    
    对历史数据赋予线性递减权重，越近期的数据权重越大
    
    参数:
        x: 输入时间序列
        d: 窗口长度
    
    返回:
        线性加权移动平均序列
    
    权重分配:
        最近期权重最大，最远期权重最小
        权重序列: [1, 2, 3, ..., d] (归一化后)
    """
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

# ================= 时间序列函数（滚动窗口处理，防止未来函数泄漏） =================
# 这些函数专门用于时间序列分析，都采用滚动窗口避免未来函数泄漏

def ts_min(x, d): 
    """
    滚动窗口最小值
    
    计算滚动窗口内的最小值
    
    参数:
        x: 输入时间序列
        d: 窗口长度
    
    返回:
        滚动最小值序列，前d-1期为nan
    """
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
    """
    滚动窗口最大值
    
    计算滚动窗口内的最大值
    
    参数:
        x: 输入时间序列
        d: 窗口长度
    
    返回:
        滚动最大值序列，前d-1期为nan
    """
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
    """
    滚动窗口最小值位置
    
    返回滚动窗口内最小值的位置索引
    
    参数:
        x: 输入时间序列
        d: 窗口长度
    
    返回:
        最小值位置序列，前d-1期为nan
    """
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
    """
    滚动窗口最大值位置
    
    返回滚动窗口内最大值的位置索引
    
    参数:
        x: 输入时间序列
        d: 窗口长度
    
    返回:
        最大值位置序列，前d-1期为nan
    """
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
    """
    滚动窗口排名
    
    计算当前值在滚动窗口内的排名位置
    
    参数:
        x: 输入时间序列
        d: 窗口长度
    
    返回:
        排名序列，前d-1期为nan
    
    注意:
        排名越高表示当前值在窗口内越大
    """
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
    """
    滚动窗口求和
    
    计算滚动窗口内所有值的和
    
    参数:
        x: 输入时间序列
        d: 窗口长度
    
    返回:
        滚动求和序列，前d-1期为nan
    """
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
    """
    滚动窗口乘积
    
    计算滚动窗口内所有值的乘积
    
    参数:
        x: 输入时间序列
        d: 窗口长度
    
    返回:
        滚动乘积序列，前d-1期为nan
    
    注意:
        乘积容易产生极大或极小值，使用时需注意数值稳定性
    """
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
    """
    滚动窗口标准差
    
    计算滚动窗口内的标准差
    
    参数:
        x: 输入时间序列
        d: 窗口长度
    
    返回:
        滚动标准差序列，前d-1期为nan
    
    用途:
        衡量波动性，常用于风险度量
    """
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

# 函数别名，保持与其他函数命名一致性
def ts_corr(x, y, d):
    """滚动窗口相关系数（别名）"""
    return correlation(x, y, d)  # 复用优化后的correlation

def ts_cov(x, y, d):
    """滚动窗口协方差（别名）"""
    return covariance(x, y, d)  # 复用优化后的covariance

ts_prod = ts_product  # 保持一致性

def ts_zscore(x, d):
    """
    滚动窗口Z-score标准化
    
    计算 (当前值 - 窗口均值) / 窗口标准差
    
    参数:
        x: 输入时间序列
        d: 窗口长度
    
    返回:
        Z-score标准化序列，前d-1期为nan
    
    注意:
        Z-score衡量当前值偏离均值的程度（以标准差为单位）
    """
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

# ================= 排名操作函数（增强鲁棒性） =================
# 这些函数基于排名进行运算，常用于因子中性化处理

def rank_sub(x, y, d=10):
    """
    滚动窗口排名差值（防止未来函数泄漏）
    
    计算两个序列排名的差值
    
    参数:
        x, y: 两个输入时间序列
        d: 排名窗口长度
    
    返回:
        排名差值序列
    
    用途:
        构建相对强度因子，比较两个指标的相对表现
    """
    rx = rank(x, d)
    ry = rank(y, d)
    try:
        return rx - ry
    except:
        # 形状不匹配时返回0数组
        return np.zeros_like(rx) if hasattr(rx, 'size') and rx.size > 0 else 0

def rank_div(x, y, d=10):
    """
    滚动窗口排名比值（防止未来函数泄漏）
    
    计算两个序列排名的比值
    
    参数:
        x, y: 两个输入时间序列
        d: 排名窗口长度
    
    返回:
        排名比值序列
    
    用途:
        构建相对强度因子，比较两个指标的相对表现
    """
    rx = rank(x, d)
    ry = rank(y, d)
    try:
        ry_safe = np.where(ry < 1e-10, 1e-10, ry)  # 避免除零
        return rx / ry_safe
    except:
        return np.ones_like(rx) if hasattr(rx, 'size') and rx.size > 0 else 1

def sigmoid(x):
    """
    安全sigmoid函数（避免溢出）
    
    计算 1 / (1 + exp(-x))，将输入映射到(0,1)区间
    
    参数:
        x: 输入数值或数组
    
    返回:
        sigmoid变换后的结果
    
    特点:
        - 输出范围(0,1)
        - 截断极端值避免数值溢出
        - 常用于因子标准化
    """
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))  # 截断极端值


class FunctionSet:
    """
    函数集管理类
    
    管理遗传编程中可用的函数集合，支持动态增删函数
    
    属性:
        functions: 字典，存储函数名到(函数对象, 参数个数)的映射
    
    功能:
        - 注册新函数
        - 删除函数
        - 查询函数信息
        - 获取所有可用函数
    """
    def __init__(self):
        """
        初始化函数集，注册所有基础函数
        
        每个函数条目格式：'函数名': (函数对象, 参数个数)
        """
        self.functions = {
            # 基础数学运算（2个参数）
            'add': (add, 2),
            'sub': (sub, 2),
            'mul': (mul, 2),
            'div': (div, 2),
            
            # 单元函数（1个参数）
            'abs': (abs_, 1),
            'sqrt': (sqrt, 1),
            'log': (log, 1),
            'inv': (inv, 1),
            'signedpower': (signedpower, 1),
            'sigmoid': (sigmoid, 1),
            
            # 时间序列函数（2个参数：数据+窗口）
            'rank': (rank, 2),
            'delay': (delay, 2),
            'scale': (scale, 2),
            'delta': (delta, 2),
            'decay_linear': (decay_linear, 2),
            'ts_min': (ts_min, 2),
            'ts_max': (ts_max, 2),
            'ts_argmin': (ts_argmin, 2),
            'ts_argmax': (ts_argmax, 2),
            'ts_rank': (ts_rank, 2),
            'ts_sum': (ts_sum, 2),
            'ts_product': (ts_product, 2),
            'ts_stddev': (ts_stddev, 2),
            'ts_prod': (ts_prod, 2),
            'ts_zscore': (ts_zscore, 2),
            
            # 双变量时间序列函数（3个参数：数据1+数据2+窗口）
            'correlation': (correlation, 3),
            'covariance': (covariance, 3),
            'ts_corr': (ts_corr, 3),
            'ts_cov': (ts_cov, 3),
            'rank_sub': (rank_sub, 3),
            'rank_div': (rank_div, 3),
        }
    
    def add_function(self, name: str, func: Callable, arity: int):
        """
        添加新函数到函数集
        
        参数:
            name: 函数名称
            func: 函数对象
            arity: 函数参数个数
        """
        self.functions[name] = (func, arity)
    
    def get(self, name: str):
        """
        获取函数信息
        
        参数:
            name: 函数名称
        
        返回:
            (函数对象, 参数个数) 元组
        """
        return self.functions[name]
    
    def all(self):
        """
        获取所有函数的副本
        
        返回:
            包含所有函数信息的字典副本
        """
        return self.functions.copy()
    
    def remove_function(self, name: str):
        """
        从函数集中删除函数
        
        参数:
            name: 要删除的函数名称
        """
        if name in self.functions:
            del self.functions[name]


class Node:
    """
    表达式树节点类
    
    表示因子表达式的树形结构，每个节点可以是：
    1. 叶子节点：变量或常数
    2. 内部节点：函数调用
    
    属性:
        name: 节点名称（函数名或变量名）
        children: 子节点列表
        value: 节点值（变量名或常数值，仅叶子节点使用）
    """
    def __init__(self, name: str, children: List['Node'] = None, value: Any = None):
        """
        初始化节点
        
        参数:
            name: 节点名称
            children: 子节点列表
            value: 节点值（变量名或常数）
        """
        self.name = name
        self.children = children or []
        self.value = value  # 变量名（str）或常数（number）

    def evaluate(self, X, function_set):
        """
        递归评估表达式树
        
        参数:
            X: 输入数据，可以是字典{变量名: 数据}或其他格式
            function_set: 函数集对象
        
        返回:
            表达式计算结果
        
        评估逻辑:
            1. 叶子节点：返回变量值或常数
            2. 内部节点：递归计算子节点，然后调用函数
        """
        # 处理叶子节点（变量或常数）
        if self.value is not None:
            if isinstance(self.value, str):  # 变量节点
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
            else:  # 常数节点
                if isinstance(X, dict) and len(X) > 0:
                    # 返回与数据同形状的常数数组
                    arr = next(iter(X.values()))
                    return np.full_like(arr, self.value, dtype=np.float64)
                return self.value
        
        # 处理内部节点（函数调用）
        func, arity = function_set.get(self.name)
        # 递归计算所有子节点
        args = [child.evaluate(X, function_set) for child in self.children]
        
        try:
            # 调用函数
            result = func(*args)
            
            # 结果后处理：确保返回合法的数值
            if result is None or (isinstance(result, float) and np.isnan(result)):
                if isinstance(X, dict) and len(X) > 0:
                    arr = next(iter(X.values()))
                    return np.full_like(arr, np.nan, dtype=np.float64)
                return np.nan
            
            # 标量直接返回
            if np.isscalar(result):
                return result
            
            # 0维数组转为float
            if isinstance(result, np.ndarray) and result.shape == ():
                return float(result)
            
            # object类型数组转为float64
            if isinstance(result, np.ndarray) and result.dtype == object:
                try:
                    return result.astype(np.float64)
                except Exception:
                    arr = next(iter(X.values())) if isinstance(X, dict) and len(X) > 0 else 0
                    return np.full_like(arr, np.nan, dtype=np.float64)
            
            return result
        except Exception:
            # 函数调用失败时返回NaN
            if isinstance(X, dict) and len(X) > 0:
                arr = next(iter(X.values()))
                return np.full_like(arr, np.nan, dtype=np.float64)
            return np.nan
    
    def to_str(self) -> str:
        """
        将表达式树转换为字符串表示
        
        返回:
            表达式的字符串形式
        
        示例:
            add(mul(x, 2), y) -> "add(mul(x, 2), y)"
        """
        if self.value is not None:
            return str(self.value)
        return f"{self.name}({', '.join([c.to_str() for c in self.children])})"

    def depth(self) -> int:
        """
        计算表达式树的深度
        
        返回:
            树的最大深度
        
        用途:
            控制表达式复杂度，避免过深的树结构
        """
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def size(self) -> int:
        """
        计算表达式树的节点总数
        
        返回:
            树中节点的总数量
        
        用途:
            控制表达式复杂度，用于简约惩罚
        """
        return 1 + sum(child.size() for child in self.children)


class GeneticProgrammer:
    """
    遗传编程主类
    
    实现遗传编程算法用于自动化因子挖掘
    
    核心流程:
    1. 初始化随机种群
    2. 评估适应度
    3. 选择、交叉、变异产生新一代
    4. 重复进化直到收敛
    
    主要特性:
    - 多种遗传操作（交叉、子树变异、点变异等）
    - 锦标赛选择
    - 精英保留策略
    - 简约惩罚（控制树复杂度）
    - 进度监控和回调
    """
    def __init__(
        self,
        generations: int = 30,  # 进化代数
        population_size: int = 100,  # 种群规模
        n_components: int = 5,  # 保留的最优个体数量
        hall_of_fame: int = 10,  # 精英保留数量
        function_set: FunctionSet = None,  # 函数集
        parsimony_coefficient: float = 0.001,  # 简约系数（惩罚复杂树）
        tournament_size: int = 7,  # 锦标赛规模
        random_state: int = 42,  # 随机种子
        init_depth: tuple = (2, 6),  # 初始树深度范围
        const_range: Optional[tuple] = (-2, 2),  # 常数范围
        ts_window: int = 30,  # 时间窗口范围
        # 遗传操作概率
        p_crossover: float = 0.6,  # 交叉概率
        p_subtree_mutation: float = 0.25,  # 子树变异概率
        p_hoist_mutation: float = 0.05,  # 提升变异概率
        p_point_mutation: float = 0.08,  # 点变异概率
        p_point_replace: float = 0.3,  # 点替换概率
        variable_names: List[str] = None  # 变量名列表
    ):
        """
        初始化遗传编程器
        
        参数说明详见类属性注释
        """
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
        self.best_programs_ = []  # 存储最优程序
    
    def fit(
        self,
        fitness_func: Callable,  # 适应度函数
        fitness_args: tuple = (),  # 适应度函数额外参数
        fitness_kwargs: dict = None,  # 适应度函数关键字参数
        progress_callback: Callable = None  # 进度回调函数
    ):
        """
        执行遗传编程进化过程
        
        参数:
            fitness_func: 适应度评估函数，接收个体并返回适应度值
            fitness_args: 适应度函数的额外位置参数
            fitness_kwargs: 适应度函数的关键字参数
            progress_callback: 进度回调函数，接收(代数, 平均长度, 平均适应度, 最优长度, 最优适应度)
        
        返回:
            self: 训练后的遗传编程器对象
        """
        fitness_kwargs = fitness_kwargs or {}
        
        # 如果没有传入进度回调，使用默认的进度显示
        if progress_callback is None:
            start_time = time.time()
            
            def default_progress_callback(gen, avg_len, avg_fit, best_len, best_fit):
                """
                默认进度显示函数
                
                显示训练进度，包括代数、平均/最优适应度、剩余时间等信息
                """
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
        
        # 初始化种群：生成随机表达式树
        population = [self._random_program() for _ in range(self.population_size)]
        
        # 主进化循环
        for gen in range(self.generations):
            # 评估种群适应度
            raw_fitness = self._evaluate_population(population, fitness_func, fitness_args, fitness_kwargs)
            
            # 应用简约惩罚：复杂的树会被惩罚
            fitnesses = [
                raw_fitness[i] - self.parsimony_coefficient * population[i].size()
                for i in range(len(population))
            ]
            
            # 统计当前代信息
            avg_fit = np.mean(fitnesses)
            best_idx = np.argmax(fitnesses)
            best_fit = fitnesses[best_idx]
            avg_size = np.mean([p.size() for p in population])
            best_size = population[best_idx].size()
            
            # 调用进度回调
            if progress_callback:
                progress_callback(gen, avg_size, avg_fit, best_size, best_fit)
            
            # 精英保留：保留最优个体到下一代
            elite_indices = np.argsort(fitnesses)[-self.hall_of_fame:]
            elites = [population[i] for i in elite_indices]
            
            # 生成下一代种群
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                # 锦标赛选择父代
                parent = self._tournament(population, fitnesses)
                # 应用遗传操作产生子代
                child = self._mutate_or_crossover(parent, population)
                # 控制树深度：过深则剪枝
                if child.depth() > 10:
                    child = self._hoist_mutation(child)
                new_population.append(child)
            
            population = new_population
        
        # 最终筛选最优程序
        final_fitness = self._evaluate_population(population, fitness_func, fitness_args, fitness_kwargs)
        best_indices = np.argsort(final_fitness)[-self.n_components:]
        self.best_programs_ = [population[i] for i in best_indices]
        return self

    def _random_program(self, method='half_and_half'):
        """
        生成随机表达式树程序
        
        参数:
            method: 生成方法
                - 'grow': 生长法（随机深度）
                - 'full': 完全法（固定深度）
                - 'half_and_half': 混合法（随机选择上述两种）
        
        返回:
            随机生成的表达式树根节点
        
        生成策略:
            - 叶子节点：变量或常数
            - 内部节点：函数调用
            - 特殊处理时间序列函数的窗口参数
        """
        variables = self.variable_names
        min_depth, max_depth = self.init_depth
        depth = self.random_state.randint(min_depth, max_depth + 1)
        
        if method == 'grow':
            def grow(d):
                # 到达最大深度或随机决定生成叶子节点
                if d == 0 or (d > 0 and self.random_state.rand() < 0.5):
                    if self.const_range is not None and self.random_state.rand() < 0.5:
                        # 生成变量节点
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                    elif self.const_range is not None:
                        # 生成常数节点
                        const_val = self.random_state.uniform(*self.const_range)
                        return Node(name='const', value=const_val)
                    else:
                        # 只允许变量节点
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                else:
                    # 生成函数节点
                    items = list(self.function_set.all().items())
                    idx = self.random_state.randint(0, len(items))
                    func_name, (func, arity) = items[idx]
                    
                    # 特殊处理时间序列函数的窗口参数
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
                        # 普通函数：生成对应数量的子节点
                        children = [grow(d - 1) for _ in range(arity)]
                    
                    return Node(name=func_name, children=children)
            return grow(depth)
            
        elif method == 'full':
            def full(d):
                # 完全法：只在最大深度生成叶子节点
                if d == 0:
                    if self.const_range is not None and self.random_state.rand() < 0.5:
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                    elif self.const_range is not None:
                        const_val = self.random_state.uniform(*self.const_range)
                        return Node(name='const', value=const_val)
                    else:
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
                else:
                    # 与grow方法类似的函数节点生成逻辑
                    items = list(self.function_set.all().items())
                    idx = self.random_state.randint(0, len(items))
                    func_name, (func, arity) = items[idx]
                    
                    # 特殊处理时间序列函数（代码逻辑同grow方法）
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
            # 随机选择grow或full方法
            if self.random_state.rand() < 0.5:
                return self._random_program(method='grow')
            else:
                return self._random_program(method='full')

    def _evaluate_population(self, population, fitness_func, args, kwargs):
        """
        评估种群中所有个体的适应度
        
        参数:
            population: 种群列表
            fitness_func: 适应度函数
            args: 适应度函数参数
            kwargs: 适应度函数关键字参数
        
        返回:
            适应度值列表
        
        注意:
            此处为串行评估，可扩展为并行评估以提高性能
        """
        return [fitness_func(prog, *args, **kwargs) for prog in population]

    def _tournament(self, population, fitnesses):
        """
        锦标赛选择
        
        从种群中随机选择若干个体进行比较，返回适应度最高的个体
        
        参数:
            population: 种群列表
            fitnesses: 适应度列表
        
        返回:
            选中的个体
        
        优势:
            - 选择压力可调（通过锦标赛规模控制）
            - 不需要适应度排序
            - 支持负适应度
        """
        idxs = self.random_state.choice(len(population), self.tournament_size, replace=False)
        best = idxs[np.argmax([fitnesses[i] for i in idxs])]
        return population[best]

    def _mutate_or_crossover(self, parent, population):
        """
        应用遗传操作：根据概率选择交叉或变异
        
        参数:
            parent: 父代个体
            population: 当前种群
        
        返回:
            经过遗传操作的子代个体
        
        操作类型:
            1. 交叉：与随机个体交换子树
            2. 子树变异：替换随机子树
            3. 提升变异：将子节点提升为根
            4. 点变异：替换单个节点
            5. 复制：直接复制（保持多样性）
        """
        op = self.random_state.rand()
        
        if op < self.p_crossover:
            # 交叉操作
            mate_idx = self.random_state.randint(0, len(population))
            mate = population[mate_idx]
            return self._crossover(parent, mate)
        elif op < self.p_crossover + self.p_subtree_mutation:
            # 子树变异
            return self._subtree_mutation(parent)
        elif op < self.p_crossover + self.p_subtree_mutation + self.p_hoist_mutation:
            # 提升变异
            return self._hoist_mutation(parent)
        elif op < self.p_crossover + self.p_subtree_mutation + self.p_hoist_mutation + self.p_point_mutation:
            # 点变异
            return self._point_mutation(parent)
        else:
            # 直接复制
            return self._copy_tree(parent)

    def _copy_tree(self, node: Node) -> Node:
        """
        深拷贝表达式树
        
        参数:
            node: 要复制的节点
        
        返回:
            复制后的新节点
        """
        return Node(
            name=node.name,
            value=node.value,
            children=[self._copy_tree(c) for c in node.children]
        )
    
    def _all_nodes(self, node: Node) -> List[Node]:
        """
        获取表达式树中所有节点的列表
        
        参数:
            node: 根节点
        
        返回:
            包含所有节点的列表（前序遍历）
        """
        nodes = [node]
        for child in node.children:
            nodes.extend(self._all_nodes(child))
        return nodes

    def _crossover(self, parent1, parent2):
        """
        交叉操作：交换两个个体的随机子树
        
        参数:
            parent1, parent2: 两个父代个体
        
        返回:
            交叉后的子代个体
        
        过程:
            1. 复制两个父代
            2. 随机选择交叉点
            3. 交换子树
        """
        t1 = self._copy_tree(parent1)
        t2 = self._copy_tree(parent2)
        
        nodes1 = self._all_nodes(t1)
        nodes2 = self._all_nodes(t2)
        
        if not nodes1 or not nodes2:
            return t1
        
        # 随机选择交叉点
        n1_idx = self.random_state.randint(0, len(nodes1))
        n2_idx = self.random_state.randint(0, len(nodes2))
        n1 = nodes1[n1_idx]
        n2 = nodes2[n2_idx]
        
        # 交换子树
        n1.name, n1.value, n1.children = n2.name, n2.value, [self._copy_tree(c) for c in n2.children]
        return t1

    def _subtree_mutation(self, node: Node) -> Node:
        """
        子树变异：用随机生成的子树替换原有子树
        
        参数:
            node: 要变异的个体
        
        返回:
            变异后的个体
        """
        t = self._copy_tree(node)
        nodes = self._all_nodes(t)
        
        # 随机选择变异点
        n_idx = self.random_state.randint(0, len(nodes))
        n = nodes[n_idx]
        
        # 生成新子树替换
        depth = self.random_state.randint(*self.init_depth)
        new_subtree = self._random_program_with_depth(depth)
        
        if new_subtree is not None:
            n.name, n.value, n.children = new_subtree.name, new_subtree.value, new_subtree.children
        return t

    def _random_program_with_depth(self, depth: int) -> Node:
        """
        生成指定深度的随机程序
        
        参数:
            depth: 目标深度
        
        返回:
            生成的表达式树根节点
        """
        variables = self.variable_names
        
        def build(current_depth):
            if current_depth >= depth:
                # 到达目标深度，生成叶子节点
                if self.random_state.rand() < 0.6:
                    var_name = self.random_state.choice(variables)
                    return Node(name=var_name, value=var_name)
                else:
                    if self.const_range is not None:
                        const_val = self.random_state.uniform(*self.const_range)
                        return Node(name='const', value=const_val)
                    else:
                        var_name = self.random_state.choice(variables)
                        return Node(name=var_name, value=var_name)
            
            # 生成函数节点
            functions = list(self.function_set.all().items())
            idx = self.random_state.randint(0, len(functions))
            func_name, (_, arity) = functions[idx]
            
            # 特殊处理时间序列函数（与_random_program方法逻辑相同）
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
        
        # 确保返回有效节点
        if result is None:
            var_name = self.random_state.choice(variables)
            return Node(name=var_name, value=var_name)
        return result

    def _hoist_mutation(self, node: Node) -> Node:
        """
        提升变异：随机选择一个子节点提升为根节点
        
        参数:
            node: 要变异的个体
        
        返回:
            变异后的个体
        
        作用:
            减少树的复杂度，实现自动剪枝
        """
        t = self._copy_tree(node)
        nodes = self._all_nodes(t)
        
        if len(nodes) <= 1:
            return t
        
        # 选择非根节点进行提升
        sub_nodes = nodes[1:]
        n_idx = self.random_state.randint(0, len(sub_nodes))
        n = sub_nodes[n_idx]
        
        return self._copy_tree(n)

    def _point_mutation(self, node: Node) -> Node:
        """
        点变异：随机替换树中的单个节点
        
        参数:
            node: 要变异的个体
        
        返回:
            变异后的个体
        
        变异类型:
            - 变量/常数节点：替换为其他变量或常数
            - 函数节点：替换为同参数数量的其他函数
        """
        t = self._copy_tree(node)
        
        def mutate(n: Node):
            if self.random_state.rand() < self.p_point_replace:
                if n.value is not None:
                    # 叶子节点变异
                    if self.random_state.rand() < 0.5:
                        # 变为其他变量
                        var_name = self.random_state.choice(self.variable_names)
                        n.name = var_name
                        n.value = var_name
                    else:
                        # 变为新常数
                        if self.const_range is not None:
                            n.name = 'const'
                            n.value = self.random_state.uniform(*self.const_range)
                        else:
                            # 如果禁止常数，变为变量
                            var_name = self.random_state.choice(self.variable_names)
                            n.name = var_name
                            n.value = var_name
                else:
                    # 函数节点变异：保持参数个数不变
                    current_arity = len(n.children)
                    candidates = [
                        (name, func) for name, (func, arity) in self.function_set.all().items()
                        if arity == current_arity
                    ]
                    if candidates:
                        idx = self.random_state.randint(0, len(candidates))
                        func_name, _ = candidates[idx]
                        n.name = func_name
            
            # 递归变异子节点
            for c in n.children:
                mutate(c)
        
        mutate(t)
        return t