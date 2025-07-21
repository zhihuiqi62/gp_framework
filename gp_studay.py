import pandas as pd
import numpy as np
from gp_framework import GeneticProgrammer, FunctionSet
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 读取数据
df = pd.read_csv('stocks_data.csv')

# 选择特征和目标
features = [
    'open', 'close', 'high', 'low', 'avg', 'volume', 'money',
    'pre_close', 'market_cap', 'circulating_market_cap',
    'turnover_ratio', 'pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio'
]
target = 'future_5day_return'

# 数据预处理
print(f"原始数据行数: {len(df)}")
data = df[features + [target, 'code', 'time']].copy()



# 构造训练集X, y（T, N）格式，T为时间，N为股票数
print("\n构造训练数据...")
pivoted = {}
for col in features + [target]:
    try:
        pivoted[col] = data.pivot(index='time', columns='code', values=col)
        print(f"特征 {col} 转换成功，形状: {pivoted[col].shape}")
    except Exception as e:
        print(f"特征 {col} 转换失败: {e}")
        # 创建全零矩阵作为替代
        unique_dates = pd.Series(data['time']).unique()
        unique_codes = pd.Series(data['code']).unique()
        pivoted[col] = pd.DataFrame(0, index=unique_dates, columns=unique_codes)

# 训练集
X_dict = {f: pivoted[f].values for f in features}
y = pivoted[target].values  # (T, N)
print(f"\n训练数据准备完成:")
print(f"时间点数量: {y.shape[0]}")
print(f"股票数量: {y.shape[1]}")


def fitness_func(prog, X_dict, y, return_details=False):
    """
    计算因子表达式的适应度（时间截面rankIC均值）
    
    参数:
        prog: Node表达式树
        X_dict: dict of ndarray, 每个特征 (T, N)
        y: (T, N) 目标变量
        return_details: 是否返回详细信息（用于分析）
    
    返回:
        float: 所有时间截面rankIC的均值
    """
    try:
        # 计算因子值（直接使用传入的function_set实例，避免重复创建）
        pred = prog.evaluate(X_dict, FunctionSet())  # (T, N)
        
        # 确保pred是二维数组
        # 全量兜底，保证pred是float64二维数组
        pred = np.asarray(pred, dtype=np.float64)
        # if pred.size == 1:
        #     pred = np.full_like(y, float(pred), dtype=np.float64)
        # elif pred.shape == ():  # 0维
        #     pred = np.full_like(y, pred, dtype=np.float64)
        # elif pred.ndim == 1:
        #     if pred.size == y.shape[0]:
        #         pred = np.tile(pred.reshape(-1, 1), (1, y.shape[1]))
        #     elif pred.size == y.shape[1]:
        #         pred = np.tile(pred.reshape(1, -1), (y.shape[0], 1))
        #     else:
        #         return -np.inf
        # elif pred.shape != y.shape:
        #     try:
        #         pred = np.broadcast_to(pred, y.shape)
        #     except Exception:
        #         return -np.inf
        if pred.shape != y.shape:
            return -np.inf
        
        # 向量化计算每个时间截面的rankIC
        ic_values = []
        valid_ts = 0
        
        for t in range(y.shape[0]):
            # 当前时间截面的预测值和真实值
            pred_t = pred[t].ravel()
            y_t = y[t].ravel()
            
            # 过滤缺失值
            mask = ~np.isnan(pred_t) & ~np.isnan(y_t)
            if np.sum(mask) < 10:  # 至少需要10个有效样本
                continue
                
            # 计算Spearman秩相关系数（使用numpy实现，避免Series转换开销）
            pred_rank = pd.Series(pred_t[mask]).rank()
            y_rank = pd.Series(y_t[mask]).rank()
            corr = np.corrcoef(pred_rank, y_rank)[0, 1]
            
            if not np.isnan(corr):
                ic_values.append(corr)
                valid_ts += 1
        
        # 计算最终适应度（增加稳定性惩罚项）
        if len(ic_values) < 5:  # 至少需要5个有效时间点
            return -np.inf
            
        mean_ic = np.mean(ic_values)
        icir = mean_ic / (np.std(ic_values) + 1e-8)
        
        # 综合评分：IC均值 + ICIR（稳定性）
        fitness = mean_ic + 0.2 * np.tanh(icir)
        
        if return_details:
            return {
                'fitness': fitness,
                'mean_ic': mean_ic,
                'icir': icir,
                'valid_ts': valid_ts,
                'total_ts': y.shape[0]
            }
        
        return fitness
        
    except Exception as e:
        print(f"适应度计算错误: {e}")
        return -np.inf

# 优化后的遗传规划器配置
gp = GeneticProgrammer(
    generations=7,  # 增加迭代轮数
    population_size=50,  # 扩大种群规模
    tournament_size=10,  # 添加这个：每次选10个个体进行锦标赛
    n_components=5,  # 输出更多候选因子
    hall_of_fame=7,  # 增加精英保留数量
    function_set=FunctionSet(),
    variable_names=features,
    ts_window=15,  # d 时间窗口
    random_state=30,
    const_range=None,  # 禁止常数使用
    p_crossover=0.7,
    p_subtree_mutation=0.2,
    p_hoist_mutation=0.05,
    p_point_mutation=0.05,
    # 增强复杂度控制
    parsimony_coefficient=0,  # 增加对复杂树的惩罚
    init_depth=(4, 7)  # 扩大初始树深度范围
)

# 训练
gp.fit(fitness_func=fitness_func, fitness_args=(X_dict, y))

# 输出最优因子及其绩效指标
print('\n最优因子表达式及其绩效:')
print('=' * 100)

def export_graphviz(node, dot=None, parent=None, node_id=None, counter=[0]):
    import graphviz
    if dot is None:
        dot = graphviz.Digraph(format='png')
        dot.attr('node', shape='ellipse', style='filled', color='lightblue')
        counter[0] = 0  # reset counter
    if node_id is None:
        node_id = f'n{counter[0]}'
        counter[0] += 1
    label = node.name if node.value is None else str(node.value)
    dot.node(node_id, label=label)
    for i, child in enumerate(node.children):
        child_id = f'n{counter[0]}'
        counter[0] += 1
        export_graphviz(child, dot, node_id, child_id, counter)
        dot.edge(node_id, child_id)
    return dot

for i, prog in enumerate(gp.best_programs_[:]):  # 显示所有因子
    details = fitness_func(prog, X_dict, y, return_details=True)
    expr = prog.to_str()
    depth = prog.depth()
    size = prog.size()
    print(f"因子 {i+1}:")
    print(f"  表达式: {expr}")
    print(f"  适应度: {details['fitness']:.6f} | 平均IC: {details['mean_ic']:.6f} | ICIR: {details['icir']:.6f}")
    print(f"  复杂度: 深度={depth}, 节点数={size}, 有效时间点={details['valid_ts']}/{details['total_ts']}")
    
    # 生成 graphviz 可视化图片
    try:
        dot = export_graphviz(prog)
        filename = f'factor_{i+1}_graphviz'
        dot.render(filename, format='png', cleanup=True)
        print(f"  Graphviz可视化: 已保存至 {filename}.png")
    except ImportError:
        print(f"  Graphviz可视化: 失败（需要安装 graphviz 包和系统依赖）")
    
    print('-' * 100)