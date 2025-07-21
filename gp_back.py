import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置matplotlib字体，解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.family'] = 'sans-serif'  # 设置字体族

from gp_framework import  div, sub

# 1. 读取数据
features = [
    'open', 'close', 'high', 'low', 'avg', 'volume', 'money',
    'pre_close', 'market_cap', 'circulating_market_cap',
    'turnover_ratio', 'pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio'
]
target = 'future_5day_return'

# 只保留需要的列，节省内存
use_cols = features + [target, 'code', 'time']
df = pd.read_csv('stocks_data.csv', usecols=lambda col: col in use_cols)
# 确保time为datetime类型

df['time'] = pd.to_datetime(df['time'])

# # 2. 数据预处理
# # 删除目标缺失
# df = df.dropna(subset=[target]).reset_index(drop=True)
# # 填充特征缺失
# for col in features:
#     df[col] = df.groupby('code')[col].apply(lambda x: x.ffill().bfill()).fillna(0)

# 3. 构造面板数据 (T, N)
pivoted = {}
for col in features + [target]:
    pivoted[col] = df.pivot(index='time', columns='code', values=col)

X_dict = {f: pivoted[f].values for f in features}
y = pivoted[target].values  # (T, N)
# 确保dates为datetime类型

dates = pd.to_datetime(pivoted[target].index)
codes = pivoted[target].columns

# 4. 定义因子表达式（使用gp_framework函数）
# sub(div(volume, pre_close), volume)
factor1 = sub(div(X_dict['volume'], X_dict['pre_close']), X_dict['volume'])
# sub(div(volume, open), volume)
factor2 = sub(div(X_dict['volume'], X_dict['open']), X_dict['volume'])

factors = {'因子1': factor1, '因子2': factor2}

# 5. 分组回测函数
def group_backtest(factor, y, dates, codes, n_group=5, rebalance_freq=5,
                   slippage=0.001, open_commission=0.0003, close_commission=0.0003, close_tax=0.001):
    """
    增加手续费和滑点的分组回测
    正确的逻辑：只在调仓日使用未来收益，非调仓日净值不变
    返回净值序列和调仓详情DataFrame
    """
    T, N = factor.shape
    navs = {i: [1.0] for i in range(n_group)}
    last_group = np.full(N, -1)
    fee = (slippage + open_commission) + (slippage + close_commission + close_tax)
    rebalance_dates = list(range(0, T, rebalance_freq))
    rebalance_records = []
    for t in range(T):
        if t in rebalance_dates:
            this_factor = factor[t]
            valid = ~np.isnan(this_factor)
            ranks = np.full(N, np.nan)
            if np.sum(valid) >= n_group:
                ranks[valid] = pd.qcut(this_factor[valid], n_group, labels=False, duplicates='drop')
            last_group = ranks
            for g in range(n_group):
                navs[g][-1] *= (1 - fee)
            for g in range(n_group):
                idx = (last_group == g)
                if np.any(idx) and not np.all(np.isnan(y[t, idx])):
                    ret = np.nanmean(y[t, idx]) / 100
                    navs[g].append(navs[g][-1] * (1 + ret))
                    selected_codes = codes[idx]
                    selected_factors = this_factor[idx]
                    selected_returns = y[t, idx]
                    for i, (code, factor_val, return_val) in enumerate(zip(selected_codes, selected_factors, selected_returns)):
                        if not np.isnan(factor_val) and not np.isnan(return_val):
                            rebalance_records.append({
                                '调仓日期': dates[t].strftime('%Y-%m-%d'),
                                '分组': f'组{g+1}',
                                '股票代码': code,
                                '因子值': factor_val,
                                '未来5日收益率(%)': return_val,
                                '分组平均收益率(%)': ret * 100
                            })
                else:
                    navs[g].append(navs[g][-1])
        else:
            for g in range(n_group):
                navs[g].append(navs[g][-1])
    for g in navs:
        navs[g] = navs[g][1:]
    rebalance_df = pd.DataFrame(rebalance_records) if rebalance_records else None
    return navs, rebalance_df

# 6. 画图（每个因子单独一个子图，x轴只显示年和月）
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
colors = ['#b71c1c', '#f57c00', '#fbc02d', '#388e3c', '#1a237e']
rebalance_dfs = {}
for ax, (name, factor) in zip(axes, factors.items()):
    navs, rebalance_df = group_backtest(factor, y, dates, codes, n_group=5, rebalance_freq=5,
                         slippage=0.001, open_commission=0.0003, close_commission=0.0003, close_tax=0.001)
    if rebalance_df is not None:
        rebalance_dfs[name] = rebalance_df
    for g in range(5):
        ax.plot(dates, navs[g], label=f'{name} 组{g+1}', color=colors[g])
    ax.set_yscale('log')
    ax.set_ylabel('累计收益（对数轴）')
    ax.set_title(f'{name} 分组累计净值曲线（5组，5天调仓）')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=30)
axes[-1].set_xlabel('日期')
plt.tight_layout()
plt.show()
# 保存所有因子的调仓详情到一个Excel文件，不同sheet
if rebalance_dfs:
    with pd.ExcelWriter('rebalance_details_all.xlsx', engine='openpyxl') as writer:
        for name, df in rebalance_dfs.items():
            df.to_excel(writer, sheet_name=name, index=False)
    print(f"所有因子的调仓详情已保存到 rebalance_details_all.xlsx，sheet分别为：{list(rebalance_dfs.keys())}")
