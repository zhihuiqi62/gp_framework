# gp_framework

## 项目简介

本项目基于遗传规划（Genetic Programming, GP）方法，自动挖掘股票量化因子，并支持因子回测、分组分析、可视化等功能。适用于量化投资、金融数据挖掘、因子研究等场景。

## 主要功能

- 遗传规划自动生成和优化因子表达式
- 支持多种金融时间序列特征（如开盘价、收盘价、成交量等）
- 分组回测与绩效评估（如分组净值、IC、ICIR等）
- 支持大文件（如数据库、CSV）管理（已集成 Git LFS）
- 结果可视化（因子表达式树、分组净值曲线等）
- 支持多因子回测结果导出（Excel 多 sheet）

## 目录结构

```
.
├── gp_framework.py           # 遗传规划主框架与核心函数
├── gp_back.py                # 回测与分组分析脚本
├── gp_studay.py              # 因子训练与可视化脚本
├── gp.ipynb                  # Jupyter Notebook 示例
├── stocks_data.csv           # 股票历史数据（大文件，LFS管理）
├── stock_data.db             # 股票数据库（大文件，LFS管理）
├── rebalance_details_all.xlsx# 多因子分组回测结果
├── factor_X_graphviz.png     # 因子表达式树可视化图片
├── .gitattributes            # Git LFS 配置
├── README.md                 # 项目说明文档
└── ...
```

## 依赖环境

- Python 3.7+
- pandas
- numpy
- matplotlib
- openpyxl
- graphviz
- git-lfs（用于大文件管理）

安装依赖（推荐使用虚拟环境）：

```bash
pip install -r requirements.txt
```

## 快速开始

1. **克隆项目并初始化 LFS：**
   ```bash
   git clone https://github.com/zhihuiqi62/gp_framework.git
   cd gp_framework
   git lfs install
   ```

2. **准备数据：**
   - 将你的股票数据放入 `stocks_data.csv` 或 `stock_data.db`。

3. **运行因子训练与可视化：**
   ```bash
   python gp_studay.py
   ```
   - 自动生成因子表达式、IC/ICIR评估、表达式树可视化。

4. **运行回测与分组分析：**
   ```bash
   python gp_back.py
   ```
   - 输出分组净值曲线、分组回测明细（Excel）、图表等。

5. **Jupyter Notebook 交互式体验：**
   - 打开 `gp.ipynb`，可交互式运行和调试。

## 主要文件说明

- `gp_framework.py`：遗传规划算法实现，包含表达式树、算子、变异、交叉等。
- `gp_back.py`：分组回测、净值计算、结果导出与可视化。
- `gp_studay.py`：因子训练、IC/ICIR评估、表达式树可视化。
- `stocks_data.csv` / `stock_data.db`：股票历史数据（大文件，LFS管理）。
- `rebalance_details_all.xlsx`：多因子分组回测明细，按 sheet 分类。
- `factor_X_graphviz.png`：因子表达式树可视化图片。

## 注意事项

- **大文件管理**：本项目已集成 [Git LFS](https://git-lfs.com/)，请确保本地已安装并初始化 LFS，否则大文件无法正常 clone/pull。
- **数据隐私**：请勿上传包含敏感信息的数据文件到公开仓库。

## 贡献

欢迎提交 issue、pull request 或建议！

## License

[MIT](LICENSE)（如有需要可补充）

---

> 如需进一步定制 README 内容（如添加算法原理、示例代码、贡献者名单等），请告知你的具体需求！
