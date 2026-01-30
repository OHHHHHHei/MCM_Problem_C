# SMC-Inverse: 序贯蒙特卡洛粒子滤波隐状态反演系统

基于《与星共舞》(Dancing with the Stars) 淘汰数据，反演选手隐藏的粉丝投票份额。

## 快速开始

### 1. 环境配置

**方式一：使用批处理脚本 (推荐)**

双击运行 `setup_env.bat`，或在命令行：
```bash
setup_env.bat
```

**方式二：手动配置**

```bash
# 创建 conda 环境
conda create -n smc_inverse python=3.10 -y

# 激活环境
conda activate smc_inverse

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行模型

```bash
# 激活环境
conda activate smc_inverse

# 测试模式 (快速验证，~1分钟)
python main.py --test

# 完整运行 (所有赛季，~10分钟)
python main.py

# 运行指定赛季
python main.py --seasons 27 28 29

# 使用更多粒子 (更精确，更慢)
python main.py --particles 1000

# 查看所有选项
python main.py --help
```

### 3. 输出结果

运行后在 `output/` 目录生成：
- `analysis_report.txt` - 文本报告
- `analysis_results.json` - JSON格式分析结果
- `vote_share_estimates.json` - 各选手投票份额估计

## 项目结构

```
Code/
├── main.py               # 主程序入口
├── smc_inverse.py        # 粒子滤波核心实现
├── data_processor.py     # 数据加载与预处理
├── competition_rules.py  # 比赛规则模块
├── analysis.py           # 结果分析模块
├── requirements.txt      # Python依赖
├── setup_env.bat         # 环境配置脚本
└── 2026_MCM_Problem_C_Data.csv  # 数据集
```

## 模型参数

可通过命令行调整关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--particles` | 500 | 粒子数量 |
| `--kappa` | 0.3 | 表现冲击系数 |
| `--delta` | 1.5 | 冲击阈值 |
| `--rho` | 0.6 | 记忆衰减系数 |
| `--gamma` | 0.5 | 评审引导系数 |

## 核心算法

1. **状态空间模型**
   - 长期基准 (μ): 带冲击的随机游走
   - 短期动量 (x): 均值回归过程

2. **似然函数**
   - 基础淘汰似然: Sigmoid乘积
   - Judge Save似然: 两阶段条件概率

3. **重采样策略**
   - 系统重采样
   - ESS阈值触发

## 参考文献

建模思路详见 `C题.md`
