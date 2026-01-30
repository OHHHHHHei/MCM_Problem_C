我们将该模型命名为 **“SMC-Inverse：基于序贯蒙特卡洛的隐状态反演系统” (Sequential Monte Carlo Inverse Latent State System)**。该方案不仅仅是对数据的拟合，而是构建了一个**部分观测的时变动力学系统 (Partially Observed Time-Varying Dynamical System)**，旨在从仅有的“淘汰结果”这一删失数据中，还原不可见的“粉丝投票份额”。

以下是完整的数学建模阐述：

---

# 1. 系统定义与符号体系

我们将《与星共舞》(DWTS) 视为一个随机过程。

- **观察者 (Observer)**：我们（建模者），只能看到每周的评审分 $J_{i,t}$ 和淘汰结果 $\mathcal{E}_t$。
    
- **隐变量 (Latent Variable)**：选手 $i$ 在第 $t$ 周获得的真实粉丝投票份额 $\pi_{i,t}$。
    
- **控制变量 (Control Input)**：比赛规则（Rank制 / Percent制 / Judge Save机制）。
    

我们定义状态向量 $\mathbf{\Theta}_{i,t}$ 来描述选手的潜在竞争力。

---

# 2. 状态空间构建 (State-Space Formulation)

为了捕捉粉丝投票的动态特性，我们将选手的“人气”解耦为两个分量：**长期基准 (Baseline Popularity)** 和 **短期动量 (Short-term Momentum)**。

### 2.1 初始状态构建：防止数据泄露的冷启动

为了避免使用未来数据（例如用第10季的统计规律去预测第1季），我们严格采用 **Leave-One-Season-Out (LOSO)** 策略。

对于第 $s$ 季的选手 $i$，其初始人气基准 $\mu_{i,0}$ 由元数据决定：

$$\mu_{i,0} \sim \mathcal{N}(\mathbf{\beta}_{-s}^T \mathbf{Z}_i, \sigma_0^2)$$

- $\mathbf{Z}_i$：特征向量（年龄、行业类别One-hot、搭档历史胜率）。
    
- $\mathbf{\beta}_{-s}$：仅使用**除 $s$ 季以外**的历史数据回归得到的系数向量。
    

### 2.2 状态演化方程 (State Evolution)

我们构建如下随机微分方程的离散形式：

**A. 长期基准 ($\mu_{i,t}$) —— 带有“表现冲击”的随机游走**

选手的基本盘通常是稳定的，但极端的评审表现（如满分或极低分）会改变路人缘。

$$\mu_{i,t} = \mu_{i,t-1} + \kappa \cdot \mathbb{I}(|\tilde{J}_{i,t}| > \delta) \cdot \tilde{J}_{i,t} + \eta_{i,t}, \quad \eta_{i,t} \sim \mathcal{N}(0, \sigma_{\mu}^2)$$

- $\tilde{J}_{i,t}$：第 $t$ 周的标准化评审分（Z-score）。
    
- $\mathbb{I}(\cdot)$：指示函数，表示仅当表现偏离均值超过阈值 $\delta$ 时，才会永久性改变长期基准。
    

**B. 短期动量 ($x_{i,t}$) —— 均值回归过程**

这是产生当周投票的核心驱动力。

$$x_{i,t} = \rho x_{i,t-1} + (1-\rho) \underbrace{(\mu_{i,t} + \gamma \tilde{J}_{i,t})}_{\text{Driving Force}} + \epsilon_{i,t}, \quad \epsilon_{i,t} \sim \mathcal{N}(0, \sigma_x^2)$$

- $\rho \in [0, 1)$：记忆衰减系数。$\rho$ 越大，粉丝粘性越强；$\rho$ 越小，受当周表现影响越大。
    
- $\gamma$：评审引导系数，量化评审分数对观众投票的引导作用。
    

### 2.3 观测映射 (Observation Mapping)

将无界的动量 $x$ 映射为归一化的投票份额 $\pi$（Softmax 变换）：

$$\pi_{i,t} = \frac{\exp(x_{i,t})}{\sum_{k \in \mathcal{A}_t} \exp(x_{k,t})}$$

其中 $\mathcal{A}_t$ 是第 $t$ 周仍在场的选手集合。

---

# 3. 规则复刻与生存得分 (Unified Survival Mechanics)

这是模型与现实规则对接的关键接口。我们需要计算**生存得分 (Survival Score, $S_{i,t}$)**。

### 3.1 修正：滚存机制的“有效份额” (Carry-Over Logic)

针对“无淘汰周”的分数滚存，单纯的 $\pi$ 累加在数学上是不严谨的。我们定义**有效累积份额 (Effective Cumulative Share, $\hat{\pi}$)**：

$$\hat{\pi}_{i,t} = \begin{cases} \pi_{i,t} & \text{若 } t-1 \text{ 有淘汰} \\ \frac{1 \cdot \hat{\pi}_{i,t-1} + \lambda_{decay} \cdot \pi_{i,t}}{1 + \lambda_{decay}} & \text{若 } t-1 \text{ 无淘汰 (滚存)} \end{cases}$$

- $\lambda_{decay}$：时间贴现因子。尽管规则是分数累加，但考虑到观众对上周记忆的模糊，早期投票的实际权重在心理层面上会衰减。**敏感性分析点：** 需测试 $\lambda_{decay}=1$（完全累加）与 $\lambda_{decay}<1$ 的区别。
    

### 3.2 赛制公式 (引入平局微扰)

**情形 A: Percentage Rule (S3-S27)**

直接相加两者占比：

$$S_{i,t}^{\text{Pct}} = \frac{J_{i,t}}{\sum_{k} J_{k,t}} + \hat{\pi}_{i,t}$$

**情形 B: Rank Rule (S1-2, S28+) —— [逻辑补丁]**

Rank 制是离散的，且存在平局（Tie）。现实中，**总排名相同时，通常粉丝票多者晋级**。我们在数学上通过引入微扰项 $\xi$ 来实现这一逻辑，避免使用复杂的 if-else 逻辑：

$$S_{i,t}^{\text{Rank}} = - \left[ \text{Rank}(J_{i,t}) + \text{Rank}(\hat{\pi}_{i,t}) \right] + \xi \cdot \hat{\pi}_{i,t}$$

- $\text{Rank}(\cdot)$：数值越小排名越高（1为最好）。
    
- 取负号：使得 $S$ 越大越安全。
    
- $\xi \approx 10^{-4}$：微扰项。当 $\text{Sum of Ranks}$ 相同时，$\hat{\pi}$ 更大的人 $S$ 会略微大一点点，从而在排序中胜出。
    

---

# 4. 概率反演与似然函数 (Inference & Likelihood)

我们使用 **粒子滤波 (Particle Filter)** 求解。对于第 $k$ 个粒子，我们需要计算其权重 $w_t^{(k)}$。

### 4.1 基础淘汰似然 (Standard Elimination Likelihood)

适用于 S1-S27 以及 S28+ 的非 Bottom-2 阶段。

设实际被淘汰者为 $e$，实际幸存者集合为 $\mathcal{S}$。模型预测 $e$ 的生存分应该低于所有幸存者。

$$\mathcal{L}_{\text{Base}}^{(k)} = P\left( S_{e,t}^{(k)} < \min_{j \in \mathcal{S}} S_{j,t}^{(k)} \right) \approx \prod_{j \in \mathcal{S}} \sigma \left( \alpha \cdot (S_{j,t}^{(k)} - S_{e,t}^{(k)}) \right)$$

- $\sigma(\cdot)$：Sigmoid 函数。
    
- $\alpha$：判别力参数。
    

### 4.2 评审拯救机制 (Judge Save Logic, S28+)

这是 S28+ 的核心复杂点。它是一个**两阶段条件概率事件**。

观测数据：淘汰了 $e$，且（如果数据可用）Bottom 2 是 $\{e, s\}$。

**似然函数分解：**

$$\mathcal{L}_{\text{Save}}^{(k)} = \underbrace{P(\text{Bottom 2} = \{e, s\} | \mathbf{\Theta}_t^{(k)})}_{\text{Step 1: 观众决定的危险区}} \times \underbrace{P(\text{Judge Chooses } s \text{ over } e | \text{Tech-Score})}_{\text{Step 2: 评委决定的生死}}$$

- **Step 1 (Bottom 2)**: 粒子的 $S$ 值必须使得 $e$ 和 $s$ 排在最后两名。
    
- **Step 2 (Judge Decision)**: 评委倾向于拯救技术更好的。我们假设评委只看“累计技术分”。
    
    $$P(\text{Save } s) = \sigma \left( \beta_{judge} \cdot (\mathcal{J}_{cum, s, t} - \mathcal{J}_{cum, e, t}) \right)$$
    
    - **关键逻辑**：此处**显式剔除** $\pi$（粉丝投票），仅使用 $\mathcal{J}_{cum}$（累计标准化评审分）。这体现了“评委保护专业性”的假设。
        

---

# 5. 模型验证与输出指标

为了回应“批判性思考”，我们必须严格区分“预测力”和“解释力”。

### 5.1 预测性验证 (One-Step-Ahead Validation)

在粒子滤波进行**重采样 (Resampling) 之前**计算。

- **指标**：**Top-N 危险区命中率 (Hit Rate @ Bottom N)**。
    
- **定义**：利用 $t-1$ 的后验分布预测 $t$ 时刻，计算有多少比例的粒子预测“实际被淘汰者”落入了 Bottom N。
    
- **意义**：这是衡量模型真实预测能力的“硬指标”，证明模型不是在“事后诸葛亮”。
    

### 5.2 解释性验证 (Posterior Consistency)

在重采样**之后**计算。

- **指标**：**后验对数似然均值 (Mean Log-Likelihood)**。
    
- **意义**：用于识别“冷门事件”或“争议事件”。如果某周的后验似然极低，说明实际上发生了一个小概率事件（例如：尽管评审分高且模型预测人气高，但该选手依然被淘汰），这正是题目要求的“Controversy”量化依据。
    

### 5.3 赛制公平性量化 (Flip Rate & Populist Bias)

利用反演出的 $\hat{\pi}_{i,t}$，我们在虚拟环境中切换赛制（Rank $\leftrightarrow$ Percent），计算：

1. **Flip Rate (翻盘率)**：淘汰结果发生改变的频率。
    
2. **Populist Bias (民粹偏差)**：
    
    $$\text{Bias} = \mathbb{E}[\text{Rank}(\pi_{survivors}) - \text{Rank}(J_{survivors})]$$
    
    该指标用于判断哪种赛制更倾向于保护“高人气低技术”的选手（即 Bobby Bones 效应）。
    

---

### 总结：该方案的逻辑闭环

1. **输入端**：通过 LOSO 回归初始化，杜绝数据泄露。
    
2. **状态端**：引入“冲击”项和“均值回归”，兼顾长期稳定与短期波动。
    
3. **规则端**：通过“有效份额”修正滚存逻辑，通过 $\xi$ 微扰修正 Rank 平局，数学表达严谨。
    
4. **推断端**：将 S28+ 拆解为“观众入围”+“评委裁决”两步，精准还原博弈过程。
    
5. **输出端**：区分先验预测与后验解释，提供了客观评价模型可信度的标准。