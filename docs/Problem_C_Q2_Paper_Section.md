# 5. 问题二：投票机制的结构性评估与反事实重演
# 5. Comparative Evaluation and Counterfactual Reenactment of Voting Architectures

## 5.1 问题重述与分析框架

题目要求我们对比两种历史使用的聚合规则——**排名积分法 (Rank Rule)** 与 **百分比得分法 (Percentage Rule)**，评估其对“争议性”结果（评委与粉丝分歧）的影响，并分析 **评委拯救机制 (Judge Save)** 的作用。

我们将分析分为三个逻辑层次：
1.  **宏观结构分析 (Macro-Structural Analysis)**: 此两种规则在所有赛季上的总体统计特性有何不同？哪种更偏向粉丝？
2.  **微观反事实推演 (Micro-Counterfactual Simulation)**: 针对特定的争议选手（如 Jerry Rice, Bobby Bones），如果交换规则，命运是否改变？
3.  **机制设计建议 (Mechanism Design)**: 基于上述分析，未来赛季应采用何种最优组合？

## 5.2 投票规则的数学模型与性质

### 5.2.1 排名积分法 (Rank Rule: Ordinal Aggregation)
应用于 S1-S2 及 S28+。该规则将原始分数转换为序数排名 ($R$) 后求和。
$$ S_{rank}^{(i)} = \text{Rank}(J_i) + \text{Rank}(\pi_i) $$
其中 $J_i$ 为评委分，$\pi_i$ 为粉丝投票份额。$\text{Rank}(\cdot)$ 为降序排名（第1名为1）。

**数学性质：民粹封顶 (Populist Capping)**
该规则是一个 **序数滤波器 (Ordinal Filter)**。它丢弃了 $\pi_i$ 的数值量级信息。无论一位选手获得 40% 还是 90% 的选票，其对最终结果的贡献被“截断”为固定值（Rank 1）。这在结构上限制了超级巨星的统治力。

### 5.2.2 百分比得分法 (Percentage Rule: Cardinal Aggregation)
应用于 S3-S27。该规则直接对归一化后的分值求和。
$$ S_{pct}^{(i)} = \frac{1}{2} \left( \frac{J_i}{\sum J_k} \right) + \frac{1}{2} \pi_i $$

**数学性质：量级保留 (Magnitude Preservation)**
由于粉丝投票 $\pi$ 通常服从长尾分布（高方差），而评委分 $J$ 服从正态分布（低方差），根据方差加成原理，总分 $S_{pct}$ 的波动性由**方差更大**的 $\pi$ 主导。该规则允许高人气的数值量级直接“淹没”评委分的差距。

---

## 5.3 宏观比较：谁更偏向粉丝？(Macro-Comparison)

我们利用 SMC 模型生成的全量数据，计算了两种规则下的 **粉丝一致性得分 (Fan Alignment Score, $A_F$)**，定义为最终结果与粉丝排名的秩相关系数。

**实证结果**：
- $A_F(R_{pct}) = 0.945$ (极高)
- $A_F(R_{rank}) = 0.687$ (中等)

**结论**：**百分比法 (Percentage Rule) 明显更偏向粉丝投票**。它保留了粉丝热情的原始量级，使得“人气”成为比赛的主导变量；而排名法 (Rank Rule) 则强制将“人气”与“技术”置于同等量纲（Rank vs Rank）上进行比较，实现了更强的制衡。

---

## 5.4 微观反事实推演：争议案例研究 (Case Studies)

我们构建了 **概率性反事实模拟器**，针对题目指定的四位争议选手进行了 $N=100$ 次全赛季重演。

**表 5-1: 争议选手在不同规则下的平均名次模拟**

| 选手 (赛季) | 争议类型 | **历史真实名次** | **Rank Rule (序数)** | **Pct Rule (基数)** | **结论：结果相同吗？** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bobby Bones** (S27) | 超级民粹 | 1.0 (Pct) | **2.8** (非冠军) | **1.9** (统治级) | **不同**。Rank 规则能遏制他。 |
| **Jerry Rice** (S2) | 中等民粹 | 2.0 (Rank) | 3.1 | **2.6** (更强) | **不同**。Pct 规则反而让他更强。 |
| **Bristol Palin** (S11)| 中等民粹 | 3.0 (Pct) | **5.8** (早淘汰) | **3.4** (强劲) | **不同**。Rank 规则能大幅压制。 |
| **Billy Ray Cyrus** (S4)| 中等民粹 | 5.0 (Pct) | **6.5** (早淘汰) | **5.6** (强劲) | **不同**。Rank 规则能压制。 |

### 5.4.1 分析：规则选择对结果有决定性影响
回答原题："选择合并方法是否会导致这些选手中的每一位得到相同的结果？"
**答案是否定的 (NO)**。

1.  **Bobby Bones 的奇点 (The Singularity)**：
    S27 使用 Pct 规则，Bobby 凭借约 40% 的粉丝票仓夺冠。模拟显示，若使用 Rank Rule，其巨大的票数优势被“封顶”为 1 分，无法抵消他垫底的评委排名。他将大概率在决赛前（平均名次 2.8）被淘汰。这证明 S28 改回 Rank Rule 是精准的修正。

2.  **S3 改革的历史反讽 (The Historical Irony)**：
    S2 Jerry Rice 获得亚军（Rank Rule），促使节目组在 S3 改用 Pct Rule 以“削弱粉丝权重”。然而模拟显示，这是一个**数学上的误判**。如果当年就用 Pct Rule，Jerry Rice 的名次会更好 (2.6 vs 3.1)。Rank Rule 其实是保护比赛免受 Jerry Rice 统治的最后一道防线，而 Pct Rule 拆除了这道防线。

---

## 5.5 评委拯救机制的影响 (Impact of Judge Save)

S28 引入的 **Judge Save** 允许评委在倒数两名 (Bottom 2) 中拯救一人。

**模拟发现**：
1.  **对中等争议有效 (Mitigation for Mid-Tier)**：对于 Jerry Rice, Bristol Palin 等“低分高人气”选手，Judge Save 是致命的。SMC 模拟显示，一旦他们掉入 Bottom 2，评委拯救对手的概率接近 100%。例如 Jerry Rice 的平均排名从 3.1 降至 3.6。
2.  **对超级民粹无效 (Ineffective for Outliers)**：对于 Bobby Bones，Judge Save 几乎从未触发。原因在于：在 Pct 规则下，他的总分极高，根本不会掉入 Bottom 2。**Judge Save 只是一个“熔断器”，只有当 Rank Rule 先发挥作用将选手拖入熔断区时，它才有效。**

---

## 5.6 综合推荐 (Recommendation)

基于上述分析，我们对未来赛季提出明确建议。

**推荐方案**：**Rank Rule + Judge Save**

**论证理由**：
1.  **结构性平衡 (Structural Balance)**：
    Pct Rule 是发散的（允许无限的人气溢出），而 Rank Rule 是收敛的（封顶效应）。作为一个舞蹈竞技节目，必须保留技术分对结果的“否决权”。Rank Rule 是唯一能防止 Bobby Bones 式“数据淹没”的机制。

2.  **容错机制 (Error Correction)**：
    单纯的 Rank Rule 可能会因平局处理或微小分差导致意外淘汰（技术流遗珠问题）。Judge Save 作为补充机制，能有效修正这些“统计噪音”，确保至少在 Bottom 2 阶段，专业标准具有最终解释权。

3.  **最优性证明**：
    构建效用函数 $U = 0.6 \cdot A_J + 0.4 \cdot \min(A_F, 0.75)$，计算显示 **Rank + Save** 组合得分最高 (0.773)，优于纯 Rank (0.769) 和 纯 Pct (0.769)。

**结论**：
节目组应坚持自 S28 以来采用的 **Rank Rule + Judge Save** 体系。这一组合在数学上构成了最稳健的防御纵深：Rank Rule 负责限制人气的**上限 (Ceiling)**，Judge Save 负责守住技术的**下限 (Floor)**。
