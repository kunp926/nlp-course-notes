# Word2Vec 中的 PMI、矩阵分解与低秩假设

> 理论笔记：理解 `king - man + woman = queen` 背后的数学原理。
>
> 主要参考：Piotr Migdał, [*king - man + woman is queen; but why?*](https://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html)（2017）；以及 Omer Levy & Yoav Goldberg, *Neural Word Embeddings as Implicit Matrix Factorization*（NIPS 2014）。

## 0. 核心问题

**word2vec** 把单词映射成向量，使得语义相近的词在向量空间中距离接近，并且支持向量算术来做类比，最著名的例子是：

$$
\vec v_{king} - \vec v_{man} + \vec v_{woman} \approx \vec v_{queen}
$$

这份笔记要回答几个层层递进的问题：

1. 词向量的向量形式是怎么来的？
2. 为什么要用「词向量点积 ≈ PMI」来建模？
3. PMI 矩阵是什么？它和矩阵怎么联系起来？
4. 什么是矩阵分解？
5. 什么是矩阵的「秩」和「低秩假设」？

---

## 1. 分布假说（Distributional Hypothesis）

word2vec 的思想源头是**分布假说**，语言学家 J. R. Firth 的名言概括为：

> **a word is characterized by the company it keeps.**
> （一个词的含义，由它周围出现的词所决定。）

例如句子 *"A small, fluffy roosety climbed a tree."* 中，即使 `roosety` 是个生造词，我们也能从它的上下文（small、fluffy、climbed a tree）猜测它大概是某种类似松鼠的动物。

### 用条件概率形式化

设 $P(a \mid b)$ 为条件概率：给定词 $b$，在它附近（比如窗口距离不超过 2 个词）出现词 $a$ 的概率。

我们声称两个词 $a$ 和 $b$ **相似**，当且仅当对每一个词 $w$ 都有：

$$
P(w \mid a) = P(w \mid b)
$$

也就是说：无论中心词是 $a$ 还是 $b$，其他所有词在它周围出现的频率都相同。这正是"两个词可以互换"的概率表达。

---

## 2. 点互信息 PMI

### 2.1 定义

**点互信息（Pointwise Mutual Information, PMI）** 衡量两个词"有多喜欢一起出现"：

$$
\operatorname{PMI}(a, b)
= \log \frac{P(a, b)}{P(a)\,P(b)}
= \log \frac{P(a \mid b)}{P(a)}
$$

它的直观含义是：**一对词共同出现的概率，比它们"随机独立"情况下高出多少。**

- $\operatorname{PMI}(a,b) > 0$：$a$、$b$ 一起出现的频率高于随机（如 `dog` 和 `bark`）；
- $\operatorname{PMI}(a,b) \approx 0$：两词近似独立、没有关联；
- $\operatorname{PMI}(a,b) < 0$：两词很少一起出现。

取对数是为了方便处理频率相差好几个数量级的词。

### 2.2 从条件概率相等到 PMI 相等

由相似条件 $P(w \mid a) = P(w \mid b)$，两边同除以 $P(w)$：

$$
\frac{P(w, a)}{P(w)\,P(a)} = \frac{P(w, b)}{P(w)\,P(b)}
$$

两边取对数，恰好就是 PMI 的定义：

$$
\boxed{\operatorname{PMI}(w, a) = \operatorname{PMI}(w, b)}
$$

### 2.3 关键建模假设：PMI ≈ 向量点积

word2vec 假设每个词 $i$ 对应一个 $d$ 维向量 $\vec v_i$（$d$ 通常为 50–300），并且：

$$
\boxed{\operatorname{PMI}(a, b) \approx \vec v_a \cdot \vec v_b}
$$

代入上式并利用点积的分配律：

$$
\vec v_w \cdot \vec v_a = \vec v_w \cdot \vec v_b
$$

$$
\Longrightarrow \quad \boxed{\vec v_w \cdot (\vec v_a - \vec v_b) = 0}
$$

**几何含义**：向量 $\vec v_w$ 与向量 $\vec v_a - \vec v_b$ 互相正交（垂直）。若该式对任意 $\vec v_w$ 都成立，则 $\vec v_a = \vec v_b$，即两词向量相同。实际中只是近似相等，于是表现为"两词在空间中很接近"。

---

## 3. 为什么用「点积 ≈ PMI」建模？动机是什么？

核心一句话：**PMI 衡量两个词的关联强度，而点积天然衡量两个向量的对齐程度；我们希望把高维共现统计压缩到低维向量空间，于是用点积去拟合 PMI。**

### 3.1 PMI 是词对的关联分数，但矩阵太大

词汇表大小记为 $V$。所有词对的 PMI 构成一个 $V \times V$ 的巨大矩阵。当 $V = 10^5$ 时，需要存储 $10^{10}$ 个数值，约需数 GB 甚至更多，无法直接处理，因此需要降维压缩。

### 3.2 点积正好表达"关联强度"

两个向量的点积：

$$
\vec v_a \cdot \vec v_b = \sum_{i=1}^{d} v_{a,i}\,v_{b,i}
$$

- 两向量方向越接近，点积越大；方向相反，点积越小；
- 这正好对应"关联强的词向量靠近、点积大；关联弱的词点积小"。

### 3.3 为什么偏偏选点积（线性函数）

1. **计算极快**：点积只是乘法求和，便于 GPU 批量并行和最近邻检索；非线性函数开销大得多。
2. **天然支持向量加减法**：这是 `king - man + woman = queen` 词类比成立的前提。若用非线性相似度，就没有漂亮的向量算术。
3. **契合分布假说**：把上下文共现统计（PMI）编码成向量方向，向量越近代表上下文分布越相似。

### 3.4 重要提醒

> **「点积 ≈ PMI」是一个近似的建模假设，不是自然规律。**

word2vec 并不是先算好 PMI 再去分解，而是通过神经网络 + 负采样训练，**间接**拟合这个目标。PMI 矩阵的说法是一种理论解释视角，而非代码实现流程。

---

## 4. PMI 矩阵是什么？

### 4.1 定义

构造一个方阵 $\boldsymbol M$，行和列都对应词汇表中的单词：

- 行 $i$ 对应词 $w_i$，列 $j$ 对应词 $w_j$；
- 矩阵元素：

$$
\boldsymbol M_{i,j} = \operatorname{PMI}(w_i, w_j)
$$

即**每一个单元格存放一对词的 PMI 值**。

### 4.2 小例子

假设词表只有 `king, man, woman` 三个词：

$$
\boldsymbol M =
\begin{bmatrix}
\operatorname{PMI}(king,king) & \operatorname{PMI}(king,man) & \operatorname{PMI}(king,woman)\\
\operatorname{PMI}(man,king) & \operatorname{PMI}(man,man) & \operatorname{PMI}(man,woman)\\
\operatorname{PMI}(woman,king) & \operatorname{PMI}(woman,man) & \operatorname{PMI}(woman,woman)
\end{bmatrix}
$$

若 `king` 和 `man` 经常一起出现，对应单元格的值就大；若两词从不共现，PMI 为负，对应单元格就是负数。

### 4.3 与词向量的联系：矩阵分解

我们希望把大矩阵压缩为两个低维矩阵的乘积：

$$
\boldsymbol M_{V \times V} \approx \boldsymbol V_{V \times d}\,\boldsymbol V_{d \times V}^{\top}
$$

- $\boldsymbol V$ 是 $V \times d$ 矩阵，它的**第 $i$ 行就是词 $w_i$ 的词向量** $\vec v_{w_i}$；
- 由矩阵乘法：$(\boldsymbol V \boldsymbol V^\top)_{i,j} = \vec v_{w_i} \cdot \vec v_{w_j}$。

对比 PMI 矩阵定义：

$$
\boldsymbol M_{i,j} = \operatorname{PMI}(w_i, w_j) \approx \vec v_{w_i} \cdot \vec v_{w_j}
$$

**矩阵分解的目标，就是让两个词向量的点积去近似这对词原始的 PMI 值。**

### 4.4 修正：实践中用 PPMI

如果一对词完全不共现，$P(a,b)=0$，则 $\operatorname{PMI} = -\infty$，无法放入矩阵。因此实践中使用 **PPMI（Positive PMI，正点互信息）**，把负值截断为 0：

$$
\operatorname{PPMI}(a, b) = \max\big(\operatorname{PMI}(a, b),\ 0\big)
$$

Levy & Goldberg（2014）证明：**Skip-Gram + 负采样（SGNS）等价于隐式地对 PPMI 矩阵做矩阵分解。**

---

## 5. 什么是矩阵分解？

### 5.1 定义

> **矩阵分解（Matrix Factorization）就是把一个大矩阵 $\boldsymbol M$，拆成两个（或多个）更小矩阵的乘积。**

在词向量场景：

$$
\boldsymbol M_{V \times V} \approx \boldsymbol V_{V \times d}\,\boldsymbol V_{d \times V}^{\top}, \qquad d \ll V
$$

### 5.2 核心性质

1. **降维、压缩存储**：存储量从 $O(V^2)$ 降到 $O(Vd)$。$V = 10^5$、$d = 200$ 时，从 $10^{10}$ 个数值降到 $2\times10^7$，缩小数百倍。
2. **低秩假设**：能这样分解的前提是原始矩阵近似低秩（见第 6 节）。
3. **内积 = 相似度**：分解后 $\boldsymbol M_{ij} \approx \vec v_i \cdot \vec v_j$，点积越大代表关联越强，这也是用余弦相似度衡量词近义的原因。
4. **线性叠加性**：线性内积满足 $\vec v_w \cdot (\vec v_a - \vec v_b) = \vec v_w \cdot \vec v_a - \vec v_w \cdot \vec v_b$，词类比正源于此。
5. **有损近似**：$\approx$ 而非 $=$，目标是让整体误差最小，保留全局模式、舍弃噪声。

### 5.3 数字小例子

设词表为 `king, man, woman`，$V=3$，向量维度 $d=2$。人为构造 PMI 矩阵：

$$
\boldsymbol M =
\begin{bmatrix}
2.0 & 1.8 & 0.3\\
1.8 & 2.0 & 0.4\\
0.3 & 0.4 & 2.0
\end{bmatrix}
$$

假设算法学到的词向量矩阵：

$$
\boldsymbol V =
\begin{bmatrix}
1.4 & 0.8\\
1.3 & 0.6\\
0.1 & 1.4
\end{bmatrix}
$$

- $\vec v_{king} = [1.4, 0.8]$，$\vec v_{man} = [1.3, 0.6]$，$\vec v_{woman} = [0.1, 1.4]$；
- 计算 $\vec v_{king}\cdot\vec v_{man} = 1.4\times1.3 + 0.8\times0.6 = 2.30$，去近似原矩阵中的 $1.8$，存在误差——这就是有损压缩。

### 5.4 另一个经典例子：推荐系统

用户-电影评分矩阵中，行是用户、列是电影、单元格是评分（大量为空）。矩阵分解为：

$$
\boldsymbol R \approx \boldsymbol U\,\boldsymbol P^\top
$$

- $\boldsymbol U$ 每行是用户偏好向量，$\boldsymbol P$ 每行是电影特征向量；
- 用户 $u$ 对电影 $m$ 的预测评分 $\approx \vec u_u \cdot \vec p_m$，从而预测没看过的电影评分。

### 5.5 SVD vs word2vec 的隐式分解

- **SVD（奇异值分解）**：显式分解，需要输入完整矩阵 $\boldsymbol M$。但 PMI 矩阵太大、很多元素缺失，难以直接做 SVD。
- **word2vec（SGNS）**：隐式分解，不需要构造完整矩阵，通过神经网络在语料上训练，间接学到等价于 PPMI 分解的结果。

---

## 6. 矩阵的秩与低秩假设

### 6.1 什么是矩阵的秩

> **矩阵的秩（rank）= 矩阵中"真正独立、不冗余的信息维度数量"。**

如果某一行能完全由其他行做线性组合得到，它就是冗余的；秩就是独立行的最大数量。

**例 1（秩为 1）**：

$$
\boldsymbol A = \begin{bmatrix} 1 & 2 & 3\\ 2 & 4 & 6\end{bmatrix}
$$

第二行 = 第一行 × 2，没有新信息，独立行只有 1 条，故 $\operatorname{rank}(\boldsymbol A) = 1$。

**例 2（满秩）**：

$$
\boldsymbol B = \begin{bmatrix} 1 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & 1\end{bmatrix}
$$

三行谁都不能由另外两行组合得到，秩为 3（等于矩阵尺寸），即**满秩**。

**生活类比**：记录一批人的身高、体重、年龄。若身高完全能由体重推算、年龄也重复，那本质只用 1 个变量就能描述，秩为 1；若三者互相独立、谁也预测不了谁，秩为 3。

### 6.2 低秩 vs 满秩

- **低秩矩阵**：信息有强关联、存在少量隐藏因子。例如动物的体长、体重、食量、活动量都由"体型大小"这一隐藏因子主导，4 列虽多但秩约为 1，可以高度压缩。
- **满秩矩阵**：信息完全独立、毫无规律。例如随机彩票号码，没有任何隐藏因子能从一列预测另一列，无法压缩。

### 6.3 套回 PMI 矩阵

PMI 矩阵是 $V \times V$（如 $V = 10000$）的巨大方阵。

**若矩阵近似低秩（我们希望的情况）**：词的共现不是随机的，背后由少数几个隐语义因子控制，例如：

- 性别（男 / 女）
- 物种（人 / 猫 / 狗）
- 年龄（成年 / 幼崽）
- 职业……

总共只需几十个隐因子（$d = 50 \sim 300$）就能近似表达全部共现规律。

> 注意：模型并不知道这些维度叫"性别""年龄"，它们只是数学上的坐标轴，语义解释是人赋予的。

于是 $\boldsymbol M \approx \boldsymbol V\boldsymbol V^\top$（$d \ll V$）成立，矩阵分解效果好，词向量能学到类比关系。

**若矩阵是满秩**：词的共现完全随机、不存在稳定语义规律，那么用很小的 $d$ 维向量无法还原矩阵，强行低秩近似误差巨大，词向量学不到任何语义，`king - man + woman` 也算不出 `queen`。

### 6.4 直观比喻

- PMI 矩阵 = 一张巨大的世界地图；矩阵分解 = 用很少的参数画近似地图。
- **低秩**：地理有内在规律（海拔、纬度、降水），几十个参数就能画出不错的近似地图；
- **满秩**：每个像素都是随机噪声，再少的参数也无法还原。

### 6.5 注意：是「近似低秩」

真实 PMI 矩阵并非严格数学意义上的低秩，而是**近似低秩**：语料中含噪声、特例、多义词，所以只能近似拟合、存在误差。这也是词类比不总成立、甚至会学出社会偏见的原因。

---

## 7. 局限性与偏见

1. **线性假设只是近似**：语义并非严格的线性空间，词类比会出错。
2. **学到的是语料统计规律，而非客观语义**：会复现文本中的社会偏见，例如：

$$
\vec v_{doctor} - \vec v_{man} + \vec v_{woman} \approx \vec v_{nurse}
$$

3. **只利用共现，不理解句法与多义**：同一个词的不同含义被混在同一个向量里。
4. **结果依赖训练数据**：换一个语料，向量关系就会不同。

---

## 8. 关键结论速查

| 概念 | 一句话 |
|---|---|
| 分布假说 | 词由它周围的"同伴"定义 |
| PMI | 词对共同出现比随机独立高出多少（取对数） |
| 建模假设 | $\operatorname{PMI}(a,b) \approx \vec v_a \cdot \vec v_b$ |
| PMI 矩阵 | 行、列都是词，单元格存词对 PMI 的 $V\times V$ 方阵 |
| PPMI | 把 PMI 负值截断为 0，实践中实际使用 |
| 矩阵分解 | 大矩阵拆成小矩阵乘积 $\boldsymbol M \approx \boldsymbol V\boldsymbol V^\top$ |
| 秩 | 矩阵中独立、不冗余的信息维度数量 |
| 低秩假设 | 词共现由少数隐语义因子控制，故可用 $d\ll V$ 的向量近似 |
| SGNS | 等价于隐式分解 PPMI 矩阵 |

**完整逻辑链**：

$$
\underbrace{P(w\mid a)=P(w\mid b)}_{\text{分布假说}}
\;\Longrightarrow\;
\underbrace{\operatorname{PMI}(w,a)=\operatorname{PMI}(w,b)}_{\text{除以 }P(w)\text{ 取对数}}
\;\Longrightarrow\;
\underbrace{\vec v_w\cdot\vec v_a=\vec v_w\cdot\vec v_b}_{\text{点积近似 PMI}}
\;\Longrightarrow\;
\underbrace{\vec v_w\cdot(\vec v_a-\vec v_b)=0}_{\text{词相近 / 可类比}}
$$

---

## 参考资料

- Piotr Migdał, [king - man + woman is queen; but why?](https://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html)
- Omer Levy, Yoav Goldberg, [Neural Word Embeddings as Implicit Matrix Factorization](https://papers.nips.cc/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html), NIPS 2014
- Jeffrey Pennington, Richard Socher, Christopher D. Manning, [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf), EMNLP 2014
- Tomas Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781), 2013
- Daniel Jurafsky, James H. Martin, *Speech and Language Processing*, Chapter 15–16
- Chris Moody, [A Word is Worth a Thousand Vectors](https://multithreaded.stitchfix.com/blog/2015/03/11/word-is-worth-a-thousand-vectors/)
