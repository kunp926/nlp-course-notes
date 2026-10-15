# NLP Course Notes

自然语言处理课程（CSC6052 / 5051 / MDS5110, Natural Language Processing）的课堂教程与第一次作业。所有代码均以 Jupyter Notebook 形式给出，可在 Google Colab（建议 T4 GPU）或本地运行。

## 目录结构

```
.
├── pytorch/
│   └── PyTorch_tutorial.ipynb            # PyTorch 入门：autograd 与线性回归
├── feedforward-network/
│   └── NLP_f26_pytorch.ipynb            # 前馈神经网络：手写实现 + PyTorch Module
├── word-embeddings/
│   ├── WordEmbedding_Tutorials.ipynb     # 词向量：Skip-Gram / GloVe / FastText
│   └── word2vec_pmi_matrix_factorization.md  # 理论笔记：PMI / 矩阵分解 / 低秩假设
└── assignment/
    └── MDS5110_Assignment1.ipynb         # Assignment 1：CBoW 词向量训练与探索
```

## 教程内容

### 1. PyTorch 入门 — `pytorch/PyTorch_tutorial.ipynb`

中文版 PyTorch 快速上手，核心概念：

- **自动微分（Autograd）与计算图**：`requires_grad`、`backward()`、`grad_fn`，动态图与静态图的区别。
- **张量（Tensor）**：与 numpy `ndarray` 的对应关系，`.data` / `.grad` / `.shape` / `.device` 等元信息。
- **`no_grad` 上下文**：在不需要梯度的场景下临时关闭反向追踪。
- **热身任务**：用纯 numpy 实现线性回归（前向 → MSE 损失 → 解析梯度 → 参数更新），再用 PyTorch 的 `loss.backward()` 与手动 `w.data -= lr * w.grad` 复现同一过程。

### 2. 前馈神经网络 — `feedforward-network/NLP_f26_pytorch.ipynb`

从"裸写权重"到"继承 `nn.Module`"的完整演进：

- **Naive 实现**：手动定义 `W1/b1 … W4/b4`，自定义 `sigmoid`，逐层 `matmul` 完成前向，`J.backward()` 后观察各参数梯度，再用 `W <- W - lr * W.grad` 或 `torch.optim.SGD` 一步更新。
- **PyTorch 实现**：用 `torch.nn.Linear` 替代手写矩阵，继承 `torch.nn.Module` 组织网络，`optimizer.zero_grad()` 清空梯度累加。
- **numpy ↔ torch 互转**：`torch.from_numpy` / `.numpy()`。
- **数据管道**：`torchvision.datasets.CIFAR10`、`DataLoader`、自定义 `Dataset`（`__getitem__` / `__len__`）。

### 3. 词向量教程 — `word-embeddings/WordEmbedding_Tutorials.ipynb`

以 Quora Question Pairs 数据集为语料，完整走一遍词向量的训练、评估与横向对比：

- **数据准备**：`nltk.word_tokenize` 分词、大小写归一、构建 `word2index` / `index2word`、按窗口收集中心词-上下文对。
- **Skip-Gram 实现**：批生成器（中心词重复 `num_skips` 次、随机采样上下文）、`nn.Embedding + nn.Linear` 模型、CrossEntropyLoss + Adam 训练曲线。
- **效果检验**：基于余弦相似度的 `most_similar` 查询（如 `my → our / your / his …`），以及 t-SNE + Bokeh 交互式散点可视化。
- **GloVe 与 FastText**：通过 `gensim.downloader` 加载 `glove-wiki-gigaword-300` 与 `fasttext-wiki-news-subwords-300`。
- **三方对比**：
  - 低频词 / OOV 词（`serendipity`）：Skip-Gram 直接返回 `None`，FastText 借助子词（subword n-gram）仍能给出向量。
  - 多义词（`king`）与语义关系（`king` ↔ `queen` 的余弦相似度）。
  - PCA 二维散点对比 `king / queen / man / woman / prince / princess` 的聚簇形态。
  - 内存与计算开销对比表（Word2Vec 最省、GloVe 居中、FastText 因子词开销最大）。

**配套理论笔记** — `word-embeddings/word2vec_pmi_matrix_factorization.md`

从数学角度解释 `king - man + woman = queen` 为什么成立，层层递进：

- **分布假说**：词由它周围的"同伴"定义，用条件概率 $P(w\mid a)=P(w\mid b)$ 形式化。
- **点互信息 PMI**：衡量词对共现相对随机独立的强度，以及如何从条件概率相等推导出 $\operatorname{PMI}(w,a)=\operatorname{PMI}(w,b)$。
- **核心建模假设**：为什么用「词向量点积 ≈ PMI」（$\operatorname{PMI}(a,b)\approx\vec v_a\cdot\vec v_b$），以及它的动机。
- **PMI 矩阵与 PPMI**：词对 PMI 构成的 $V\times V$ 方阵，以及实践中对负值截断的 PPMI。
- **矩阵分解**：$\boldsymbol M\approx\boldsymbol V\boldsymbol V^\top$ 的性质、数字例子与推荐系统类比。
- **秩与低秩假设**：用通俗例子解释矩阵的秩、低秩 vs 满秩，说明词共现由少数隐语义因子控制。
- **局限性**：线性假设只是近似，词向量会复现语料中的社会偏见（如 `doctor - man + woman ≈ nurse`）。

### 4. Assignment 1 — `assignment/MDS5110_Assignment1.ipynb`

课程第一次作业（满分 10 分），在 Skip-Gram 教程基础上要求学生补全三处 `[Task]`：

- **Task 1（5 分）训练 CBoW 词向量**：补全 `make_cbow_batches_iter` 批生成器、`CBoWModel.forward`（上下文 embedding 取平均后接输出层）、以及训练循环。
- **Task 2（3 分）探索词向量**：用 `sklearn.decomposition.TruncatedSVD` 把词向量降到 2 维，对石油领域词簇（`barrels / bpd / ecuador / oil / venezuela …`）做聚类分析，并讨论多义词现象。
- **Task 3（2 分）应用词向量**：利用训练得到的词向量完成下游任务（类比 / 相似度）。

作业在 Google Colab 上完成，最终导出 `.ipynb` 上传至 Blackboard。

## 环境依赖

- Python 3.10+
- `torch` / `torchvision`
- `numpy` / `pandas` / `matplotlib` / `scikit-learn` / `scipy`
- `nltk`（需 `punkt` / `punkt_tab` tokenizer）
- `gensim`（用于下载 GloVe / FastText 预训练向量）
- `bokeh`（交互式散点图）
- `tqdm`

快速安装：

```bash
pip install -q "nltk>=3.9.2,<4" "bokeh>=3.8,<3.10" gensim torch torchvision \
    numpy pandas matplotlib scikit-learn tqdm
```

词向量教程会自动从 Google Drive 拉取 Quora `train.csv`（如链接失效，可从 [Kaggle Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs/data) 下载并放到 notebook 工作目录）。

## 运行方式

- **Google Colab**：上传 `.ipynb` 后 `文件 → 在云端硬盘中保存副本`，选择 GPU 运行时，自上而下执行单元格。
- **本地**：`jupyter lab` 打开对应 notebook，确保已安装上述依赖；CIFAR-10 与 GloVe / FastText 首次运行会自动下载数据。

## 参考资料

- [Backpropagation, Intuitions — CS231n](http://cs231n.github.io/optimization-2/)
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [Efficient Estimation of Word Representations in Vector Space (Skip-Gram / CBoW)](https://arxiv.org/abs/1301.3781)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [FastText: Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
- [Word2Vec 交互式演示 — wevi](https://ronxin.github.io/wevi/)
