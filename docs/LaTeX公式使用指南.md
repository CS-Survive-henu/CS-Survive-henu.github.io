# LaTeX公式使用指南

本指南将帮助您在CS生存指北中正确使用LaTeX数学公式。

## 基本语法

### 行内公式

在文本中插入数学公式，使用单个美元符号包围：

```markdown
这是一个行内公式：$E = mc^2$
```

效果：这是一个行内公式：$E = mc^2$

### 块级公式

独立成行的公式，使用双美元符号包围：

```markdown
$$\int_{a}^{b} f(x) dx = F(b) - F(a)$$
```

效果：
$$\int_{a}^{b} f(x) dx = F(b) - F(a)$$

## 常用数学符号

### 运算符
- 加减乘除：$+$, $-$, $\times$, $\div$
- 分数：$\frac{a}{b}$
- 根号：$\sqrt{x}$, $\sqrt[n]{x}$
- 求和：$\sum_{i=1}^{n}$
- 积分：$\int$, $\iint$, $\iiint$
- 极限：$\lim_{x \to \infty}$

### 希腊字母
- 小写：$\alpha$, $\beta$, $\gamma$, $\delta$, $\epsilon$, $\theta$, $\lambda$, $\mu$, $\pi$, $\sigma$, $\phi$, $\omega$
- 大写：$\Alpha$, $\Beta$, $\Gamma$, $\Delta$, $\Theta$, $\Lambda$, $\Sigma$, $\Phi$, $\Omega$

### 上标和下标
- 上标：$x^2$, $x^{n+1}$
- 下标：$x_1$, $x_{i,j}$
- 组合：$x_i^j$

### 比较符号
- 等于：$=$, $\neq$
- 大于小于：$>$, $<$, $\geq$, $\leq$
- 约等于：$\approx$
- 正比：$\propto$

## 数学环境

### 矩阵

```markdown
$$\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}$$
```

效果：
$$\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}$$

### 方程组

```markdown
$$\begin{cases}
x + y = 1 \\
x - y = 0
\end{cases}$$
```

效果：
$$\begin{cases}
x + y = 1 \\
x - y = 0
\end{cases}$$

### 多行公式

```markdown
$$\begin{aligned}
f(x) &= (x+a)(x+b) \\
&= x^2 + (a+b)x + ab
\end{aligned}$$
```

效果：
$$\begin{aligned}
f(x) &= (x+a)(x+b) \\
&= x^2 + (a+b)x + ab
\end{aligned}$$

## 计算机科学中的常用公式

### 算法复杂度
- 时间复杂度：$O(n)$, $O(n \log n)$, $O(n^2)$
- 空间复杂度：$\Theta(n)$, $\Omega(n)$

### 概率与统计
- 条件概率：$P(A|B) = \frac{P(A \cap B)}{P(B)}$
- 贝叶斯定理：$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$
- 期望值：$E[X] = \sum_{i} x_i P(x_i)$

### 线性代数
- 向量点积：$\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|\cos\theta$
- 矩阵乘法：$(AB)_{ij} = \sum_{k} A_{ik}B_{kj}$
- 特征值：$A\vec{v} = \lambda\vec{v}$

### 信息论
- 熵：$H(X) = -\sum_{i} P(x_i) \log_2 P(x_i)$
- 互信息：$I(X;Y) = H(X) - H(X|Y)$

### 机器学习
- 损失函数：$L(\theta) = \frac{1}{n}\sum_{i=1}^{n} (y_i - f(x_i, \theta))^2$
- 梯度下降：$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$
- Sigmoid函数：$\sigma(x) = \frac{1}{1 + e^{-x}}$

## 注意事项

1. **转义字符**：在LaTeX中，某些字符有特殊含义，如需显示原字符需要转义：
   - 下划线：`\_` 显示为 \_
   - 百分号：`\%` 显示为 \%

2. **换行**：在块级公式中使用 `\\` 进行换行

3. **空格**：LaTeX会自动处理空格，如需强制空格可使用：
   - `\,` 小空格
   - `\;` 中等空格  
   - `\quad` 大空格
   - `\qquad` 超大空格

4. **字体**：
   - 数学斜体（默认）：$x$
   - 正体：$\mathrm{d}x$
   - 黑体：$\mathbf{x}$
   - 打字机字体：$\mathtt{x}$

希望这个指南能帮助您在文档中正确使用数学公式！
