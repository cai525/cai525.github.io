## ✅ 一、RoPE 的本质与数学形式

### 🧠 基本目标（来自图3）

我们希望通过一个函数 f(q,m)f(q, m)，给查询向量 qq 添加位置 mm 的编码，使得：
$$
\langle f(q, m), f(k, n) \rangle \propto \langle q, k \rangle \cdot \phi(m - n)
$$


也就是说，在注意力中让位置编码体现为**相对位置信息**。

------

### 📐 RoPE 的复数形式（图2 + 图3）

将 $q, k \in \mathbb{R}^d $拆成两两一组（偶数维），构成复数向量：
$$
z = q_0 + iq_1, q_2 + iq_3, \dots
$$


然后对位置 m 编码为旋转角度 $\theta m$，每个维度频率不同（类似正弦编码）：
$$
f(q, m) = \|q\| e^{i\theta(m)} = q \cdot e^{i m \theta}
$$


这表示将每个 token 的 hidden 向量**绕原点旋转一个角度**（不同维度频率不同），最终在 attention 点积中保留相对位置的信息：
$$
\text{Re}[f(q, m) f(k, n)^*] = \text{Re}[qk^* e^{i(m - n)\theta}]
$$


------

### 🧮 RoPE 的矩阵实现（图1）

为了避免使用复数，RoPE 实际上使用二维旋转矩阵：
$$
f(q, m) = R_m q \quad \text{with } R_m = \begin{bmatrix} \cos(m\theta_0) & -\sin(m\theta_0) & & \\ \sin(m\theta_0) & \cos(m\theta_0)  & & \\  & & \ddots & \\  & & & \cos(m\theta_{d/2-1}) & -\sin(m\theta_{d/2-1}) \\  & & & \sin(m\theta_{d/2-1}) & \cos(m\theta_{d/2-1}) \end{bmatrix}
$$


优点：**不会改变向量模长，保持模型稳定性**。

------

## ✅ 二、TMRoPE：RoPE 的三维扩展

### 📌 扩展动机

传统 RoPE 是一维的（时间位置）。但在多模态中，我们有：

- 文本：只需要时间位置
- 图像：需要二维空间位置（height, width）
- 视频：需要时间 + 图像（3D位置）
- 音频：只需时间，但需要精确对齐到每 40ms

因此，我们需要从一维位置编码 mm 扩展为三维位置编码 (t,h,w)(t, h, w)。

------

### TMRoPE 的数学扩展

假设我们有 token 向量  $x \in \mathbb{R}^d$，将其等分为三段：
$$
x = x^{(t)} \parallel x^{(h)} \parallel x^{(w)} \quad \text{with } d = d_t + d_h + d_w
$$


分别对时间维 tt、高度 hh、宽度 ww 应用 RoPE 旋转：
$$
\text{TMRoPE}(x, t, h, w) = R_t x^{(t)} \parallel R_h x^{(h)} \parallel R_w x^{(w)}
$$


其中：

- $R_t = \text{RoPE }$旋转矩阵对应时间位置  t;

- $R_h, R_w $同理，对应图像 patch 的空间位置;

> 可以理解为将高维向量分成 3 份，每一部分加一个“独立方向上的旋转”。

------

### ✅ TMRoPE vs RoPE 总结对比

| 特性         | 1D-RoPE            | TMRoPE                                                       |
| ------------ | ------------------ | ------------------------------------------------------------ |
| 编码维度     | 1D：时间位置 mm    | 3D：时间 tt、高 hh、宽 ww                                    |
| 向量处理     | 整体旋转（偶数维） | 拆分为 3 段分别旋转                                          |
| 模态支持     | 主要是文本         | 文本、音频、图像、视频                                       |
| 相对位置建模 | 时间上的相对位置   | 时空上的相对位置（视频、图像）                               |
| 编码结构     | RmqR_m q           | Rtx(t)∥Rhx(h)∥Rwx(w)R_t x^{(t)} \parallel R_h x^{(h)} \parallel R_w x^{(w)} |

------

## ✅ 三、优势总结

- 保留了 **RoPE 的相对位置建模能力**；
- 能处理 **不同时间粒度的模态统一对齐**（如音频40ms/frame vs 视频每帧100ms）；
- 对图像结构建模更自然（h/w）；
- 通过「拼接旋转编码」，**提高多模态对齐精度**；
- 避免了 learnable PE 的参数增长问题，仍然是无参编码。