[wiki 最小二乘法](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95)

# 引入
某次实验得到了四个数据点`$(x,y):(1,6),(2,5),(3,7),(4,10)$`。

我们希望找出一条和这四个点最匹配的直线`$y=\beta_0+\beta_1x$`，即找出在某种“最佳情况”下能够大致符合如下超定线性方程组的`$\beta_1$`和`$\beta_2$`：
```math
\beta_0+1\beta_1x_1+0\beta_2=6

\beta_0+2\beta_1x_1+0\beta_2=5

\beta_0+3\beta_1x_1+0\beta_2=7

\beta_0+4\beta_1x_1+0\beta_2=10
```
最小二乘法采用的手段是尽量使得等号两边的方差最小，也就是找出这个函数的最小值：
```math
S(\beta_0,\beta_1)=[6-(\beta_0+1\beta_1x)]^2-[5-(\beta_0+2\beta_1x)]^2-[7-(\beta_0+3\beta_1x)]^2-[10-(\beta_0+4\beta_1x)]^2
```
最小值可以通过对`$S(\beta_0,\beta_1)$`分别求`$\beta_0$`和`$\beta_1$`偏导数，然后使它们等于零得到:
```math
\frac{\partial S}{\partial \beta_0}=8\beta_0+20\beta_1-56=0

\frac{\partial S}{\partial \beta_1}=20\beta_0+60\beta_1-154=0
```
如此就得到了一个只有两个未知数的方程组，很容易就可以解出：
```math
\beta_0=3.5

\beta_1=1.4
```
也就是说直线`$y=3.5+1.4x$`是最佳的。

用 sk-learn 求解会得到同样的结果：
```
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

reg = linear_model.LinearRegression()
X = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
y = np.array([6, 5, 7, 10])
reg.fit(X, y)

figure, ax = plt.subplots()
ax.set_xlim(left=0, right=5)
ax.set_ylim(bottom=0, top=11)

plt.scatter(X[:, 0], y)

print(reg.coef_)
print(reg.intercept_)

line1 = [(0, reg.intercept_ + reg.coef_[0] * 0), (5, reg.intercept_ +
                                                  reg.coef_[0] * 5)]
(line1_xs, line1_ys) = zip(*line1)
ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))

plt.plot()
plt.show()
```
```
[ 1.4  0. ]
3.5
```
![image](http://note.youdao.com/yws/api/personal/file/1D1F1656FA9048AEA4C81331DA7E2092?method=download&shareKey=06bff9333cb6714693ed9f2a3fc29447)

# 方法
人们对由某一变量`$t$`或多个变量`$t_1,...,t_n$`构成的相关变量`$y$`感兴趣。如弹簧的形变与所用的力相关，一个企业的盈利与其营业额，投资收益和原始资本有关。为了得到这些变量同y之间的关系，便用不相关变量去构建y，使用如下函数模型：
```math
y_m=f(t_1,...,t_q;b_1,...,b_p)
```
`$q$`个独立变量`$t$`与`$p$`个系数`$b$`去拟合。

通常人们将一个可能的、对不相关变量t的构成都无困难的函数类型称作函数模型（如抛物线函数或指数函数）。参数b是为了使所选择的函数模型同观测值y相匹配。（如在测量弹簧形变时，必须将所用的力与弹簧的膨胀系数联系起来）。其目标是合适地选择参数，使函数模型最好的拟合观测值。一般情况下，观测值远多于所选择的参数。
其次的问题是怎样判断不同拟合的质量。高斯和勒让德的方法是，假设测量误差的平均值为0。令每一个测量误差对应一个变量并与其它测量误差不相关（随机无关）。人们假设，在测量误差中绝对不含系统误差，它们应该是纯偶然误差(有固定的变异数)，围绕真值波动。除此之外，测量误差符合正态分布，这保证了偏差值在最后的结果y上忽略不计。

确定拟合的标准应该被重视，并小心选择，较大误差的测量值应被赋予较小的权。并建立如下规则：被选择的参数，应该使算出的函数曲线与观测值之差的平方和最小。用函数表示为：
```math
\min_{\bar x}\sum_{i=1}^n(y_m-y_i)^2
```
用欧几里得度量表达为：
```math
\min_{\bar b} \| \bar y_m(\bar b)-\bar y \|_2
```
最小化问题的精度，依赖于所选择的函数模型。

# 线性函数模型
典型的一类函数模型是线性函数模型。最简单的线性式是`$y=b_{0}+b_{1}t$`，写成矩阵式，为
```math
\min_{b_0,b_1} \left \|
\begin{bmatrix}
1      & \cdots & t_1      \\
\vdots & \ddots & \vdots \\
1      & \cdots & t_n
\end{bmatrix}
\begin{bmatrix}
b_0      \\
b_1
\end{bmatrix}
-
\begin{bmatrix}
y_1      \\
\vdots \\
y_n
\end{bmatrix}
\right\|_2
=
\min_{b}\left\|Ab-Y\right\|_2
```
直接给出该式的参数解：
```math
b_1=\frac{\sum_{i=1}^nt_iy_i-n\bar t \bar y}{\sum_{i=1}^nt_i^2-n(\bar t)^2}

b_0=\bar y-b_1 \bar t
```
其中，`$\bar t=\frac{1}{n}\sum_{i=1}^nt_i$`，为`$t$`的平均数，也可解得如下形式：
```math
b_1=\frac{\sum_{i=1}^n(t_i-\bar t)(y_i-\bar y)}{\sum_{i=1}^n(t_i-\bar t)^2}
```
### 简单线性模型是`$y=b_{0}+b_{1}t$`的例子
随机选定10艘战舰，并分析它们的长度与宽度，寻找它们长度与宽度之间的关系。由下面的描点图可以直观地看出，一艘战舰的长度（t）与宽度（y）基本呈线性关系。散点图如下：

以下图表列出了各战舰的数据，随后步骤是采用最小二乘法确定两变量间的线性关系。

![](http://note.youdao.com/yws/api/personal/file/C4D1A8F2CD5943D0B2A6A8824CB70442?method=download&shareKey=02bb4ca1f54b7f5dae64d200b6db094e)

根据上面的数据，`$\bar t=\frac{\sum_{i=1}^nt_i}{n}=\frac{1678}{10}=167.8$`，得到`$\bar y=18.41$`。

然后确定`$\bar b_1$`：
```math
b_1=\frac{\sum_{i=1}^n(t_i-\bar t)(y_i-\bar y)}{\sum_{i=1}^n(t_i-\bar t)^2}
=\frac{3287.820}{20391.60}=0.1612
```
可以看出，战舰的长度每变化1m，相对应的宽度便要变化16cm。并由下式得到常数项`$b_0$`：
```math
b_0=\bar y-b_1\bar t=18.41-0.1612*167.8=-8.6394
```
在这里随机理论不加阐述。可以看出点的拟合非常好，长度和宽度的相关性大约为96.03％。 

### 一般线性情况
![](http://note.youdao.com/yws/api/personal/file/A8288692AB884AB6B866FEF429943BEA?method=download&shareKey=9448be77d774dcc48e6d55c049662017)

### 最小二乘法的解

![](http://note.youdao.com/yws/api/personal/file/66CF94C00F7D4C58AC0AADDF9A06B68E?method=download&shareKey=274e58e05e2c744e29e134b8a93da2e9)