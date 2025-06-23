
**1. (10 points)** Given numeric truth values:

$\mu(A) = 0.4/0.5 + 0.7/0.8 + 0.6/0.9$

$\mu(B) = 0.5/0.5 + 0.6/0.8 + 0.7/0.9$

Calculate the fuzzy logic truth of the following:

1. $\neg A$
2. $A \land B$
3. $A \lor B$
4. $A \to B$
5. $B \to A$



**2. (15 points)**
 Given information about the name, age, nationality of a person from a group using the following deffemplate:

```
(deffemplate person
  (slot name)
  (slot age)
  (slot occupation)
  (slot nationality))
```

For example, we can write a rule to identify the people who are from France:

```
(defrule example
  (person (name ?nm) (nationality France))
  =>
  (printout t ?nm))
```

**(a)** (5 points) Write a rule that matches persons who are from the U.S. and are computer programmers.

**(b)** (5 points) Write a rule that matches persons whose age is between 20 and 25 and whose nationality is either "French" or "German".

**(c)** (5 points) Write a rule that modifies the age of persons who are "Chinese" to increase it by 5 years.



下面是详细解答：

### 问题解析

给定模板 `person`，其中包括了 `name`（姓名）、`age`（年龄）、`occupation`（职业）、`nationality`（国籍）四个槽位。你需要根据这些信息编写规则（rules）。

### (a) 写一个规则，匹配来自美国且职业是计算机程序员的人：

```clips
(defrule match-us-programmers
  (person (nationality "U.S.") (occupation "computer programmer"))
  =>
  (printout t "Found a computer programmer from the U.S." crlf))
```

**解释：**

- `(person (nationality "U.S.") (occupation "computer programmer"))`：这条规则匹配的是国籍是美国（`U.S.`）且职业是计算机程序员（`computer programmer`）的 `person` 实例。
- 如果匹配成功，执行 `(printout t "Found a computer programmer from the U.S." crlf)` 打印消息。

### (b) 写一个规则，匹配年龄在20到25岁之间，且国籍是“法国”或“德国”的人：

```clips
(defrule match-young-french-or-german
  (person (age ?age) (nationality ?nation))
  (test (and (>= ?age 20) (<= ?age 25) (or (eq ?nation "French") (eq ?nation "German"))))
  =>
  (printout t "Found a person aged between 20 and 25 from either France or Germany." crlf))
```

**解释：**

- `(person (age ?age) (nationality ?nation))`：匹配所有 `person` 实例，其中包含年龄和国籍。
- `test` 部分使用逻辑运算符进行判断：
  - `(>= ?age 20)` 和 `(<= ?age 25)` 确保年龄在20到25岁之间。
  - `(or (eq ?nation "French") (eq ?nation "German"))` 确保国籍是“法国”或“德国”。
- 如果条件满足，则执行 `(printout t "Found a person aged between 20 and 25 from either France or Germany." crlf)` 打印消息。

### (c) 写一个规则，修改中国人（`Chinese`）的年龄，将其增加5岁：

```clips
(defrule increase-age-for-chinese
  (person (nationality "Chinese") (age ?age))
  =>
  (modify ?person (age (+ ?age 5)))
  (printout t "Increased age for Chinese person by 5 years." crlf))
```

**解释：**

- `(person (nationality "Chinese") (age ?age))`：匹配所有国籍是“中国”（`Chinese`）的 `person` 实例。
- `modify` 用来修改符合条件的事实的槽位值。在这里，`(modify ?person (age (+ ?age 5)))` 会将找到的中国人的年龄增加5岁。
- `printout t "Increased age for Chinese person by 5 years." crlf` 打印消息，表示年龄已增加。

------

### 总结

这些规则都通过 `deffacts` 给定的事实模板 `person` 来匹配并执行条件判断。每个规则都基于不同的匹配条件，利用 `test` 语句来进行逻辑判断，最终在规则执行时进行相应的操作或输出。





**3. (10 points) Use resolution to solve the following problem.**

1. Every investor bought stocks or bonds.

2. If the Dow-Jones Average crashes, then all stocks that are not gold stocks fail.

3. If the T-Bill interest rate rises, then all bonds fail.

4. Every investor who bought something that falls is not happy. 

   Prove this conclusion: If the Dow-Jones Average crashes and the T-Bill interest rate rises, then any investor who is happy bought some gold stock.

   

   



我们使用**分辨法（Resolution）**来解决这个逻辑问题。分辨法是一种通过消去命题中的变量来推导结论的技术。首先，我们需要将问题中的自然语言表述转换为逻辑公式，然后使用分辨法一步步推导出结论。

### 问题的分析

我们需要证明如下的结论：

> **如果道琼斯指数崩盘并且T-Bill利率上升，那么任何感到开心的投资者都购买了一些黄金股票。**

这个结论可以用形式化的语言表达为：

$\text{(Dow crash)} \land \text{(T-Bill rise)} \rightarrow \text{(Happy investor)} \rightarrow \text{(Bought gold stock)}$

### 1. 将给定的语句转化为逻辑公式

根据题意，我们将每个条件转化为命题逻辑公式：

1. **每个投资者购买了股票或债券**：
   这是一个关于投资者的普遍性陈述。可以用命题符号表示为：

   ∀x(B(x)∨S(x))(每个投资者x购买了股票或债券)\forall x (B(x) \lor S(x)) \quad \text{(每个投资者} x \text{购买了股票或债券)}

   其中 $B(x)$ 表示“投资者 $x$ 购买了债券”，$S(x)$ 表示“投资者 $x$ 购买了股票”。

2. **如果道琼斯指数崩盘，则所有非黄金股票都会失败**：
   我们用命题符号表示为：

   D→(∀x(¬G(x)→F(x)))D \rightarrow (\forall x (\neg G(x) \rightarrow F(x)))  

   其中 $D$ 表示“道琼斯指数崩盘”，$G(x)$ 表示“投资者 $x$ 买的是黄金股票”，$F(x)$ 表示“股票 $x$ 失败”。

3. **如果T-Bill利率上升，则所有债券都会失败**：
   用命题符号表示为：

   T→∀x(B(x)→F(x))T \rightarrow \forall x (B(x) \rightarrow F(x))  

   其中 $T$ 表示“T-Bill利率上升”。

4. **每个购买了失败股票的投资者都不开心**：
   用命题符号表示为：

   ∀x((F(x)→¬H(x)))\forall x ((F(x) \rightarrow \neg H(x)))  

   其中 $H(x)$ 表示“投资者 $x$ 开心”。

### 2. 目标推导

我们需要推导出如下的结论：

$(D \land T) \rightarrow \forall x (H(x) \rightarrow G(x))$

即：

> 如果道琼斯指数崩盘并且T-Bill利率上升，那么任何感到开心的投资者都购买了一些黄金股票。

### 3. 逻辑推导步骤

我们将进行分辨法推导，从而得出结论。为此，我们将命题公式转换为合取范式（CNF，Conjunctive Normal Form），然后通过分辨法逐步消去变量。

#### 步骤 1: 转换为合取范式（CNF）

- 公式 1：每个投资者购买股票或债券 $\forall x (B(x) \lor S(x))$。
  - 对于任意投资者 $x$，我们有 $B(x) \lor S(x)$（这个公式已经是合取范式）。
- 公式 2：如果道琼斯指数崩盘，则所有非黄金股票失败 $D \rightarrow (\forall x (\neg G(x) \rightarrow F(x)))$。
  - 转换为 $\neg D \lor (\neg G(x) \rightarrow F(x))$。
  - 再转换为 $\neg D \lor (\neg \neg G(x) \lor F(x))$，得到 $\neg D \lor (G(x) \lor F(x))$。
- 公式 3：如果T-Bill利率上升，则所有债券失败 $T \rightarrow \forall x (B(x) \rightarrow F(x))$。
  - 转换为 $\neg T \lor (B(x) \rightarrow F(x))$。
  - 再转换为 $\neg T \lor (\neg B(x) \lor F(x))$，得到 $\neg T \lor (\neg B(x) \lor F(x))$。
- 公式 4：购买失败股票的投资者都不开心 $F(x) \rightarrow \neg H(x)$。
  - 转换为 $\neg F(x) \lor \neg H(x)$。

#### 步骤 2: 使用分辨法进行推导

我们现在有了如下的命题公式（CNF）：

1. $B(x) \lor S(x)$
2. $\neg D \lor (G(x) \lor F(x))$
3. $\neg T \lor (\neg B(x) \lor F(x))$
4. $\neg F(x) \lor \neg H(x)$

现在我们从 $D$ 和 $T$ 为真的假设出发，推导出 $H(x) \rightarrow G(x)$ 的结论。首先，我们假设 $D$ 和 $T$ 为真：

- 假设 $D$ 为真，得到 $\neg D$ 为假，因此公式 2 必须为真。
- 假设 $T$ 为真，得到 $\neg T$ 为假，因此公式 3 必须为真。

接下来，我们考虑 $H(x)$ 为真时的推导：

- 假设 $H(x)$ 为真，我们需要证明 $G(x)$ 为真。
- 如果 $H(x)$ 为真，则根据公式 4（$\neg F(x) \lor \neg H(x)$），我们得到 $F(x)$ 为假。
- 由于 $F(x)$ 为假，根据公式 2 和公式 3，我们可以推导出 $G(x)$ 必须为真。

因此，我们得出了结论：如果道琼斯指数崩盘并且T-Bill利率上升，那么任何感到开心的投资者都购买了一些黄金股票。

### 结论

通过分辨法推导，我们证明了如下结论：

**如果道琼斯指数崩盘并且T-Bill利率上升，那么任何感到开心的投资者都购买了一些黄金股票。**



**4. (10 points)**
 Considering the following database of dogs represented by 7 training examples. The target is to classify dangerous or well-behaved dogs, based on the other 3 attributes (Color, Fur, Size) of the dog. Construct the decision tree from the following examples and show the value of the information gain for each candidate attribute at each step in the construction of the tree.

The values of $\log_2(p)$ are provided as:

- $\log_2(4/7) = -0.807$
- $\log_2(3/7) = -1.222$
- $\log_2(1/3) = -1.585$
- $\log_2(2/3) = -0.585$
- $\log_2(0.75) = -0.415$

**Training Examples:**

| Dog  | Color | Fur    | Size  | Class        |
| ---- | ----- | ------ | ----- | ------------ |
| 1    | brown | ragged | small | well-behaved |
| 2    | black | ragged | big   | dangerous    |
| 3    | black | smooth | big   | dangerous    |
| 4    | black | curly  | small | well-behaved |
| 5    | white | curly  | small | well-behaved |
| 6    | white | smooth | small | dangerous    |
| 7    | brown | ragged | big   | well-behaved |



**5. (10 points)**
 Design a genetic algorithm to solve the **Fair Treasure Split** problem:
 Several people found a treasure that contains diamonds of different prices. They need to split the treasure into parts in such a way that the total difference in the price is minimal.

**Formal definition**:
 We have a set of numbers $S$. We need to split it into $n$ subsets $S_1, S_2, ..., S_n$, such that:
$$
\sum_{1 \leq i,j \leq n} \left| \sum_{x \in S_i} x - \sum_{y \in S_j} y \right| \to \min
$$
and

$S_1 \cup S_2 \cup \dots \cup S_n = S, \quad S_i \cap S_j = \emptyset$

**Hints**: Write the steps of the genetic algorithm and think about how to utilize them in this problem.



**问题分析**：
 这个问题是一个经典的 **公平宝藏分配问题**，其目标是将一组不同价值的钻石（数值集合 $S$）分配给 $n$ 个人，使得每个人获得的钻石的总价值尽量接近，从而使得总的价值差异最小。其数学表达式是：

$\sum_{1 \leq i,j \leq n} \left| \sum_{x \in S_i} x - \sum_{y \in S_j} y \right| \to \min$

其中，$S_1, S_2, \dots, S_n$ 是从集合 $S$ 中分出的 $n$ 个子集，且满足 $S_1 \cup S_2 \cup \dots \cup S_n = S$ 且 $S_i \cap S_j = \emptyset$ （即每个子集之间没有交集）。

### 解决方案：基于遗传算法的思路

**遗传算法**是一种模拟自然选择和遗传机制的优化算法，适用于求解诸如该问题等的组合优化问题。我们将设计一个遗传算法来实现该问题的求解。遗传算法的主要步骤包括：初始化种群、选择操作、交叉操作、变异操作和终止条件。

### 遗传算法的具体设计步骤：

1. **个体表示**：

   - 我们的个体是一个数组，其中每个元素表示一个钻石属于哪一个子集。假设集合 $S = \{s_1, s_2, \dots, s_m\}$，每个个体由 $m$ 个值组成，每个值表示该位置的钻石属于哪个子集。
   - 例如，如果有 3 个人（$n = 3$），并且 $S = \{10, 20, 30, 40\}$，一个个体可能是 `[1, 2, 1, 3]`，表示第 1 个和第 3 个钻石分别属于子集 1，第 2 个钻石属于子集 2，第 4 个钻石属于子集 3。

2. **适应度函数**：

   - 适应度函数用于评估一个个体的质量，即其分配的钻石价值差异。
   - 我们需要计算每两个子集的总价值差异之和。具体地，对于每两个子集 $S_i$ 和 $S_j$，计算它们的价值差异：

   差异=∣∑x∈Six−∑y∈Sjy∣\text{差异} = \left| \sum_{x \in S_i} x - \sum_{y \in S_j} y \right|

   然后，对所有子集对的差异求和。适应度越低，表示分配结果越公平。

3. **初始化种群**：

   - 随机生成多个个体（每个个体是一个合法的分配方案）。这些个体将构成我们的初始种群。
   - 每个个体代表一种可能的钻石分配方式。

4. **选择操作**：

   - 选择操作用于从当前种群中选择出优秀的个体，准备进行交叉和变异。
   - 常用的选择方法有**轮盘赌选择**和**锦标赛选择**。在本问题中，我们可以选择适应度更高（差异更小）的个体参与下一代的繁殖。

5. **交叉操作**：

   - 交叉操作是遗传算法中的核心操作之一，它模拟了基因的重组过程，产生新的个体。
   - 交叉方法可以采用**单点交叉**或**多点交叉**。例如，我们可以选择两个个体，随机选择一个位置，然后交换两个个体在该位置之后的部分。

   例如，如果父代个体1是 `[1, 2, 1, 3]`，父代个体2是 `[2, 1, 3, 1]`，在第 2 个位置交叉后，子代可能变成 `[1, 1, 3, 3]`。

6. **变异操作**：

   - 变异操作用于引入多样性，避免算法陷入局部最优解。
   - 变异方法可以是随机改变某些钻石的分配，使其属于其他子集。例如，将个体中的某个位置的子集标签随机改变，表示某个钻石从一个人那里被换到另一个人那里。

7. **终止条件**：

   - 遗传算法通常会根据以下条件之一停止：
     - 达到最大迭代次数。
     - 找到满足预期的解，适应度达到一定阈值。
     - 适应度不再有显著变化。

### 算法步骤：

1. **初始化**：随机生成多个个体，构成初始种群。
2. **评估适应度**：计算每个个体的适应度（即价值差异的总和）。
3. **选择**：根据适应度选择父代个体。
4. **交叉**：对选择出的父代个体进行交叉，生成新的子代。
5. **变异**：对子代个体进行变异，增加多样性。
6. **替换**：根据适应度选择下一代种群。
7. **终止条件**：检查是否满足终止条件，若满足，则输出最优解；否则，返回第 2 步继续迭代。

### 示例代码框架（Python）：

```python
import random
import numpy as np

# 适应度函数：计算所有子集之间的价值差异总和
def fitness(individual, values, n):
    subsets = [[] for _ in range(n)]
    for i in range(len(individual)):
        subsets[individual[i] - 1].append(values[i])
    
    total_diff = 0
    for i in range(n):
        for j in range(i + 1, n):
            sum_i = sum(subsets[i])
            sum_j = sum(subsets[j])
            total_diff += abs(sum_i - sum_j)
    
    return total_diff

# 交叉操作
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# 变异操作
def mutate(individual, n):
    index = random.randint(0, len(individual) - 1)
    individual[index] = random.randint(1, n)

# 遗传算法主函数
def genetic_algorithm(values, n, population_size=100, generations=1000):
    # 初始化种群
    population = [random.choices(range(1, n + 1), k=len(values)) for _ in range(population_size)]
    
    for generation in range(generations):
        # 计算适应度
        population_fitness = [fitness(ind, values, n) for ind in population]
        
        # 选择最好的个体
        sorted_population = [x for _, x in sorted(zip(population_fitness, population))]
        
        # 生成新种群
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = sorted_population[i], sorted_population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            mutate(child1, n)
            mutate(child2, n)
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]

    # 返回最优解
    best_individual = sorted_population[0]
    return best_individual, fitness(best_individual, values, n)

# 示例使用
values = [10, 20, 30, 40]  # 钻石的价值
n = 3  # 分配给3个人
best_solution, best_fitness = genetic_algorithm(values, n)
print("最优分配方案:", best_solution)
print("最小差异总和:", best_fitness)
```

### 总结：

通过遗传算法，我们可以有效地为多个人分配钻石，并尽量让每个人的钻石价值差异最小化。这个方法通过模拟自然选择和遗传机制，探索可能的解决方案，并逐步优化，最终得到一个较优的分配方案。



**6. (15 points)** Hidden Markov Model
 Given a hidden Markov model consisting of a box and a ball $\lambda = (A, B, \pi)$, where:

$A = \begin{bmatrix}  0.6 & 0.3 & 0.1 \\ 0.2 & 0.5 & 0.3 \\ 0.7 & 0.3 & 0.0 \\ \end{bmatrix},  \quad B = \begin{bmatrix}  0.7 & 0.3 \\ 0.4 & 0.6 \\ 0.5 & 0.5 \\ \end{bmatrix},  \quad \pi = (0.5, 0.3, 0.2)$

Box state collection: $\{ \text{box1}, \text{box2}, \text{box3} \}$

Color of ball collection: $\{ \text{red}, \text{white} \}$

Let $\tau = 3$, and the observations are $\{ \text{red}, \text{white}, \text{white} \}$. Compute the best path.



122



**7. (15 points) Neural Networks**

(1) The following figure shows the architecture of a simple neural network with an input as $X = [X_1, X_2, X_3]^T$. It includes two hidden layers, represented as $H = [H_1, H_2, H_3]^T$ and $V = [V_1, V_2]^T$, and the output is denoted as $Y$ (no bias used).

The activation function of the hidden and output layers takes the form of ReLU:

$$
s(z) = \max(0, z)
$$

The loss function is defined as:

$$
L(Y, T) = \frac{1}{2}(Y - T)^2
$$

where $T$ is target value of $Y$. The weight of input layer is $W_1$, and the weight of hidden layer is $W_2, W_3$.

We assume that:

$$
W_1 = \begin{pmatrix}
1 & 2 & -1 \\
2 & -2 & 1 \\
-1 & 0 & 2
\end{pmatrix},\quad
W_2 = \begin{pmatrix}
0 & 1 & -1 \\
1 & 1 & -2
\end{pmatrix},\quad
W_3 = \begin{pmatrix}
2 & -1
\end{pmatrix}.
$$
Also, the input $X = [1, 0, -2]^T$, and $T = 1$.

(i) Please calculate the value of $Y$.

(ii) Assume that the weight edge connecting $X_1$ and $H_3$ is called $A$, the weight edge connecting $H_3$ and $V_1$ is called $B$, the weight edge connecting $V_1$ and $Y$ is called $C$. Please calculate:
$$
\frac{\partial L}{\partial A}, \frac{\partial L}{\partial B}, \frac{\partial L}{\partial C}
$$

(2) A picture is represented as:

$$
\begin{pmatrix}
0 & -1 & 1 & 3 \\
1 & 0  & 2 & 1 \\
0 & -1 & 1 & 3
\end{pmatrix}
$$
The convolution kernel is represented as:

$$
\begin{pmatrix}
1  & -1 \\
1  &  2
\end{pmatrix}
$$
The stride is 1. 

(i)Please show the result after convolution operation.

(ii) Assume the pooling size is $2 \times 2$ and the pooling stride is 1. Please calculate the results after max pooling and average pooling based on the result above. Please explain the reason why pooling is important.

(3) Batch normalization and layer normalization are often used in practice. Please answer why the normalization is important, and explain the differences of the batch normalization and layer normalization.







**8. (15 points)** Kobe decides to train a single sigmoid unit using the following error function:

$E(w) = \frac{1}{2} \sum_i (y(x_i, w) - y_i)^2 + \frac{1}{2} \beta \sum_j w_j^2$

where:

- $y(x_i, w) = s(x_i, w)$ with $s(z) = \frac{1}{1 + e^{-z}}$
- $s'(z) = s(z)(1 - s(z))$ being our usual sigmoid function.

**(1)** Write an expression for $\frac{\partial E}{\partial w_j}$. Your answer should not involve derivatives.

**(2)** According to (1), what update should be made to weight $w_j$ given a single training example $\langle x, y \rangle$? Your answer should not involve derivatives.

**(3)** Here are two graphs of the output of the sigmoid unit as a function of a single feature $x$. The unit has a weight for $x$ and an offset. The two graphs are made using different values of the magnitude of the weight vector $\| w \|^2 = \sum_j w_j^2$.

**(4)** Which of the graphs is produced by the larger $\| w \|^2$? Explain.

**(5)** Why might penalizing large $\| w \|^2$, as we could do above by choosing a positive $\beta$, be desirable? How might Grady select a good value for $\beta$ for a particular classification problem?

图A：展示了一个较缓慢的sigmoid函数。

图B：展示了一个更陡峭的sigmoid函数。

![image-20250609210652476](C:\Users\27556\AppData\Roaming\Typora\typora-user-images\image-20250609210652476.png)

