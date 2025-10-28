from scipy.optimize import linprog

# 状态-动作对的约束信息
constraints = [
    # (state s, action a, cost, [P(s'=1), P(s'=2)])
    (1, 1, 10, [0.2, 0.8]),
    (1, 2, 20, [0.5, 0.5]),
    (1, 3, 30, [0.9, 0.1]),
    (2, 1, 30, [0.5, 0.5]),
    (2, 2, 40, [0.8, 0.2])
]

# 决策变量：[V1, V2, gamma]，我们将 max gamma 转化为 min -gamma
A_ub = []
b_ub = []

for s, a, cost, prob in constraints:
    row = [0, 0, 0]  # 对应 [V1, V2, gamma]
    row[2] = 1       # γ 系数为 +1（移到 LHS）
    row[s - 1] += 1  # V(s)
    row[0] -= prob[0]  # -P(s' = 1)
    row[1] -= prob[1]  # -P(s' = 2)
    A_ub.append(row)
    b_ub.append(cost)

# 目标函数: max γ <=> min -γ
c_obj = [0, 0, -1]

# 求解 LP
res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, method="highs")

# 输出
if res.success:
    V1, V2, gamma = res.x
    print("Optimal solution:")
    print(f"V(1) = {V1:.4f}")
    print(f"V(2) = {V2:.4f}")
    print(f"γ = {gamma:.4f}")
else:
    print("LP solution failed:", res.message)
