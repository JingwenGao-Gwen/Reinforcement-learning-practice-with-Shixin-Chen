import numpy as np

def build_mdp():
    """定义MDP的代价和转移概率"""
    costs = {
        (1, 1): 10, (1, 2): 20, (1, 3): 30,
        (2, 1): 30, (2, 2): 40
    }
    transitions = {
        (1, 1): [0.2, 0.8],
        (1, 2): [0.5, 0.5],
        (1, 3): [0.9, 0.1],
        (2, 1): [0.5, 0.5],
        (2, 2): [0.8, 0.2]
    }
    return costs, transitions

def policy_iteration_fixed(max_iter=100, tol=1e-6):
    costs, transitions = build_mdp()

    # 初始策略
    policy = {1: 2, 2: 1}
    history = []

    for iteration in range(max_iter):
        # === 策略评估 ===
        c1 = costs[(1, policy[1])]
        c2 = costs[(2, policy[2])]
        p1 = transitions[(1, policy[1])]
        p2 = transitions[(2, policy[2])]

        # h(1) = 0, 求解 gamma, h(2)
        # 方程组：
        # gamma = c1 + p1[1] * h2
        # gamma = c2 + (p2[1] - 1) * h2
        A = np.array([
            [1, -p1[1]],
            [1, -(p2[1] - 1)]
        ])
        b = np.array([c1, c2])

        gamma, h2 = np.linalg.solve(A, b)
        bias = {1: 0.0, 2: h2}

        history.append({
            'iter': iteration,
            'policy': dict(policy),
            'V': dict(bias),
            'gamma': gamma
        })

        # === 策略改进 ===
        new_policy = {}
        for s in [1, 2]:
            actions = [1, 2, 3] if s == 1 else [1, 2]
            Qs = {}
            for a in actions:
                c = costs[(s, a)]
                p = transitions[(s, a)]
                Qs[a] = c + p[0]*bias[1] + p[1]*bias[2]
            new_policy[s] = min(Qs.items(), key=lambda x: x[1])[0]

        # === 是否收敛 ===
        if all(policy[s] == new_policy[s] for s in [1, 2]):
            break
        policy = new_policy

    return history

def print_results(history):
    print("迭代历史：")
    print(f"{'迭代':<5} | {'状态1动作':<10} | {'状态2动作':<10} | {'V(1)':<10} | {'V(2)':<10} | {'γ':<10}")
    for h in history:
        print(f"{h['iter']:<5} | {h['policy'][1]:<10} | {h['policy'][2]:<10} | "
              f"{h['V'][1]:<10.4f} | {h['V'][2]:<10.4f} | {h['gamma']:<10.4f}")
    print("\n最优策略：")
    final = history[-1]
    print(f"状态1：动作{final['policy'][1]}")
    print(f"状态2：动作{final['policy'][2]}")
    print(f"最优长期平均成本 γ = {final['gamma']:.4f}")

# 主函数运行
if __name__ == "__main__":
    history = policy_iteration_fixed()
    print_results(history)
