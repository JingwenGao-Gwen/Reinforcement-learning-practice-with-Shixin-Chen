import numpy as np

def build_mdp():
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

def value_iteration_normalized(max_iter=500, tol=1e-5):
    costs, transitions = build_mdp()
    V = {1: 0.0, 2: 0.0}
    history = []

    for iteration in range(max_iter):
        new_V = {}
        for s in [1, 2]:
            actions = [1, 2, 3] if s == 1 else [1, 2]
            Q_vals = []
            for a in actions:
                c = costs[(s, a)]
                p = transitions[(s, a)]
                Q = c + p[0]*V[1] + p[1]*V[2]
                Q_vals.append(Q)
            new_V[s] = min(Q_vals)

        # Normalize: set V(1) = 0
        offset = new_V[1]
        for s in new_V:
            new_V[s] -= offset
        gamma = offset

        history.append({
            "iter": iteration,
            "V": dict(new_V),
            "gamma": gamma
        })

        delta = max(abs(new_V[s] - V[s]) for s in [1, 2])
        if delta < tol:
            break
        V = new_V.copy()

    # Extract optimal policy
    policy = {}
    for s in [1, 2]:
        actions = [1, 2, 3] if s == 1 else [1, 2]
        Qs = {}
        for a in actions:
            c = costs[(s, a)]
            p = transitions[(s, a)]
            Qs[a] = c + p[0]*V[1] + p[1]*V[2]
        policy[s] = min(Qs.items(), key=lambda x: x[1])[0]

    return history, policy

# ===== 打印结果函数 =====
def print_results(history, policy):
    print("值迭代历史：")
    print(f"{'迭代':<5} | {'V(1)':<10} | {'V(2)':<10} | {'γ':<10}")
    for h in history:
        print(f"{h['iter']:<5} | {h['V'][1]:<10.4f} | {h['V'][2]:<10.4f} | {h['gamma']:<10.4f}")
    
    print("\n最终策略：")
    print(f"状态 1 → 动作 {policy[1]}")
    print(f"状态 2 → 动作 {policy[2]}")
    print(f"收敛的长期平均成本 γ ≈ {history[-1]['gamma']:.4f}")

# ===== 执行并打印 =====
if __name__ == "__main__":
    hist, pol = value_iteration_normalized()
    print_results(hist, pol)
