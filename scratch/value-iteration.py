import numpy as np

gamma = 0.5
num_states = 5
A = [0, 1, 2]   # 0 = stay, 1 = left, 2 = right

def transitions(s, a):
    # stay
    if a == 0:
        if s in [1, 5]:
            return {s: 0.5, (2 if s == 1 else 4): 0.5}
        else:
            return {s: 0.5, s-1: 0.25, s+1: 0.25}
    # move left
    if a == 1:
        if s == 1:
            return {1: 0.5, 2: 0.5}
        else:
            return {s-1: 1/3, s: 2/3}
    # move right
    if a == 2:
        if s == 5:
            return {5: 0.5, 4: 0.5}
        else:
            return {s+1: 1/3, s: 2/3}

def value_iteration(num_iters):
    V = np.zeros(num_states + 1)  # index from 1
    for _ in range(num_iters):
        V_new = np.zeros_like(V)
        for s in range(1, num_states + 1):
            R = 1.0 if s == 5 else 0.0
            best = -1e9
            for a in A:
                trans = transitions(s, a)
                expV = sum(p * V[s2] for s2, p in trans.items())
                val = R + gamma * expV
                best = max(best, val)
            V_new[s] = best
        V = V_new
    return V[1:]  # ignore index 0


V10  = value_iteration(10)
V100 = value_iteration(100)
V200 = value_iteration(200)
print("10 :",  V10)
print("100:", V100)
print("200:", V200)
