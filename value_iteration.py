import gymnasium as gym
env = gym.make("Taxi-v3", render_mode="ansi")
state, _ = env.reset()
import numpy as np


P = env.unwrapped.P  # dict: {state: {action: [(prob, next_state, reward, done)]}}
nS = env.observation_space.n #number of states
nA = env.action_space.n #number of actions

def value_iteration(P, nS, nA, gamma = 0.99, theta = 1e-8):
    V = np.zeros(nS)
    count = 0

    while True:
        delta = 0
        for s in range(nS):
            v_old = V[s]
            Q_values = []

            for a in range(nA):
                q_sa = 0
                for (p, s_n, r, done) in P[s][a]:
                    q_sa += p * (r + gamma * V[s_n])
                Q_values.append(q_sa)

            
            V[s] = max(Q_values)
            delta = max(delta, abs(v_old - V[s]))
        count += 1

        if delta < theta:
            break
    return V, count

def extract_policy(P, nS, nA, V, gamma=0.99):
    policy = np.zeros(nS, dtype=int)

    for s in range(nS):
        best_a = 0
        best_q = -np.inf

        for a in range(nA):
            q_sa = 0
            for (p, s_n, r, done) in P[s][a]:
                q_sa += p * (r + gamma * V[s_n])
            if q_sa > best_q:
                best_q = q_sa
                best_a = a

        policy[s] = best_a

    return policy

V, count = value_iteration(P, nS, nA)
policy = extract_policy(P, nS, nA, V)

def run_policy(env, policy, max_steps=100):
    state, _ = env.reset()
    total_reward = 0

    for t in range(max_steps):
        action = policy[state]
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        print(env.render())  # show environment as text

        if done:
            break

    return total_reward, t + 1

  # or "human" for GUI
reward, steps = run_policy(env, policy)
print(f"Total reward: {reward}, Steps: {steps}, Count: {count}")
