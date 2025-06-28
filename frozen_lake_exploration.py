import gymnasium as gym
import numpy as np
import time
env = gym.make("FrozenLake-v1", is_slippery = True, render_mode = "ansi")
nA = env.action_space.n
nS = env.observation_space.n
initial_obs, info = env.reset()

def get_model_tensors(env):
    nS = env.observation_space.n
    nA = env.action_space.n

    # Initialize P_tensor (Transition Probabilities)
    # Shape: (n_states, n_actions, n_states)
    P_tensor = np.zeros((nS, nA, nS))

    # Initialize R_tensor (Expected Immediate Rewards)
    # Shape: (n_states, n_actions)
    R_tensor = np.zeros((nS, nA))

    # Iterate through the env.unwrapped.P dictionary to populate the tensors
    for s in range(nS):
        for a in range(nA):
            # env.unwrapped.P[s][a] returns a list of (prob, next_state, reward, terminated) tuples
            for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:
                # Populate P_tensor
                P_tensor[s, a, next_state] += prob # Use += because multiple outcomes might lead to the same next_state (though rare in FrozenLake)

                # Calculate the expected immediate reward R(s,a)
                # This is the sum of (probability * reward) for all possible next states from (s,a)
                R_tensor[s, a] += prob * reward

    return P_tensor, R_tensor

def L(V_n, R, P, gamma):
    Q_values = R + gamma * np.matmul(P, V_n)
    V_next = np.max(Q_values, axis=1) 
    return V_next

def value_iteration(R, P, gamma = 0.7, theta = 1e-7):
    V = np.zeros(nS)
    
    while True:
        V_next = L(V, R, P, gamma)
        delta = np.max(np.abs(V_next - V))
        if delta < theta:
            break
        V = V_next
    return V

def extract_optimal_policy(P, R, V_star, nS, nA, gamma):
    optimal_policy = np.zeros(nS, dtype=int)

    for s in range(nS):
        Q_values_for_s = R[s, :] + gamma * np.matmul(P[s, :, :], V_star)
        optimal_policy[s] = np.argmax(Q_values_for_s)
    
    return optimal_policy

P, R = get_model_tensors(env)
optimal_V = value_iteration(R, P, gamma=0.99, theta=1e-8)
optimal_policy = extract_optimal_policy(P, R, optimal_V, nS, nA, gamma=0.99)
