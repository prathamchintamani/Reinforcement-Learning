# main.py

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import dqn_helpers as helper

# Instantiate the environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Environment parameters
nS = env.observation_space.n 
nA = env.action_space.n

# Hyperparemeters
BATCH_SIZE = 128
BUFFER_CAPACITY = 10000
LEARNING_RATE = 0.001
GAMMA = 0.99
TARGET_UPDATE_FREQ = 100
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.9995
MIN_BUFFER_SIZE_FOR_TRAIN = 1000
NUM_EPISODES = 20000

# initializing replay buffer
buffer = helper.ReplayBuffer(BUFFER_CAPACITY)

# initializing the two Deep Q Networks
train_model = helper.DQN(nS, nA, 16)
target_model = helper.DQN(nS, nA, 16)

target_model.load_state_dict(train_model.state_dict())
target_model.eval()

# optimizer and loss fn
loss_fn = nn.MSELoss()
optimizer = optim.Adam(train_model.parameters(), lr = LEARNING_RATE)

epsilon = EPS_START
train_step_counter = 0
current_state_idx, info = env.reset()
current_state = helper.one_hot_encode(current_state_idx, nS)

# fill replay buffer
while True:
    action = env.action_space.sample()
    next_state_idx, reward, terminated, truncated, info = env.step(action)
    next_state = helper.one_hot_encode(next_state_idx, nS)
    buffer_element = (current_state, action, reward, next_state, terminated or truncated)
    buffer.append(*buffer_element)
    current_state = next_state

    if terminated or truncated:
        current_state_idx, info = env.reset()
        current_state = helper.one_hot_encode(current_state_idx, nS)
    
    if len(buffer) >= MIN_BUFFER_SIZE_FOR_TRAIN:
        print(f'buffer has achieved min capacity')
        break
net_episodic_reward = 0
for episode in range(NUM_EPISODES):
    initial_state_idx, info = env.reset()
    initial_state = helper.one_hot_encode(initial_state_idx, nS)
    terminated = False
    truncated = False
    episode_reward = 0
    current_state = initial_state
    while not (terminated or truncated):
        rand = np.random.rand()
        if rand < epsilon:
            action = env.action_space.sample()
        else:
            state = torch.from_numpy(current_state).float().unsqueeze(0)
            with torch.no_grad():
                q_values = train_model.forward(state)
            action = torch.argmax(q_values).item()
        
        next_state_idx, reward, terminated, truncated, info = env.step(action)
        next_state = helper.one_hot_encode(next_state_idx, nS)
        buffer_element = (current_state, action, reward, next_state, terminated or truncated)
        buffer.append(*buffer_element)
        current_state = next_state
        episode_reward += reward

        if len(buffer) >= MIN_BUFFER_SIZE_FOR_TRAIN:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
            states_tensor = torch.from_numpy(states).float()
            actions_tensor = torch.from_numpy(actions).long()
            rewards_tensor = torch.from_numpy(rewards).float()
            next_states_tensor = torch.from_numpy(next_states).float()
            dones_tensor = torch.from_numpy(dones).float()

            all_current_q_values = train_model.forward(states_tensor)
            current_q_values = all_current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_values_next_from_train = train_model(next_states_tensor)
                best_actions_from_train = torch.argmax(q_values_next_from_train, dim=1)

                q_values_next_from_target = target_model(next_states_tensor)
                max_next_q_values = q_values_next_from_target.gather(1, best_actions_from_train.unsqueeze(1)).squeeze(1)
            target_q_values = rewards_tensor + GAMMA * max_next_q_values * (1 - dones_tensor)
            loss = loss_fn(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step_counter += 1
            if train_step_counter >= TARGET_UPDATE_FREQ:
                target_model.load_state_dict(train_model.state_dict())
                target_model.eval()
                train_step_counter = 0
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    net_episodic_reward += episode_reward
    if (episode+1)%500 == 0:
        print(f'Episode {episode+1}/{NUM_EPISODES} \n average episode reward = {net_episodic_reward/500}, current epsilon = {epsilon}\n')
        net_episodic_reward = 0



NUM_TEST_EPISODES = 1000
epsilon = 0
net_reward = 0
for episode in range(NUM_TEST_EPISODES):
    initial_state_idx, info = env.reset()
    initial_state = helper.one_hot_encode(initial_state_idx, nS)
    terminated = False
    truncated = False
    episode_reward = 0
    current_state = initial_state
    while not (terminated or truncated):
        rand = np.random.rand()
        if rand < epsilon:
            action = env.action_space.sample()
        else:
            state = torch.from_numpy(current_state).float().unsqueeze(0)
            with torch.no_grad():
                q_values = train_model.forward(state)
            action = torch.argmax(q_values).item()
        
        next_state_idx, reward, terminated, truncated, info = env.step(action)
        current_state = helper.one_hot_encode(next_state_idx, nS)
        episode_reward += reward

    net_reward += episode_reward

print(f'average reward per episode in test runs = {net_reward/NUM_TEST_EPISODES}')
torch.save(train_model.state_dict(), 'frozen_lake_dqn_state_dict.pth')
env.close()