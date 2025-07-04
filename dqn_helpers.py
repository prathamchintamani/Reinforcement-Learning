import numpy as np
import collections
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen= capacity)
        self.capacity = capacity

    def append(self, state, action, reward, next_state, done):
        return self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError("batch_size > buffer size, ensure batch_size <= capacity, \n if capacity >= batch_size, then aquire more experience")
        
        idx = np.random.randint(0,len(self.buffer), size = batch_size)
        experiences = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*experiences)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        # allows us to use len(buffer)
        return len(self.buffer)
    

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=8):
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features= hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_dim)
        )

    def forward(self, x):
        return self.output(x)
    
def one_hot_encode(state_idx, num_states):
    one_hot = np.zeros(num_states, dtype=np.float32)
    one_hot[state_idx] = 1.0
    return one_hot
