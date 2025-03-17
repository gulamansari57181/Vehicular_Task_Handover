import numpy as np
import random
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Simulation Parameters
class SimulationParameters:
    def __init__(self):
        self.vehicle_cpu = 2.5e9  # 2.5 GHz
        self.rsu_cpu = 10e9  # 10 GHz
        self.task_size = 5e6  # 5 MB
        self.computation_intensity = 1e3  # 1000 cycles per bit
        self.bandwidth = 10e6  # 10 MHz
        self.transmit_power = 0.1  # 100 mW
        self.noise_power = 1e-9  # Noise power in Watts
        self.distance = 100  # Distance between vehicle and RSU (meters)
        self.path_loss_exponent = 3.5
        self.gamma = 0.99  # Discount factor

    def calculate_transmission_rate(self):
        channel_gain = self.noise_power / (self.distance ** self.path_loss_exponent)
        return self.bandwidth * np.log2(1 + (self.transmit_power * channel_gain) / self.noise_power)

    def calculate_local_delay(self):
        return self.task_size * self.computation_intensity / self.vehicle_cpu

    def calculate_local_energy(self):
        return 1e-27 * (self.vehicle_cpu ** 2) * self.task_size * self.computation_intensity

    def calculate_offloading_delay(self):
        uplink_rate = self.calculate_transmission_rate()
        uplink_delay = self.task_size / uplink_rate
        execution_delay = (self.task_size * self.computation_intensity) / self.rsu_cpu
        return uplink_delay + execution_delay

# Deep Q-Network (DDQN)
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tau = 0.1  # Soft update rate

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_target_model()

# Reward Calculation
def calculate_reward(delay, energy, lambda_factor=0.5):
    return - (lambda_factor * delay + (1 - lambda_factor) * energy)

# Training Loop
EPISODES = 100
state_size = 6  # Features (CPU, transmission rate, task size, etc.)
action_size = 3  # Offload to vehicle, RSU, or local

env = SimulationParameters()
agent = DQNAgent(state_size, action_size)

for e in range(EPISODES):
    state = np.reshape([env.vehicle_cpu, env.calculate_transmission_rate(), env.task_size,
                         env.computation_intensity, env.distance, env.bandwidth], [1, state_size])
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        if action == 0:
            delay = env.calculate_local_delay()
            energy = env.calculate_local_energy()
        else:
            delay = env.calculate_offloading_delay()
            energy = 0  # Offloading consumes minimal vehicle energy

        reward = calculate_reward(delay, energy)
        next_state = np.reshape([env.vehicle_cpu, env.calculate_transmission_rate(), env.task_size,
                                 env.computation_intensity, env.distance, env.bandwidth], [1, state_size])
        done = True  # Single-step simulation for now
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    agent.train(batch_size=32)
    print(f"Episode {e+1}/{EPISODES}, Reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
