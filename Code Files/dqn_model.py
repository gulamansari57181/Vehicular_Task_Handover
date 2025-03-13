import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
from collections import deque

# Simulation Parameters
NUM_VEHICLES = 80
NUM_RSUS = 30
STATE_SIZE = 6  # Features: CPU freq, transmission rates, distance, task size
ACTION_SIZE = 3  # Offloading choices: Local, RSU, Vehicle
GAMMA = 0.9  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MEMORY_SIZE = 5000
EPISODES = 250

class SimulationParameters:
    def __init__(self):
        self.num_vehicles = NUM_VEHICLES
        self.num_rsus = NUM_RSUS
        self.vehicle_speed_range = [30, 120]  # km/h
        self.vehicle_cpu_freq_range = [2e9, 8e9]  # Hz
        self.vehicle_transmit_power = 0.1  # W
        self.rsu_cpu_freq_range = [8e9, 16e9]  # Hz
        self.rsu_transmit_power = 0.2  # W
        self.input_data_size = 16e6  # bits (16 Mbits)
        self.processing_complexity = 1000  # CPU cycles per bit
        self.kappa = 1e-27  # Power coefficient
        self.bandwidth = 10e6  # Hz
        self.noise_power = 1  # W
        self.path_loss_exponent = 4
        self.channel_fading = 1.0  # Fading coefficient
        self.alpha = 1  # Uplink transmission overhead coefficient
        self.beta = 0.01  # Downlink transmission overhead coefficient
        
        # Initialize properties
        self.initialize_vehicles()
        self.initialize_rsus()
        self.initialize_distances()
    
    def initialize_vehicles(self):
        self.vehicle_speeds = np.random.uniform(self.vehicle_speed_range[0], self.vehicle_speed_range[1], self.num_vehicles)
        self.vehicle_cpu_freqs = np.random.uniform(self.vehicle_cpu_freq_range[0], self.vehicle_cpu_freq_range[1], self.num_vehicles)
    
    def initialize_rsus(self):
        self.rsu_cpu_freqs = np.random.uniform(self.rsu_cpu_freq_range[0], self.rsu_cpu_freq_range[1], self.num_rsus)
    
    def initialize_distances(self):
        self.distances_VT_Vi = np.random.uniform(1e-9, 200, self.num_vehicles)
        self.distances_VT_Bj = np.random.uniform(1e-9, 200, self.num_rsus)

    def calculate_transmission_rate(self, P, d):
        """Calculate data transmission rate using the Shannon-Hartley theorem."""
        d = max(d, 1e-9)  # Avoid division by zero
        SNR = (P * (d ** (-self.path_loss_exponent)) * (np.abs(self.channel_fading) ** 2)) / self.noise_power
        return self.bandwidth * np.log2(1 + SNR)

    def calculate_local_delay(self, f_loc):
        return (self.processing_complexity * self.input_data_size) / f_loc

    def calculate_local_energy(self, f_loc):
        P_CPU = self.kappa * (f_loc ** 2)
        return P_CPU * self.processing_complexity * self.input_data_size

    def calculate_offloading_delay(self, f_edge, transmission_rate):
        """Total offloading delay = UL delay + Edge Processing delay + DL delay"""
        D_UL = self.alpha * (self.input_data_size / transmission_rate)
        D_edge = (self.processing_complexity * self.input_data_size) / f_edge
        D_DL = self.beta * (self.input_data_size / transmission_rate)
        return D_UL + D_edge + D_DL

    def calculate_offloading_energy(self, P, transmission_rate):
        """Total energy = transmission energy"""
        return (P * self.processing_complexity * self.input_data_size) / transmission_rate

# DQN Model
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)  # Efficient memory handling
        self.epsilon = EPSILON
        self.model = self.build_model()

    def build_model(self):
        """Builds a Deep Q-Network"""
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy policy for action selection"""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def train(self):
        """Batch training of the DQN"""
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, targets = [], []
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += GAMMA * np.max(self.model.predict(np.array([next_state]), verbose=0)[0])
            
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            
            states.append(state)
            targets.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0, batch_size=BATCH_SIZE)

    def update_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

# Initialize Simulation and DQN Agent
sim_params = SimulationParameters()
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

# Training Loop
rewards_per_episode = []
for episode in range(EPISODES):
    state = np.random.rand(STATE_SIZE)  # Placeholder for actual environment state
    total_reward = 0
    for step in range(50):
        action = agent.act(state)

        if action == 0:  # Local execution
            reward = - (sim_params.calculate_local_delay(2.5e9) + sim_params.calculate_local_energy(2.5e9))
        elif action == 1:  # RSU Offloading
            reward = - (sim_params.calculate_offloading_delay(10e9, 5e6) + sim_params.calculate_offloading_energy(0.2, 5e6))
        else:  # Vehicle Offloading
            reward = - (sim_params.calculate_offloading_delay(6e9, 3e6) + sim_params.calculate_offloading_energy(0.1, 3e6))

        next_state = np.random.rand(STATE_SIZE)  # Placeholder
        done = step == 49
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.train()
    agent.update_epsilon()
    rewards_per_episode.append(total_reward)
    print(f"Episode {episode+1}: Total Reward = {total_reward:.4f}")

plt.plot(range(EPISODES), rewards_per_episode, color='blue')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress')
plt.grid(True)
plt.show()
