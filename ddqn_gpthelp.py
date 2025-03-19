import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
from collections import deque

# Simulation Parameters
NUM_VEHICLES = 80
NUM_RSUS = 30
STATE_SIZE = 8  # Features: RSU count, Vehicle count, X, Y, CPU, RSU_CPU, Velocity, Direction
ACTION_SIZE = 3  # Offloading choices: Local, RSU, Vehicle
GAMMA = 0.9  # Discount factor
EPSILON = 0.9  # Initial exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 0.05
TAU = 0.01  # Soft update factor for target network
MEMORY_SIZE = 1024
EPISODES = 200
ALPHA = 1
BETA = 0.5
COMMUNICATION_RANGE = 200  
ROAD_LENGTH = 1000  
VEHICLE_SPEED_RANGE = (30, 120)  
DIRECTION_CHOICES = [-1, 1]  

# Constants for calculations
BANDWIDTH = 10e6  
NOISE_POWER = 1e-9  
PROCESSING_COMPLEXITY = 1000  
KAPPA = 1e-27  
INPUT_DATA_SIZE = 16e6  # 16 Mbits  
POWER_LOCAL = 0.5  
POWER_RSU = 0.3  
POWER_V2V = 0.2  
BANDWIDTH_COST_RSU = 0.1  
BANDWIDTH_COST_V2V = 0.05  

# Global tracking variables
global_total_latency_local = 0
global_total_latency_rsu = 0
global_total_latency_v2v = 0
global_total_energy_local = 0
global_total_energy_rsu = 0
global_total_energy_v2v = 0
global_count_local = 0
global_count_rsu = 0
global_count_v2v = 0
episode_rewards = []

# Generate random vehicle positions
true_positions = np.random.uniform(0, ROAD_LENGTH, size=(NUM_VEHICLES, 4))
true_positions[:, 2] = np.random.uniform(*VEHICLE_SPEED_RANGE, size=NUM_VEHICLES)  
true_positions[:, 3] = np.random.choice(DIRECTION_CHOICES, size=NUM_VEHICLES)  

# Mission Vehicle Class
class MissionVehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.x, self.y = true_positions[vehicle_id, :2]  
        self.velocity = true_positions[vehicle_id, 2]  
        self.direction = true_positions[vehicle_id, 3]  
        self.cpu = random.uniform(2e9, 8e9)  
        self.task_size = random.uniform(5, 50)  
        self.task_cycles = self.task_size * PROCESSING_COMPLEXITY  
        self.transmit_power = 0.1  

    def calculate_transmission_rate(self, distance):
        distance = max(distance, 1e-9)
        SNR = (self.transmit_power * (distance ** -4)) / NOISE_POWER
        return BANDWIDTH * np.log2(1 + SNR)

    def calculate_local_delay(self):
        return (PROCESSING_COMPLEXITY * INPUT_DATA_SIZE) / self.cpu

    def calculate_local_energy(self):
        return KAPPA * (self.cpu ** 2) * PROCESSING_COMPLEXITY * INPUT_DATA_SIZE

    def calculate_offloading_delay(self, f_edge, transmission_rate):
        D_UL = INPUT_DATA_SIZE / transmission_rate
        D_edge = (PROCESSING_COMPLEXITY * INPUT_DATA_SIZE) / f_edge
        return D_UL + D_edge

    def calculate_offloading_energy(self, transmission_rate):
        return (self.transmit_power * PROCESSING_COMPLEXITY * INPUT_DATA_SIZE) / transmission_rate

# Roadside Unit Class
class RoadsideUnit:
    def __init__(self, rsu_id):
        self.rsu_id = rsu_id
        self.x = random.uniform(0, ROAD_LENGTH)  
        self.y = 0  
        self.cpu = random.uniform(8e9, 16e9)  

# DDQN Agent
class DDQNAgent:
    def __init__(self, input_dim=STATE_SIZE, action_dim=ACTION_SIZE):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.online_network = self.build_network()
        self.target_network = self.build_network()
        self.optimizer = Adam(learning_rate=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0
        self.update_target_network(hard=True)

    def build_network(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_dim, activation=None)
        ])
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def update_target_network(self, hard=False):
        if hard:
            self.target_network.set_weights(self.online_network.get_weights())
        else:
            online_weights = self.online_network.get_weights()
            target_weights = self.target_network.get_weights()
            new_weights = [TAU * ow + (1 - TAU) * tw for ow, tw in zip(online_weights, target_weights)]
            self.target_network.set_weights(new_weights)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self.online_network(np.array([state]))
        return np.argmax(q_values.numpy()[0])

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)

        next_q_online = self.online_network(next_states)
        best_actions = np.argmax(next_q_online.numpy(), axis=1)
        next_q_target = self.target_network(next_states).numpy()
        target_q = rewards + GAMMA * next_q_target[np.arange(BATCH_SIZE), best_actions] * (1 - np.array(dones))

        with tf.GradientTape() as tape:
            q_values = self.online_network(states)
            q_action = tf.reduce_sum(q_values * tf.one_hot(actions, self.action_dim), axis=1)
            loss = tf.reduce_mean(tf.square(target_q - q_action))
        grads = tape.gradient(loss, self.online_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_network.trainable_variables))
        self.update_target_network()

# Initialize vehicles and RSUs
mission_vehicles = [MissionVehicle(i) for i in range(NUM_VEHICLES)]
rsus = [RoadsideUnit(i) for i in range(NUM_RSUS)]
ddqn = DDQNAgent()

# Training Loop
for episode in range(EPISODES):
    total_reward = 0
    for mv in mission_vehicles:
        available_rsus = [r for r in rsus if abs(r.x - mv.x) <= COMMUNICATION_RANGE]
        available_vehicles = [v for v in mission_vehicles if v.vehicle_id != mv.vehicle_id and abs(v.x - mv.x) <= COMMUNICATION_RANGE]

        state = [len(available_rsus), len(available_vehicles), mv.x, mv.y, mv.cpu, 
                 available_rsus[0].cpu if available_rsus else 0, mv.velocity, mv.direction]

        action = ddqn.act(state)
        transmission_rate = mv.calculate_transmission_rate(available_rsus[0].x - mv.x if available_rsus else 200)

        if action == 0:
            reward = - (mv.calculate_local_delay() + mv.calculate_local_energy())
        elif action == 1:
            reward = - (mv.calculate_offloading_delay(available_rsus[0].cpu if available_rsus else mv.cpu, transmission_rate) + mv.calculate_offloading_energy(transmission_rate))

        total_reward += reward
        ddqn.store_experience(state, action, reward, state, done=False)

    ddqn.train()
    print(f"Episode {episode+1}: Total Reward = {total_reward:.4f}")
