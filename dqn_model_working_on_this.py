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
STATE_SIZE = 10  # Features: CPU freq, transmission rates, distance, task size
ACTION_SIZE = 3  # Offloading choices: Local, RSU, Vehicle
GAMMA = 0.9  # Discount factor
EPSILON = 0.9  # Initial exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 0.05
TAU = 0.01
MEMORY_SIZE = 1024
EPISODES = 200
ALPHA = 1  # Weight for uplink delay
BETA = 1  # Weight for downlink delay
COMMUNICATION_RANGE = 200
ROAD_LENGTH = 1000
VEHICLE_SPEED_RANGE = (30, 120)
DIRECTION_CHOICES = [-1, 1]
EPISODES = 500

# Constants for initialization
VEHICLE_CPU_RANGE = (2e9, 8e9)  # CPU frequency range in Hz (2GHz to 8GHz)
RSU_CPU_RANGE = (8e9, 16e9)  # CPU frequency range in Hz (8GHz to 16GHz)
TASK_SIZE_RANGE = (5, 50)  # Task size in MB
CPU_CYCLES_PER_MB = 1000  # Required CPU cycles per MB
ROAD_LENGTH = 10000  # Road length in meters
VEHICLE_SPEED_RANGE = (30, 120)  # Speed range in km/hr
COMMUNICATION_RANGE = 200  # Communication range in meters
DIRECTION_CHOICES = [-1, 1]  # -1 for left, 1 for right
PATH_LOSS_EXPONENT = 4
CHANNEL_FADING = 1

# Constants for calculations
BANDWIDTH = 10e6
NOISE_POWER = 1
PROCESSING_COMPLEXITY = 1000
KAPPA = 1e-27
INPUT_DATA_SIZE = 16e6
V2R_BANDWIDTH = 5e6
V2V_BANDWIDTH = 3e6
BANDWIDTH_COST_RSU = 0.01
BANDWIDTH_COST_V2V = 0.005
POWER_LOCAL = 0.5
POWER_RSU = 0.2
POWER_V2V = 0.1

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

# Offloading counts dictionary for DDQN simulation
ddqn_offloading_counts = {"Local": 0, "RSU": 0, "V2V": 0}

# Randomly generate vehicle positions (x, y), velocity, and direction
true_positions = np.random.uniform(0, ROAD_LENGTH, size=(NUM_VEHICLES, 4))
true_positions[:, 2] = np.random.uniform(*VEHICLE_SPEED_RANGE, size=NUM_VEHICLES)  # Velocity
true_positions[:, 3] = np.random.choice(DIRECTION_CHOICES, size=NUM_VEHICLES)  # Direction

# Utility Methods
def compute_reward(delay, energy, alpha=0.5, beta=0.5):
    # Normalize delay and energy to avoid very small rewards
    normalized_delay = delay / 100  # Adjust scaling factor as needed
    normalized_energy = energy / 1000  # Adjust scaling factor as needed
    cost = alpha * normalized_delay + beta * normalized_energy
    reward = -cost  # Negative cost as reward (minimize cost)
    return reward

def calculate_transmission_rate(P, d):
    d = max(d, 1e-9)
    SNR = (P * (d ** (-PATH_LOSS_EXPONENT)) * (np.abs(CHANNEL_FADING)) ** 2) / NOISE_POWER
    transmission_rate = BANDWIDTH * np.log2(1 + SNR)
    return max(transmission_rate, 1e-6)

def calculate_local_delay(f_loc, task_size):
    return (PROCESSING_COMPLEXITY * task_size) / f_loc

def calculate_local_energy(f_loc, task_size):
    P_CPU = KAPPA * (f_loc ** 2)
    return P_CPU * PROCESSING_COMPLEXITY * task_size

def select_best_vehicle_for_offloading(mv, available_vehicles):
    if not available_vehicles:
        return None
    selected_vehicle = min(
        available_vehicles,
        key=lambda v: ((mv.x - v.x) ** 2 + (mv.y - v.y) ** 2) ** 0.5 / v.cpu
    )
    return selected_vehicle

class MissionVehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.x, self.y = true_positions[vehicle_id, :2]
        self.velocity = true_positions[vehicle_id, 2]
        self.direction = true_positions[vehicle_id, 3]
        self.cpu = random.uniform(*VEHICLE_CPU_RANGE)
        self.task_size = random.uniform(*TASK_SIZE_RANGE)
        self.task_cycles = self.task_size * CPU_CYCLES_PER_MB
        self.vehicle_transmit_power = 0.1
        self.remaining_task_size = self.task_size

    def __repr__(self):
        return (f"Vehicle-{self.vehicle_id}: Loc({self.x:.2f},{self.y:.2f}), "
                f"Speed: {self.velocity:.2f} m/s, Dir: {self.direction}, "
                f"CPU: {self.cpu/1e9:.2f} GHz, Task: {self.task_size:.2f} MB")

class RoadsideUnit:
    def __init__(self, rsu_id):
        self.rsu_id = rsu_id
        self.x = random.uniform(0, ROAD_LENGTH)
        self.y = 0
        self.cpu = random.uniform(*RSU_CPU_RANGE)
        self.rsu_transmit_power = 0.2

    def __repr__(self):
        return f"RSU-{self.rsu_id}: Loc({self.x:.2f}, {self.y}), CPU: {self.cpu/1e9:.2f} GHz"

class DDQNAgent:
    def __init__(self, input_dim=10, action_dim=3):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.online_network = self.build_network()
        self.target_network = self.build_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.memory = deque(maxlen=1024)
        self.epsilon = 1.0
        self.update_target_network(hard=True)

    def build_network(self):
        model = tf.keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_dim, activation=None)
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def update_target_network(self, hard=False):
        if hard:
            self.target_network.set_weights(self.online_network.get_weights())
        else:
            online_weights = self.online_network.get_weights()
            target_weights = self.target_network.get_weights()
            new_weights = []
            for ow, tw in zip(online_weights, target_weights):
                new_weights.append(TAU * ow + (1 - TAU) * tw)
            self.target_network.set_weights(new_weights)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state = np.array([state])
        q_values = self.online_network(state)
        return np.argmax(q_values.numpy()[0])

    def train(self):
        if len(self.memory) < 32:
            return
        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for s, a, r, ns, d in batch:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        states = np.array(states)
        next_states = np.array(next_states)
        next_q_online = self.online_network(next_states)
        best_actions = np.argmax(next_q_online.numpy(), axis=1)
        next_q_target = self.target_network(next_states).numpy()
        target_q = []
        for i in range(len(batch)):
            if dones[i]:
                target_q.append(rewards[i])
            else:
                target_q.append(rewards[i] + GAMMA * next_q_target[i][best_actions[i]])
        target_q = np.array(target_q)
        with tf.GradientTape() as tape:
            q_values = self.online_network(states)
            q_action = tf.reduce_sum(q_values * tf.one_hot(actions, self.action_dim), axis=1)
            loss = tf.reduce_mean(tf.square(target_q - q_action))
        grads = tape.gradient(loss, self.online_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_network.trainable_variables))
        self.update_target_network()

# Initialize Vehicles and RSUs
mission_vehicles = [MissionVehicle(i) for i in range(NUM_VEHICLES)]
rsus = [RoadsideUnit(i) for i in range(NUM_RSUS)]

# Initialize Simulation and DQN Agent
ddqn = DDQNAgent(STATE_SIZE, ACTION_SIZE)

def run_ddqn_simulation():
    global global_total_latency_local, global_total_latency_rsu, global_total_latency_v2v
    global global_total_energy_local, global_total_energy_rsu, global_total_energy_v2v
    global global_count_local, global_count_rsu, global_count_v2v

    ddqn_episode_rewards = []
    for episode in range(EPISODES):
        total_reward = 0
        for mv in mission_vehicles:
            while mv.remaining_task_size > 0:
                available_rsus = [r for r in rsus if abs(r.x - mv.x) <= COMMUNICATION_RANGE]
                available_vehicles = [v for v in mission_vehicles if v.vehicle_id != mv.vehicle_id and abs(v.x - mv.x) <= COMMUNICATION_RANGE]

                state = [
                    len(available_rsus),
                    len(available_vehicles),
                    mv.x, mv.y,
                    mv.cpu,
                    mv.remaining_task_size,
                    available_rsus[0].cpu if available_rsus else 0,
                    available_vehicles[0].cpu if available_vehicles else 0,
                    mv.velocity,
                    mv.direction
                ]

                action = ddqn.act(state)

                if action == 0:  # Local processing
                    executed_task_size = mv.remaining_task_size  # Execute the entire remaining task
                    local_delay = calculate_local_delay(mv.cpu, executed_task_size)
                    local_energy = calculate_local_energy(mv.cpu, executed_task_size)
                    reward = compute_reward(local_delay, local_energy, alpha=1, beta=0.5)
                    ddqn_offloading_counts["Local"] += 1
                    global_total_latency_local += local_delay
                    global_total_energy_local += local_energy
                    global_count_local += 1
                    mv.remaining_task_size = 0  # Task fully executed locally

                elif action == 1 and available_rsus:  # RSU Offloading
                    selected_rsu = min(rsus, key=lambda r: ((mv.x - r.x) ** 2 + (mv.y - r.y) ** 2) ** 0.5)
                    executed_task_size = mv.remaining_task_size  # Execute the entire remaining task
                    distance = np.sqrt((mv.x - selected_rsu.x) ** 2 + (mv.y - selected_rsu.y) ** 2)
                    transmission_rate = calculate_transmission_rate(mv.vehicle_transmit_power, distance)
                    delay_uplink = ALPHA * (executed_task_size / transmission_rate)
                    delay_execution = (PROCESSING_COMPLEXITY * executed_task_size) / selected_rsu.cpu
                    delay_downlink = BETA * (executed_task_size / transmission_rate)
                    rsu_delay = delay_uplink + delay_execution + delay_downlink
                    rsu_energy = (mv.vehicle_transmit_power * PROCESSING_COMPLEXITY * executed_task_size) / transmission_rate
                    reward = compute_reward(rsu_delay, rsu_energy, alpha=1, beta=0.5)
                    ddqn_offloading_counts["RSU"] += 1
                    global_total_latency_rsu += rsu_delay
                    global_total_energy_rsu += rsu_energy
                    global_count_rsu += 1
                    mv.remaining_task_size = 0  # Task fully executed by RSU

                elif action == 2 and available_vehicles:  # V2V Offloading
                    selected_vehicle = select_best_vehicle_for_offloading(mv, available_vehicles)
                    if selected_vehicle is None:
                        executed_task_size = mv.remaining_task_size  # Execute the entire remaining task
                        local_delay = calculate_local_delay(mv.cpu, executed_task_size)
                        local_energy = calculate_local_energy(mv.cpu, executed_task_size)
                        reward = compute_reward(local_delay, local_energy, alpha=1, beta=0.5)
                        ddqn_offloading_counts["Local"] += 1
                        global_total_latency_local += local_delay
                        global_total_energy_local += local_energy
                        global_count_local += 1
                        mv.remaining_task_size = 0  # Task fully executed locally
                    else:
                        executed_task_size = mv.remaining_task_size  # Execute the entire remaining task
                        distance = np.sqrt((mv.x - selected_vehicle.x) ** 2 + (mv.y - selected_vehicle.y) ** 2)
                        transmission_rate = calculate_transmission_rate(mv.vehicle_transmit_power, distance)
                        delay_uplink = ALPHA * (executed_task_size / transmission_rate)
                        delay_execution = (PROCESSING_COMPLEXITY * executed_task_size) / selected_vehicle.cpu
                        delay_downlink = BETA * (executed_task_size / transmission_rate)
                        total_v2v_delay = delay_uplink + delay_execution + delay_downlink
                        total_v2v_energy = (mv.vehicle_transmit_power * PROCESSING_COMPLEXITY * executed_task_size) / transmission_rate
                        reward = compute_reward(total_v2v_delay, total_v2v_energy)
                        ddqn_offloading_counts["V2V"] += 1
                        global_total_latency_v2v += total_v2v_delay
                        global_total_energy_v2v += total_v2v_energy
                        global_count_v2v += 1
                        mv.remaining_task_size = 0  # Task fully executed by V2V

                else:  # Default to local processing
                    executed_task_size = mv.remaining_task_size  # Execute the entire remaining task
                    local_delay = calculate_local_delay(mv.cpu, executed_task_size)
                    local_energy = calculate_local_energy(mv.cpu, executed_task_size)
                    reward = compute_reward(local_delay, local_energy, alpha=1, beta=0.5)
                    ddqn_offloading_counts["Local"] += 1
                    global_total_latency_local += local_delay
                    global_total_energy_local += local_energy
                    global_count_local += 1
                    mv.remaining_task_size = 0  # Task fully executed locally

                total_reward += reward
                next_state = state
                ddqn.store_experience(state, action, reward, next_state, done=False)
        if ddqn.epsilon > EPSILON_MIN:
            ddqn.epsilon = max(ddqn.epsilon * EPSILON_DECAY, EPSILON_MIN)
        ddqn_episode_rewards.append(total_reward)
        ddqn.train()
        if episode % 50 == 0:
            print(f"DDQN Episode {episode} - Total Reward: {total_reward:.2f}")
    return ddqn, ddqn_episode_rewards
    

# Visualization functions (you can reuse the same functions as before)
def plot_training_rewards(episode_rewards):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(episode_rewards)), episode_rewards, marker='o', linestyle='-', label="Training Reward")
    plt.title("DDQN Training: Reward Over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_offloading_distribution(offloading_counts):
    labels = list(offloading_counts.keys())
    sizes = list(offloading_counts.values())
    plt.figure(figsize=(8,8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['gold', 'lightblue', 'lightgreen'])
    plt.title("DDQN Offloading Decision Distribution")
    plt.show()

def plot_latency_distribution():
    avg_latency_local = global_total_latency_local / global_count_local if global_count_local > 0 else 0
    avg_latency_rsu = global_total_latency_rsu / global_count_rsu if global_count_rsu > 0 else 0
    avg_latency_v2v = global_total_latency_v2v / global_count_v2v if global_count_v2v > 0 else 0
    plt.figure(figsize=(8,6))
    plt.bar(['Local','RSU','V2V'], [avg_latency_local, avg_latency_rsu, avg_latency_v2v],
            color=['gold','lightblue','lightgreen'])
    plt.title("DDQN Average Latency by Offloading Option")
    plt.ylabel("Latency (s)")
    plt.show()

def plot_energy_consumption():
    print(f"Global Local Count{global_count_local}")
    avg_energy_local = global_total_energy_local / global_count_local if global_count_local > 0 else 0
    avg_energy_rsu = global_total_energy_rsu / global_count_rsu if global_count_rsu > 0 else 0
    avg_energy_v2v = global_total_energy_v2v / global_count_v2v if global_count_v2v > 0 else 0
    plt.figure(figsize=(8,6))
    plt.bar(['Local','RSU','V2V'], [avg_energy_local, avg_energy_rsu, avg_energy_v2v],
            color=['gold','lightblue','lightgreen'])
    plt.title("DDQN Average Energy Consumption by Offloading Option")
    plt.ylabel("Energy (Joules)")
    plt.show()

# Run DDQN simulation and plot results
ddqn_agent, ddqn_episode_rewards = run_ddqn_simulation()
plot_training_rewards(ddqn_episode_rewards)
plot_offloading_distribution(ddqn_offloading_counts)
plot_latency_distribution()
plot_energy_consumption()

