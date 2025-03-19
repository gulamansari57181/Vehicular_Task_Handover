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
STATE_SIZE = 8 # Features: CPU freq, transmission rates, distance, task size
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
ALPHA=1
BETA = 0.5  
COMMUNICATION_RANGE = 200  
ROAD_LENGTH = 1000  
VEHICLE_SPEED_RANGE = (30, 120)  
DIRECTION_CHOICES = [-1, 1]  


# Constants for initialization
VEHICLE_CPU_RANGE = (2e9, 8e9)  # CPU frequency range in Hz (2GHz to 8GHz)
RSU_CPU_RANGE = (8e9, 16e9)  # CPU frequency range in Hz (8GHz to 16GHz)
TASK_SIZE_RANGE = (5, 50)  # Task size in MB
CPU_CYCLES_PER_MB = 1000  # Required CPU cycles per MB
ROAD_LENGTH = 10000  # Road length in meters
VEHICLE_SPEED_RANGE = (30, 120)  # Speed range in km/hr
COMMUNICATION_RANGE=200 # Communication range in meters
DIRECTION_CHOICES = [-1, 1]  # -1 for left, 1 for right



# Constants for calculations
BANDWIDTH = 10e6  
NOISE_POWER = 1e-9  
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



class MissionVehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.x, self.y = true_positions[vehicle_id, :2]  # Location (x, y)
        self.velocity = true_positions[vehicle_id, 2]  # Speed
        self.direction = true_positions[vehicle_id, 3]  # Moving direction (-1 or 1)
        self.cpu = random.uniform(*VEHICLE_CPU_RANGE)  # Random CPU frequency
        self.task_size = random.uniform(*TASK_SIZE_RANGE)  # Task size in MB
        self.task_cycles = self.task_size * CPU_CYCLES_PER_MB  # Total CPU cycles required
        self.vehicle_transmit_power = 0.1  # Transmit power of vehicles in Watts

    def __repr__(self):
        return (f"Vehicle-{self.vehicle_id}: Loc({self.x:.2f},{self.y:.2f}), "
                f"Speed: {self.velocity:.2f} m/s, Dir: {self.direction}, "
                f"CPU: {self.cpu/1e9:.2f} GHz, Task: {self.task_size:.2f} MB")
    
    
    # These functions need to be verified.
    def calculate_transmission_rate(self, P, d):
        d = max(d, 1e-9)
        SNR = (P * (d ** -4)) / NOISE_POWER
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

class RoadsideUnit:
    def __init__(self, rsu_id):
        self.rsu_id = rsu_id
        self.x = random.uniform(0, ROAD_LENGTH)  # Random position along the road
        self.y = 0  # Assume RSUs are placed along the road
        self.cpu_freq = random.uniform(*RSU_CPU_RANGE)  # Random CPU frequency for RSU
        self.rsu_transmit_power = 0.2 
    def __repr__(self):
        return f"RSU-{self.rsu_id}: Loc({self.x:.2f}, {self.y}), CPU: {self.cpu/1e9:.2f} GHz"

    
# DQN Model
class DDQNAgent:
    def __init__(self, input_dim=7, action_dim=3):
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
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation=None)
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self, hard=False):
        if hard:
            # Hard copy all weights from online to target network
            self.target_network.set_weights(self.online_network.get_weights())
        else:
            # Soft update using factor TAU
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
        # Compute Q-values from online network for next_states to select best action
        next_q_online = self.online_network(next_states)
        best_actions = np.argmax(next_q_online.numpy(), axis=1)
        # Evaluate Q-values of best actions using target network
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
        # Soft update target network
        self.update_target_network()

# Initialize Simulation and DQN Agent
# sim_params = SimulationParameters()

# Initialize Vehicles and RSUs
mission_vehicles = [MissionVehicle(i) for i in range(NUM_VEHICLES)]
rsus = [RoadsideUnit(i) for i in range(NUM_RSUS)]
ddqn= DDQNAgent(STATE_SIZE, ACTION_SIZE)



    
def run_ddqn_simulation():
    global global_total_latency_local, global_total_latency_rsu, global_total_latency_v2v
    global global_total_energy_local, global_total_energy_rsu, global_total_energy_v2v
    global global_count_local, global_count_rsu, global_count_v2v

    ddqn_episode_rewards = []
    for episode in range(1000):
        total_reward = 0
        for mv in mission_vehicles:
            
            available_rsus = [r for r in rsus if abs(r.x - mv.x) <= COMMUNICATION_RANGE]
            available_vehicles = [v for v in mission_vehicles if v.vehicle_id != mv.vehicle_id and abs(v.x - mv.x) <= COMMUNICATION_RANGE]

            state = [len(available_rsus), len(available_vehicles), mv.x, mv.y, mv.cpu, 
                 available_rsus[0].cpu if available_rsus else 0, mv.velocity, mv.direction]

            action = ddqn.act(state)

            if action == 0:  # Local processing
                delay = mv.task_cycles / mv.cpu
                reward = ALPHA * (comp_revenue - delay)
                ddqn_offloading_counts["Local"] += 1
                latency_local = delay
                energy_local = latency_local * POWER_LOCAL
                global_total_latency_local += latency_local
                global_total_energy_local += energy_local
                global_count_local += 1
            elif action == 1:  # RSU Offloading
                selected_rsu = min(rsus, key=lambda r: abs(mv.x - r.location))
                delay_comm = transmission_delay(mv.task_size, V2R_BANDWIDTH)
                delay_comp = mv.task_cycles / selected_rsu.cpu
                cost = BANDWIDTH_COST_RSU * V2R_BANDWIDTH
                reward = ALPHA * (comm_revenue - delay_comm) + BETA * (comp_revenue - delay_comp) - cost
                ddqn_offloading_counts["RSU"] += 1
                latency_rsu = delay_comm + delay_comp
                energy_rsu = latency_rsu * POWER_RSU
                global_total_latency_rsu += latency_rsu
                global_total_energy_rsu += energy_rsu
                global_count_rsu += 1
            else:  # V2V Offloading
                selected_cv = select_cooperative_vehicle(mv, cooperative_vehicles, model_lat, model_lon)
                if selected_cv is None:
                    latency = mv.task_cycles / mv.cpu
                    energy = latency * POWER_LOCAL
                else:
                    delay_comm = transmission_delay(mv.task_size, V2V_BANDWIDTH)
                    delay_comp = mv.task_cycles / selected_cv.cpu
                    cost = BANDWIDTH_COST_V2V * V2V_BANDWIDTH
                    reward = ALPHA * (comm_revenue - delay_comm) + BETA * (comp_revenue - delay_comp) - cost
                    ddqn_offloading_counts["V2V"] += 1
                    latency_v2v = delay_comm + delay_comp
                    energy_v2v = latency_v2v * POWER_V2V
                    global_total_latency_v2v += latency_v2v
                    global_total_energy_v2v += energy_v2v
                    global_count_v2v += 1
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

plt.plot(range(EPISODES), rewards_per_episode, color='blue')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress')
plt.grid(True)
plt.show()
