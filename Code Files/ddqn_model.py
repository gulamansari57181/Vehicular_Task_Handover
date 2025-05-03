import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sns


# Simulation Parameters
NUM_VEHICLES = 80
NUM_RSUS = 30
STATE_SIZE = 11
ACTION_SIZE = 3  
GAMMA = 0.9  
EPSILON = 0.9  
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 0.05
TAU = 0.01 
MEMORY_SIZE = 1024
EPISODES = 300
ALPHA=1
BETA = 0.5  
COMMUNICATION_RANGE = 200  
ROAD_LENGTH = 10000  
VEHICLE_SPEED_RANGE = (30, 120)  
DIRECTION_CHOICES = [-1, 1]  


EPISODES=600


# Constants for initialization
VEHICLE_CPU_RANGE = (2e9, 8e9)  # CPU frequency range in Hz (2GHz to 8GHz)
RSU_CPU_RANGE = (8e9, 16e9)  # CPU frequency range in Hz (8GHz to 16GHz)
TASK_SIZE_RANGE = (5e6, 16e6)  # Task size in Mbits
CPU_CYCLES_PER_MB = 1000*10e6  # Required CPU cycles per Mbits
ROAD_LENGTH = 10000  # Road length in meters
VEHICLE_SPEED_RANGE = (30, 120)  # Speed range in km/hr
COMMUNICATION_RANGE=200 # Communication range in meters
DIRECTION_CHOICES = [-1, 1]  # -1 for left, 1 for right
PATH_LOSS_EXPONENT=4
CHANNEL_FADING=1



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
training_losses = []  # Stores loss values
q_value_history = []  # Stores Q-values to monitor learning



# Offloading counts dictionary for DDQN simulation
ddqn_offloading_counts = {"Local": 0, "RSU": 0, "V2V": 0,"Local+RSU":0,"Local+V2V":0,"RSU+V2V":0,"All":0}

# ================== Parameters for Partial Offloading ==================
PARTIAL_OFFLOAD_RATIOS = [0.2, 0.5, 0.8]  # Allowed splitting ratios (20%, 50%, 80%)
ACTION_SIZE = 7  # Local, RSU, V2V, Local+RSU, Local+V2V, RSU+V2V, All
ACTION_MEANINGS = {
    0: "Local",
    1: "RSU", 
    2: "V2V",
    3: "Local+RSU",
    4: "Local+V2V",
    5: "RSU+V2V",
    6: "All"
}

# Randomly generate vehicle positions (x, y), velocity, and direction
true_positions = np.random.uniform(0, ROAD_LENGTH, size=(NUM_VEHICLES, 4))
true_positions[:, 2] = np.random.uniform(*VEHICLE_SPEED_RANGE, size=NUM_VEHICLES)  # Velocity
true_positions[:, 3] = np.random.choice(DIRECTION_CHOICES, size=NUM_VEHICLES)  # Direction



# Utilit Methods 

def compute_reward(delay, energy, task_complexity):
    """
    Computes a dynamic reward function with adaptive weights based on task complexity.
    """
    alpha = 1 / (1 + np.exp(-task_complexity))  # Adaptive delay weight (Sigmoid scaling)
    beta = 1 - alpha  # Adaptive energy weight (Ensures sum is 1)
    
    cost = alpha * delay + beta * energy
    reward = 1 / cost if cost > 0 else 0  # Avoid division by zero
    
    return reward



def calculate_transmission_rate( P, d):
        """
        Calculate the data transmission rate using the Shannon-Hartley theorem.

        Parameters:
            P (float): Transmit power (in Watts).
            d (float): Distance between transmitter and receiver (in meters).

        Returns:
            float: Data transmission rate (in bits per second).
        """
        # Avoid division by zero by ensuring d is not zero
        d = max(d, 1e-9)

        # Calculate the signal-to-noise ratio (SNR)
        SNR = (P * (d ** (-PATH_LOSS_EXPONENT)) * (np.abs(CHANNEL_FADING) ** 2)) / NOISE_POWER

        # Calculate the transmission rate using the Shannon-Hartley theorem
        transmission_rate = BANDWIDTH* np.log2(1 + SNR)

        return max(transmission_rate, 1e-6)



def calculate_local_delay(f_loc,task_size):
        """
        Calculate the local computing delay.

        Parameters:
            f_loc (float): CPU frequency of the target vehicle (in Hz).

        Returns:
            float: Local computing delay (in seconds).
        """
        return (PROCESSING_COMPLEXITY * task_size) / f_loc

def calculate_local_energy(f_loc,task_size):
        """
        Calculate the local energy consumption.

        Parameters:
            f_loc (float): CPU frequency of the target vehicle (in Hz).

        Returns:
            float: Local energy consumption (in Joules).
        """
        P_CPU = KAPPA* (f_loc ** 2)  # Power consumption per CPU cycle
        return P_CPU * PROCESSING_COMPLEXITY* task_size





def select_best_vehicle_for_offloading(mv, available_vehicles):
    """
    Select the best cooperative vehicle based on CPU power and distance.
    """
    if not available_vehicles:
        return None  # No vehicle available for offloading
    
    # Select the vehicle with best CPU power within the shortest distance
    selected_vehicle = min(
        available_vehicles, 
        key=lambda v: ((mv.x - v.x) ** 2 + (mv.y - v.y) ** 2) ** 0.5 / v.cpu
    )
    return selected_vehicle

# ================== NEW: Offloading Calculation Helpers ==================
def calculate_rsu_offload(mv, rsu, task_size):
    """
    Calculates delay and energy for RSU offloading .
    Args:
        mv: MissionVehicle object
        rsu: Selected RoadsideUnit object
        task_size: Size of task portion being offloaded (in MB)
    Returns:
        (total_delay, total_energy)
    """
    # Distance calculation
    distance = np.sqrt((mv.x - rsu.x)**2 + (mv.y - rsu.y)**2)
    
    # Transmission rates 
    uplink_rate = calculate_transmission_rate(mv.vehicle_transmit_power, distance)
    downlink_rate = calculate_transmission_rate(rsu.rsu_transmit_power, distance)
    
    # Delays 
    delay_uplink = ALPHA * (task_size / uplink_rate)
    delay_edge = (PROCESSING_COMPLEXITY * task_size) / rsu.cpu
    delay_downlink = BETA * (task_size / downlink_rate)
    
    # Energy
    energy = (mv.vehicle_transmit_power * PROCESSING_COMPLEXITY * task_size) / uplink_rate
    
    return (delay_edge + delay_uplink + delay_downlink, energy)

def calculate_v2v_offload(mv, vehicle, task_size):
    """
    Calculates delay and energy for V2V offloading .
    Args:
        mv: MissionVehicle object
        vehicle: Selected neighboring vehicle
        task_size: Size of task portion being offloaded (in MB)
    Returns:
        (total_delay, total_energy)
    """
    # Distance calculation
    distance = np.sqrt((mv.x - vehicle.x)**2 + (mv.y - vehicle.y)**2)
    
    # Transmission rates
    uplink_rate = calculate_transmission_rate(mv.vehicle_transmit_power, distance)
    downlink_rate = calculate_transmission_rate(vehicle.vehicle_transmit_power, distance)
    
    # Delays (with ALPHA/BETA weights)
    delay_uplink = ALPHA * (task_size / uplink_rate)
    delay_execution = (PROCESSING_COMPLEXITY * task_size) / vehicle.cpu
    delay_downlink = BETA * (task_size / downlink_rate)
    
    # Energy 
    energy = (mv.vehicle_transmit_power * PROCESSING_COMPLEXITY * task_size) / uplink_rate
    
    return (delay_uplink + delay_execution + delay_downlink, energy)


class MissionVehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.x, self.y = true_positions[vehicle_id, :2]  # Location (x, y)
        self.velocity = true_positions[vehicle_id, 2]  # Speed
        self.direction = true_positions[vehicle_id, 3]  # Moving direction (-1 or 1)
        self.cpu = random.uniform(*VEHICLE_CPU_RANGE)  # Random CPU frequency
        self.task_size = random.uniform(*TASK_SIZE_RANGE)  # Task size in MB
        self.remaining_task_size = self.task_size
        self.task_cycles = self.task_size * CPU_CYCLES_PER_MB  # Total CPU cycles required
        self.vehicle_transmit_power = 0.1  # Transmit power of vehicles in Watts
        self.remaining_task_size = self.task_size  # NEW: Track remaining task
        self.task_cycles = self.task_size * CPU_CYCLES_PER_MB
        self.vehicle_transmit_power = 0.1
    
    #Split task into local and offloaded portions
    def split_task(self, ratio):
        offloaded_size = self.task_size * ratio
        local_size = self.task_size - offloaded_size
        return local_size, offloaded_size

    def __repr__(self):
        return (f"Vehicle-{self.vehicle_id}: Loc({self.x:.2f},{self.y:.2f}), "
                f"Speed: {self.velocity:.2f} m/s, Dir: {self.direction}, "
                f"CPU: {self.cpu/1e9:.2f} GHz, Task: {self.task_size:.2f} MB")
    
    

class RoadsideUnit:
    def __init__(self, rsu_id):
        self.rsu_id = rsu_id
        self.x = random.uniform(0, ROAD_LENGTH)  # Random position along the road
        self.y = 0  # Assume RSUs are placed along the road
        self.cpu = random.uniform(*RSU_CPU_RANGE)  # Random CPU frequency for RSU
        self.rsu_transmit_power = 0.2 
    def __repr__(self):
        return f"RSU-{self.rsu_id}: Loc({self.x:.2f}, {self.y}), CPU: {self.cpu/1e9:.2f} GHz"

    
# DQN Model
class DDQNAgent:
    def __init__(self, input_dim=10, action_dim=ACTION_SIZE):
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
        q_values_per_episode = []  # Store max Q-values per episode
        executed_task_mbits_per_episode = []  # Store executed task in Mbits per episode
        
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
        
        # Store Loss and Max Q-Value
        training_losses.append(loss.numpy())
        q_value_history.append(np.max(next_q_online.numpy()))



# Initialize Vehicles and RSUs
mission_vehicles = [MissionVehicle(i) for i in range(NUM_VEHICLES)]
rsus = [RoadsideUnit(i) for i in range(NUM_RSUS)]

# Initialize Simulation and DQN Agent
ddqn= DDQNAgent(STATE_SIZE, ACTION_SIZE)


    
def run_ddqn_simulation():
    # Global metrics for full offloading
    global global_total_latency_local, global_total_latency_rsu, global_total_latency_v2v
    global global_total_energy_local, global_total_energy_rsu, global_total_energy_v2v
    global global_count_local, global_count_rsu, global_count_v2v
    
    # New global metrics for partial offloading components
    global global_partial_local_latency, global_partial_local_energy
    global global_partial_rsu_latency, global_partial_rsu_energy
    global global_partial_v2v_latency, global_partial_v2v_energy
    global global_partial_all_local_latency, global_partial_all_local_energy
    global global_partial_all_rsu_latency, global_partial_all_rsu_energy
    global global_partial_all_v2v_latency, global_partial_all_v2v_energy

    ACTION_MEANINGS = {
        0: "Local",
        1: "RSU", 
        2: "V2V",
        3: "Local+RSU",
        4: "Local+V2V",
        5: "RSU+V2V",
        6: "All"
    }

    # Initialize all global variables
    global_total_latency_local = global_total_latency_rsu = global_total_latency_v2v = 0
    global_total_energy_local = global_total_energy_rsu = global_total_energy_v2v = 0
    global_count_local = global_count_rsu = global_count_v2v = 0
    
    # Initialize partial offloading metrics
    global_partial_local_latency = global_partial_local_energy = 0
    global_partial_rsu_latency = global_partial_rsu_energy = 0
    global_partial_v2v_latency = global_partial_v2v_energy = 0
    global_partial_all_local_latency = global_partial_all_local_energy = 0
    global_partial_all_rsu_latency = global_partial_all_rsu_energy = 0
    global_partial_all_v2v_latency = global_partial_all_v2v_energy = 0

    ddqn_episode_rewards = []
    for episode in range(EPISODES):
        total_reward = 0
        for mv in mission_vehicles:
            # Initialize all metrics
            local_delay = local_energy = 0
            rsu_delay = rsu_energy = 0
            v2v_delay = v2v_energy = 0
            
            # Get available resources
            available_rsus = [r for r in rsus if abs(r.x - mv.x) <= COMMUNICATION_RANGE]
            available_vehicles = [v for v in mission_vehicles 
                               if v.vehicle_id != mv.vehicle_id 
                               and abs(v.x - mv.x) <= COMMUNICATION_RANGE]

            state = [
                mv.remaining_task_size / mv.task_size,
                len(available_rsus),
                len(available_vehicles),
                mv.x, mv.y,
                mv.cpu,
                mv.task_size,
                available_rsus[0].cpu if available_rsus else 0,
                available_vehicles[0].cpu if available_vehicles else 0,
                mv.velocity,
                mv.direction
            ]

            action = ddqn.act(state)
            action_type = ACTION_MEANINGS[action]

            selected_rsu = min(available_rsus, key=lambda r: abs(r.x - mv.x)) if available_rsus else None
            selected_vehicle = select_best_vehicle_for_offloading(mv, available_vehicles) if available_vehicles else None

            if action_type == "Local":
                local_delay = calculate_local_delay(mv.cpu, mv.task_size)
                local_energy = calculate_local_energy(mv.cpu, mv.task_size)
                total_delay = local_delay
                total_energy = local_energy
                ddqn_offloading_counts["Local"] += 1
                global_count_local += 1

            elif action_type == "RSU" and selected_rsu:
                rsu_delay, rsu_energy = calculate_rsu_offload(mv, selected_rsu, mv.task_size)
                total_delay = rsu_delay
                total_energy = rsu_energy
                ddqn_offloading_counts["RSU"] += 1
                global_count_rsu += 1

            elif action_type == "V2V" and selected_vehicle:
                v2v_delay, v2v_energy = calculate_v2v_offload(mv, selected_vehicle, mv.task_size)
                total_delay = v2v_delay
                total_energy = v2v_energy
                ddqn_offloading_counts["V2V"] += 1
                global_count_v2v += 1

            elif action_type == "Local+RSU" and selected_rsu:
                local_size, rsu_size = mv.split_task(0.5)
                local_delay = calculate_local_delay(mv.cpu, local_size)
                rsu_delay, rsu_energy = calculate_rsu_offload(mv, selected_rsu, rsu_size)
                total_delay = local_delay + rsu_delay
                total_energy = calculate_local_energy(mv.cpu, local_size) + rsu_energy
                ddqn_offloading_counts["Local+RSU"] += 1
                
                # Track partial components separately
                global_partial_local_latency += local_delay
                global_partial_local_energy += calculate_local_energy(mv.cpu, local_size)
                global_partial_rsu_latency += rsu_delay
                global_partial_rsu_energy += rsu_energy
                
                # Update effective counts
                global_count_local += 0.5
                global_count_rsu += 0.5

            elif action_type == "Local+V2V" and selected_vehicle:
                local_size, v2v_size = mv.split_task(0.5)
                local_delay = calculate_local_delay(mv.cpu, local_size)
                v2v_delay, v2v_energy = calculate_v2v_offload(mv, selected_vehicle, v2v_size)
                total_delay = local_delay + v2v_delay
                total_energy = calculate_local_energy(mv.cpu, local_size) + v2v_energy
                ddqn_offloading_counts["Local+V2V"] += 1
                
                # Track partial components
                global_partial_local_latency += local_delay
                global_partial_local_energy += calculate_local_energy(mv.cpu, local_size)
                global_partial_v2v_latency += v2v_delay
                global_partial_v2v_energy += v2v_energy
                
                global_count_local += 0.5
                global_count_v2v += 0.5

            elif action_type == "RSU+V2V" and selected_rsu and selected_vehicle:
                rsu_size, v2v_size = mv.split_task(0.5)
                rsu_delay, rsu_energy = calculate_rsu_offload(mv, selected_rsu, rsu_size)
                v2v_delay, v2v_energy = calculate_v2v_offload(mv, selected_vehicle, v2v_size)
                total_delay = rsu_delay + v2v_delay
                total_energy = rsu_energy + v2v_energy
                ddqn_offloading_counts["RSU+V2V"] += 1
                
                # Track partial components
                global_partial_rsu_latency += rsu_delay
                global_partial_rsu_energy += rsu_energy
                global_partial_v2v_latency += v2v_delay
                global_partial_v2v_energy += v2v_energy
                
                global_count_rsu += 0.5
                global_count_v2v += 0.5

            elif action_type == "All" and selected_rsu and selected_vehicle:
                local_size, offload_size = mv.split_task(0.33)
                rsu_size = v2v_size = offload_size / 2
                local_delay = calculate_local_delay(mv.cpu, local_size)
                rsu_delay, rsu_energy = calculate_rsu_offload(mv, selected_rsu, rsu_size)
                v2v_delay, v2v_energy = calculate_v2v_offload(mv, selected_vehicle, v2v_size)
                total_delay = local_delay + rsu_delay + v2v_delay
                total_energy = (calculate_local_energy(mv.cpu, local_size) + 
                              rsu_energy + v2v_energy)
                ddqn_offloading_counts["All"] += 1
                
                # Track all partial components
                global_partial_all_local_latency += local_delay
                global_partial_all_local_energy += calculate_local_energy(mv.cpu, local_size)
                global_partial_all_rsu_latency += rsu_delay
                global_partial_all_rsu_energy += rsu_energy
                global_partial_all_v2v_latency += v2v_delay
                global_partial_all_v2v_energy += v2v_energy
                
                global_count_local += 0.33
                global_count_rsu += 0.33
                global_count_v2v += 0.33

            else:  # Fallback to local processing
                local_delay = calculate_local_delay(mv.cpu, mv.task_size)
                local_energy = calculate_local_energy(mv.cpu, mv.task_size)
                total_delay = local_delay
                total_energy = local_energy
                ddqn_offloading_counts["Local"] += 1
                global_count_local += 1

            # Compute reward and update task
            reward = compute_reward(total_delay, total_energy, mv.task_cycles)
            mv.remaining_task_size = 0
            total_reward += reward

            # Store experience
            next_state = state
            ddqn.store_experience(state, action, reward, next_state, done=False)

            # Update global metrics (now safe due to initialization)
            global_total_latency_local += local_delay
            global_total_energy_local += local_energy
            global_total_latency_rsu += rsu_delay
            global_total_energy_rsu += rsu_energy
            global_total_latency_v2v += v2v_delay
            global_total_energy_v2v += v2v_energy

        # Epsilon decay and training
        if ddqn.epsilon > EPSILON_MIN:
            ddqn.epsilon = max(ddqn.epsilon * EPSILON_DECAY, EPSILON_MIN)
        ddqn.train()
        ddqn_episode_rewards.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}: Reward = {total_reward:.2f}")

    return ddqn, ddqn_episode_rewards


# Plotting Logic

def plot_training_rewards(episode_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(episode_rewards)), episode_rewards, marker='o', linestyle='-', color='dodgerblue', linewidth=2)
    plt.title("DDQN Training: Reward Over Episodes", fontsize=14, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True)
    plt.show()

# Enhanced plotting functions for partial offloading analysis
def plot_offloading_distribution(offloading_counts):
    # Categorize decisions
    categories = {
        'Full Local': ['Local'],
        'Full RSU': ['RSU'], 
        'Full V2V': ['V2V'],
        'Partial Local+RSU': ['Local+RSU'],
        'Partial Local+V2V': ['Local+V2V'],
        'Partial RSU+V2V': ['RSU+V2V'],
        'Partial All': ['All']
    }
    
    # Prepare data
    labels = []
    sizes = []
    colors = []
    color_palette = ['#FFD700', '#87CEFA', '#32CD32', '#FFA07A', '#98FB98', '#FF6347', '#9370DB']
    
    for i, (label, keys) in enumerate(categories.items()):
        count = sum(offloading_counts.get(k, 0) for k in keys)
        if count > 0:
            labels.append(label)
            sizes.append(count)
            colors.append(color_palette[i])
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Pie chart
    plt.subplot(1, 2, 1)
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=140, textprops={'fontsize': 10})
    plt.title("Offloading Decision Distribution", fontsize=12)
    
    # Bar chart
    plt.subplot(1, 2, 2)
    plt.bar(labels, sizes, color=colors)
    plt.title("Offloading Decision Counts", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.show()

def plot_latency_distribution():
    # Calculate effective counts (accounting for partial offloads)
    effective_local = global_count_local + \
                    ddqn_offloading_counts.get('Local+RSU', 0)*0.5 + \
                    ddqn_offloading_counts.get('Local+V2V', 0)*0.5 + \
                    ddqn_offloading_counts.get('All', 0)*0.33
    
    effective_rsu = global_count_rsu + \
                   ddqn_offloading_counts.get('Local+RSU', 0)*0.5 + \
                   ddqn_offloading_counts.get('RSU+V2V', 0)*0.5 + \
                   ddqn_offloading_counts.get('All', 0)*0.33
    
    effective_v2v = global_count_v2v + \
                   ddqn_offloading_counts.get('Local+V2V', 0)*0.5 + \
                   ddqn_offloading_counts.get('RSU+V2V', 0)*0.5 + \
                   ddqn_offloading_counts.get('All', 0)*0.33
    
    # Calculate weighted averages
    avg_latency = {
        'Local': global_total_latency_local/max(effective_local, 1),
        'RSU': global_total_latency_rsu/max(effective_rsu, 1),
        'V2V': global_total_latency_v2v/max(effective_v2v, 1)
    }
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_latency.keys(), avg_latency.values(), 
                  color=['#FFD700', '#87CEFA', '#32CD32'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom')
    
    plt.title("Average Latency by Resource Type", fontsize=14)
    plt.ylabel("Latency (seconds)")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def plot_energy_consumption():
    # Calculate effective counts (same as latency)
    effective_local = global_count_local + \
                    ddqn_offloading_counts.get('Local+RSU', 0)*0.5 + \
                    ddqn_offloading_counts.get('Local+V2V', 0)*0.5 + \
                    ddqn_offloading_counts.get('All', 0)*0.33
    
    effective_rsu = global_count_rsu + \
                   ddqn_offloading_counts.get('Local+RSU', 0)*0.5 + \
                   ddqn_offloading_counts.get('RSU+V2V', 0)*0.5 + \
                   ddqn_offloading_counts.get('All', 0)*0.33
    
    effective_v2v = global_count_v2v + \
                   ddqn_offloading_counts.get('Local+V2V', 0)*0.5 + \
                   ddqn_offloading_counts.get('RSU+V2V', 0)*0.5 + \
                   ddqn_offloading_counts.get('All', 0)*0.33
    
    # Calculate weighted averages
    avg_energy = {
        'Local': global_total_energy_local/max(effective_local, 1),
        'RSU': global_total_energy_rsu/max(effective_rsu, 1),
        'V2V': global_total_energy_v2v/max(effective_v2v, 1)
    }
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_energy.keys(), avg_energy.values(),
                  color=['#FFD700', '#87CEFA', '#32CD32'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}J',
                ha='center', va='bottom')
    
    plt.title("Average Energy Consumption by Resource Type", fontsize=14)
    plt.ylabel("Energy (Joules)")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def plot_partial_offloading_analysis():
    # Extract partial offloading data
    partial_data = {
        'Local+RSU': ddqn_offloading_counts.get('Local+RSU', 0),
        'Local+V2V': ddqn_offloading_counts.get('Local+V2V', 0),
        'RSU+V2V': ddqn_offloading_counts.get('RSU+V2V', 0),
        'All': ddqn_offloading_counts.get('All', 0)
    }
    
    # Only plot if there's data
    if sum(partial_data.values()) > 0:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(partial_data.keys(), partial_data.values(),
                      color=['#FFA07A', '#98FB98', '#FF6347', '#9370DB'])
        
        plt.title("Partial Offloading Strategy Distribution", fontsize=14)
        plt.ylabel("Count")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}',
                    ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        plt.show()

# Run simulation and plot results
ddqn_agent, ddqn_episode_rewards = run_ddqn_simulation()

# Print effective counts (accounting for partial offloads)
print("\n=== Effective Task Distribution ===")
print(f"Local: {global_count_local + ddqn_offloading_counts.get('Local+RSU',0)*0.5 + ddqn_offloading_counts.get('Local+V2V',0)*0.5 + ddqn_offloading_counts.get('All',0)*0.33:.1f} tasks")
print(f"RSU: {global_count_rsu + ddqn_offloading_counts.get('Local+RSU',0)*0.5 + ddqn_offloading_counts.get('RSU+V2V',0)*0.5 + ddqn_offloading_counts.get('All',0)*0.33:.1f} tasks")
print(f"V2V: {global_count_v2v + ddqn_offloading_counts.get('Local+V2V',0)*0.5 + ddqn_offloading_counts.get('RSU+V2V',0)*0.5 + ddqn_offloading_counts.get('All',0)*0.33:.1f} tasks")

# Generate all plots
plot_training_rewards(ddqn_episode_rewards)
plot_offloading_distribution(ddqn_offloading_counts)
plot_latency_distribution()
plot_energy_consumption()
plot_partial_offloading_analysis()


