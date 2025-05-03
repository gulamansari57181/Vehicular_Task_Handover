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
STATE_SIZE = 10
ACTION_SIZE = 3  
GAMMA = 0.9  
EPSILON = 0.9  
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
ROAD_LENGTH = 10000  
VEHICLE_SPEED_RANGE = (30, 120)  
DIRECTION_CHOICES = [-1, 1]  


EPISODES=1000


# Constants for initialization
VEHICLE_CPU_RANGE = (2e9, 8e9)  # CPU frequency range in Hz (2GHz to 8GHz)
RSU_CPU_RANGE = (8e9, 16e9)  # CPU frequency range in Hz (8GHz to 16GHz)
TASK_SIZE_RANGE = (5e6, 16e6)  # Task size in Mbits
CPU_CYCLES_PER_MB = 1000  # Required CPU cycles per Mbits
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
ddqn_offloading_counts = {"Local": 0, "RSU": 0, "V2V": 0}

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



class MissionVehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.x, self.y = true_positions[vehicle_id, :2]  # Location (x, y)
        self.velocity = true_positions[vehicle_id, 2]  # Speed
        self.direction = true_positions[vehicle_id, 3]  # Moving direction (-1 or 1)
        self.cpu = random.uniform(*VEHICLE_CPU_RANGE)  # Random CPU frequency
        self.task_size = random.uniform(*TASK_SIZE_RANGE)  # Task size in MB
        self.task_deadline=random.uniform(0.5,8)  
        self.remaining_task_size = self.task_size
        self.task_cycles = self.task_size * CPU_CYCLES_PER_MB  # Total CPU cycles required
        self.vehicle_transmit_power = 0.1  # Transmit power of vehicles in Watts

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
    global global_total_latency_local, global_total_latency_rsu, global_total_latency_v2v
    global global_total_energy_local, global_total_energy_rsu, global_total_energy_v2v
    global global_count_local, global_count_rsu, global_count_v2v

    ddqn_episode_rewards = []
    for episode in range(EPISODES):
        total_reward = 0
        for mv in mission_vehicles:
            
            available_rsus = [r for r in rsus if abs(r.x - mv.x) <= COMMUNICATION_RANGE]
            available_vehicles = [v for v in mission_vehicles if v.vehicle_id != mv.vehicle_id and abs(v.x - mv.x) <= COMMUNICATION_RANGE]

            state = [
                    len(available_rsus),  # Number of RSUs in range
                    len(available_vehicles),  # Number of vehicles in range
                    mv.x, mv.y,  # Location
                    mv.cpu,  # CPU frequency of vehicle
                    mv.task_size,  # Remaining task size
                    available_rsus[0].cpu if available_rsus else 0,  # CPU of selected RSU
                    available_vehicles[0].cpu if available_vehicles else 0,  # CPU of selected vehicle
                    mv.velocity,  # Speed of the vehicle
                    mv.direction  # Moving direction
                ]

            action = ddqn.act(state)


            if action == 0:  # Local processing

                local_delay = calculate_local_delay(mv.cpu,mv.task_size)
                local_energy= calculate_local_energy(mv.cpu,mv.task_size)
                
                
                
                reward = compute_reward(local_delay, local_energy,mv.task_cycles)
                ddqn_offloading_counts["Local"] += 1
                global_total_latency_local += local_delay
                global_total_energy_local += local_energy
                global_count_local += 1


            elif action == 1 and available_rsus:  # RSU Offloading
                selected_rsu = min(rsus, key=lambda r: ((mv.x - r.x) ** 2 + (mv.y - r.y) ** 2) ** 0.5)
                delay_edge=(PROCESSING_COMPLEXITY * mv.task_size ) / selected_rsu.cpu
                distance=np.sqrt((mv.x - selected_rsu.x) ** 2 + (mv.y - selected_rsu.y) ** 2) 
                delay_uplink=ALPHA*(mv.task_size/calculate_transmission_rate( mv.vehicle_transmit_power, distance))
                delay_downlink=BETA*(mv.task_size/calculate_transmission_rate( selected_rsu.rsu_transmit_power, distance))
               
                rsu_delay = delay_edge + delay_uplink + delay_downlink
                rsu_energy=(mv.vehicle_transmit_power*PROCESSING_COMPLEXITY*mv.task_size)/calculate_transmission_rate( mv.vehicle_transmit_power, distance)
                reward = compute_reward(rsu_delay, rsu_energy,mv.task_cycles)
                ddqn_offloading_counts["RSU"] += 1
                
                global_total_latency_rsu += rsu_delay
                global_total_energy_rsu += rsu_energy
                global_count_rsu += 1
            elif action==2 and available_vehicles :  # V2V Offloading
                
                
                selected_vehicle = select_best_vehicle_for_offloading(mv, available_vehicles)
                if selected_vehicle is None:
                    local_delay = calculate_local_delay(mv.cpu,mv.task_size)
                    local_energy= calculate_local_energy(mv.cpu,mv.task_size)
                    reward = compute_reward(local_delay, local_energy,mv.task_cycles)
                    ddqn_offloading_counts["Local"] += 1
                    global_total_latency_local += local_delay
                    global_total_energy_local += local_energy
                    global_count_local += 1
                    
                else:
                    distance = np.sqrt((mv.x - selected_vehicle.x) ** 2 + (mv.y - selected_vehicle.y) ** 2)
                    delay_uplink = ALPHA*(mv.task_size / calculate_transmission_rate(mv.vehicle_transmit_power, distance))
                    delay_execution = (PROCESSING_COMPLEXITY * mv.task_size) / selected_vehicle.cpu
                    delay_downlink = BETA*(mv.task_size / calculate_transmission_rate(selected_vehicle.vehicle_transmit_power, distance))
                    total_v2v_delay = delay_uplink + delay_execution + delay_downlink
                    total_v2v_energy = (mv.vehicle_transmit_power * PROCESSING_COMPLEXITY * mv.task_size) /calculate_transmission_rate(mv.vehicle_transmit_power, distance)

                    # Compute Reward
                    reward = compute_reward(total_v2v_delay, total_v2v_energy,mv.task_cycles)

                    # Update global tracking variables
                    ddqn_offloading_counts["V2V"] += 1
                    global_total_latency_v2v += total_v2v_delay
                    global_total_energy_v2v += total_v2v_energy
                    global_count_v2v += 1   
            else: 
                local_delay = calculate_local_delay(mv.cpu,mv.task_size)
                local_energy= calculate_local_energy(mv.cpu,mv.task_size)
                
                
                reward = compute_reward(local_delay, local_energy,mv.task_cycles)
                ddqn_offloading_counts["Local"] += 1
                global_total_latency_local += local_delay
                global_total_energy_local += local_energy
                global_count_local += 1
                    
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




# Set a unified theme for all plots
sns.set_style("whitegrid")
colors = ['#FFD700', '#87CEFA', '#32CD32']  # Gold, Light Blue, Light Green

def plot_training_rewards(episode_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(episode_rewards)), episode_rewards, marker='o', linestyle='-', color='dodgerblue', linewidth=2)
    plt.title("DDQN Training: Reward Over Episodes", fontsize=14, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True)
    plt.show()

def plot_offloading_distribution(offloading_counts):
    labels = list(offloading_counts.keys())
    sizes = list(offloading_counts.values())
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 12})
    plt.title("DDQN Offloading Decision Distribution", fontsize=14, fontweight='bold')
    plt.show()

def plot_latency_distribution():
    avg_latency_local = global_total_latency_local / global_count_local if global_count_local > 0 else 0
    avg_latency_rsu = global_total_latency_rsu / global_count_rsu if global_count_rsu > 0 else 0
    avg_latency_v2v = global_total_latency_v2v / global_count_v2v if global_count_v2v > 0 else 0
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Local', 'RSU', 'V2V'], [avg_latency_local, avg_latency_rsu, avg_latency_v2v], color=colors)
    plt.title("DDQN Average Latency by Offloading Option", fontsize=14, fontweight='bold')
    plt.ylabel("Latency (s)", fontsize=12)
    
    if global_count_local + global_count_rsu + global_count_v2v == 0:
        plt.text(1, 0, "No Data Available", ha='center', fontsize=12, color='red')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_energy_consumption():
    avg_energy_local = global_total_energy_local / global_count_local if global_count_local > 0 else 0
    avg_energy_rsu = global_total_energy_rsu / global_count_rsu if global_count_rsu > 0 else 0
    avg_energy_v2v = global_total_energy_v2v / global_count_v2v if global_count_v2v > 0 else 0
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Local', 'RSU', 'V2V'], [avg_energy_local, avg_energy_rsu, avg_energy_v2v], color=colors)
    plt.title("DDQN Average Energy Consumption by Offloading Option", fontsize=14, fontweight='bold')
    plt.ylabel("Energy (Joules)", fontsize=12)
    
    if global_count_local + global_count_rsu + global_count_v2v == 0:
        plt.text(1, 0, "No Data Available", ha='center', fontsize=12, color='red')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_loss_curve():
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(training_losses)), training_losses, marker='o', linestyle='-', color='crimson', linewidth=2)
    plt.title("DDQN Training Loss Over Episodes", fontsize=14, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    plt.show()

def plot_q_value_evolution():
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(q_value_history)), q_value_history, marker='o', linestyle='-', color='seagreen', linewidth=2)
    plt.title("DDQN Q-Value Evolution Over Episodes", fontsize=14, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Max Q-Value", fontsize=12)
    plt.grid(True)
    plt.show()

# Run DDQN simulation and plot results
ddqn_agent, ddqn_episode_rewards = run_ddqn_simulation()

print(f"Local Offloads: {global_count_local}")
print(f"RSU Offloads: {global_count_rsu}")
print(f"V2V Offloads: {global_count_v2v}")

plot_training_rewards(ddqn_episode_rewards)
plot_offloading_distribution(ddqn_offloading_counts)
plot_latency_distribution()
plot_energy_consumption()
# plot_loss_curve()
# plot_q_value_evolution()

