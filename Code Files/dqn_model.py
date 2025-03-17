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
STATE_SIZE = 7  # Features: CPU freq, transmission rates, distance, task size
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


# Constants for initialization
VEHICLE_CPU_RANGE = (2e9, 8e9)  # CPU frequency range in Hz (2GHz to 8GHz)
RSU_CPU_RANGE = (8e9, 16e9)  # CPU frequency range in Hz (8GHz to 16GHz)
TASK_SIZE_RANGE = (5, 50)  # Task size in MB
CPU_CYCLES_PER_MB = 1000  # Required CPU cycles per MB
ROAD_LENGTH = 1000  # Road length in meters
VEHICLE_SPEED_RANGE = (10, 30)  # Speed range in m/s
DIRECTION_CHOICES = [-1, 1]  # -1 for left, 1 for right
NUM_VEHICLES = 80
NUM_RSUS = 30



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

    def __repr__(self):
        return f"RSU-{self.rsu_id}: Loc({self.x:.2f}, {self.y}), CPU: {self.cpu/1e9:.2f} GHz"









class SimulationParameters:
    def __init__(self):
        self.num_vehicles = NUM_VEHICLES
        self.num_rsus = NUM_RSUS
        
        self.route_length = 10  # Route length in km
        self.communication_range = 200  # Communication range in meters
        
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
        # Randomly generate vehicle speeds and CPU frequencies
        self.vehicle_speeds = np.random.uniform(
            self.vehicle_speed_range[0], self.vehicle_speed_range[1], self.num_vehicles
        )  # in km/h
        self.vehicle_cpu_freqs = np.random.uniform(
            self.vehicle_cpu_freq_range[0], self.vehicle_cpu_freq_range[1], self.num_vehicles
        )  # in GHz

    def initialize_rsus(self):
        # Randomly generate RSU CPU frequencies
        self.rsu_cpu_freqs = np.random.uniform(
            self.rsu_cpu_freq_range[0], self.rsu_cpu_freq_range[1], self.num_rsus
        )  # in GHz

    def initialize_distances(self):
        """
        Randomly generate distances between the target vehicle and edge servers (vehicles and RSUs).
        Distances are within the communication range (0 to 200 meters).
        """
        self.distances_VT_Vi = np.random.uniform(1e-9, self.communication_range, self.num_vehicles)  # Avoid zero distance
        self.distances_VT_Bj = np.random.uniform(1e-9, self.communication_range, self.num_rsus)  # Avoid zero distance


    def calculate_transmission_rate(self, P, d):
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
        SNR = (P * (d ** (-self.path_loss_exponent)) * (np.abs(self.channel_fading) ** 2)) / self.noise_power

        # Calculate the transmission rate using the Shannon-Hartley theorem
        transmission_rate = self.bandwidth * np.log2(1 + SNR)

        return transmission_rate

    def calculate_local_delay(self, f_loc):
        """
        Calculate the local computing delay.

        Parameters:
            f_loc (float): CPU frequency of the target vehicle (in Hz).

        Returns:
            float: Local computing delay (in seconds).
        """
        return (self.processing_complexity * self.input_data_size) / f_loc

    def calculate_local_energy(self, f_loc):
        """
        Calculate the local energy consumption.

        Parameters:
            f_loc (float): CPU frequency of the target vehicle (in Hz).

        Returns:
            float: Local energy consumption (in Joules).
        """
        P_CPU = self.kappa * (f_loc ** 2)  # Power consumption per CPU cycle
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
    # Not Used here currently
    
    def calculate_total_delay(self, mu_V, mu_B, mu_loc, n):
        """
        Calculate the total delay for computing the whole task.

        Parameters:
            mu_V (list): List of task fractions offloaded to vehicles.
            mu_B (list): List of task fractions offloaded to RSUs.
            mu_loc (float): Fraction of the task computed locally.
            n (int): Number of areas where the task is computed locally.

        Returns:
            float: Total delay (in seconds).
        """
        total_delay = 0

        # Calculate delay for vehicle edge servers
        for i in range(self.num_vehicles):
            R_VT_Vi = self.calculate_transmission_rate(self.vehicle_transmit_power, self.distances_VT_Vi[i])
            D_UL_Vi = self.alpha * (self.input_data_size / R_VT_Vi)
            D_edge_Vi = (self.processing_complexity * mu_V[i] * self.input_data_size) / self.vehicle_cpu_freqs[i]
            R_Vi_VT = self.calculate_transmission_rate(self.vehicle_transmit_power, self.distances_VT_Vi[i])
            D_DL_Vi = self.beta * (mu_V[i] * self.input_data_size / R_Vi_VT)
            total_delay += D_UL_Vi + D_edge_Vi + D_DL_Vi

        # Calculate delay for RSU edge servers
        for j in range(self.num_rsus):
            R_VT_Bj = self.calculate_transmission_rate(self.vehicle_transmit_power, self.distances_VT_Bj[j])
            D_UL_Bj = self.alpha * ((1 - sum(mu_V)) * self.input_data_size / R_VT_Bj)
            D_edge_Bj = (self.processing_complexity * mu_B[j] * self.input_data_size) / self.rsu_cpu_freqs[j]
            R_Bj_VT = self.calculate_transmission_rate(self.rsu_transmit_power, self.distances_VT_Bj[j])
            D_DL_Bj = self.beta * (mu_B[j] * self.input_data_size / R_Bj_VT)
            total_delay += D_UL_Bj + D_edge_Bj + D_DL_Bj

        # Calculate delay for local computing
        total_delay += n * (self.processing_complexity * mu_loc * self.input_data_size) / (2.5 * 1e9)  # Example: f_loc = 2.5 GHz

        return total_delay

    def calculate_total_energy(self, mu_V, mu_B, mu_loc, n):
        """
        Calculate the total energy consumption for computing the whole task.

        Parameters:
            mu_V (list): List of task fractions offloaded to vehicles.
            mu_B (list): List of task fractions offloaded to RSUs.
            mu_loc (float): Fraction of the task computed locally.
            n (int): Number of areas where the task is computed locally.

        Returns:
            float: Total energy consumption (in Joules).
        """
        total_energy = 0

        # Calculate energy for vehicle edge servers
        for i in range(self.num_vehicles):
            R_VT_Vi = self.calculate_transmission_rate(self.vehicle_transmit_power, self.distances_VT_Vi[i])
            total_energy += (self.vehicle_transmit_power * self.processing_complexity * mu_V[i] * self.input_data_size) / R_VT_Vi

        # Calculate energy for RSU edge servers
        for j in range(self.num_rsus):
            R_VT_Bj = self.calculate_transmission_rate(self.vehicle_transmit_power, self.distances_VT_Bj[j])
            total_energy += (self.vehicle_transmit_power * self.processing_complexity * mu_B[j] * self.input_data_size) / R_VT_Bj

        # Calculate energy for local computing
        total_energy += n * (self.kappa * (2.5 * 1e9) ** 2 * self.processing_complexity * mu_loc * self.input_data_size)  # Example: f_loc = 2.5 GHz

        return total_energy
 
 
    
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
agent = DDQNAgent(STATE_SIZE, ACTION_SIZE)

# Offloading counts dictionary for DDQN simulation
ddqn_offloading_counts = {"Local": 0, "RSU": 0, "V2V": 0}


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
    
    
    
def run_ddqn_simulation():
    global global_total_latency_local, global_total_latency_rsu, global_total_latency_v2v
    global global_total_energy_local, global_total_energy_rsu, global_total_energy_v2v
    global global_count_local, global_count_rsu, global_count_v2v

    ddqn_episode_rewards = []
    for episode in range(1000):
        total_reward = 0
        for mv in mission_vehicles:
            mv.update_position(predicted_positions_dict)
            # State vector
            state = [mv.task_size, mv.task_cycles, mv.x, mv.y, mv.velocity, mv.direction, mv.cpu, V2R_BANDWIDTH, V2V_BANDWIDTH]
            action = agent.act(state)
            comm_revenue = mv.task_size * 10
            comp_revenue = mv.task_cycles / 1000
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
