import numpy as np
import matplotlib.pyplot as plt

class SimulationParameters:
    def __init__(self):
        # Environment Parameters
        self.num_vehicles = 80  # Number of vehicles
        self.num_rsus = 30  # Number of RSUs
        self.route_length = 10  # Route length in km
        self.communication_range = 200  # Communication range in meters

        # Vehicle Parameters
        self.vehicle_speed_range = [30, 120]  # Speed range in km/h
        self.vehicle_cpu_freq_range = [2, 8]  # CPU frequency range in GHz
        self.vehicle_transmit_power = 0.1  # Transmit power of vehicles in Watts

        # RSU Parameters
        self.rsu_cpu_freq_range = [8, 16]  # CPU frequency range in GHz
        self.rsu_transmit_power = 0.2  # Transmit power of RSUs in Watts

        # Task Parameters
        self.input_data_size = 16 * 1e6  # Input data size in bits (16 Mbits)
        self.processing_complexity = 1000  # Processing complexity in cycles/bit
        self.kappa = 1e-27  # Power consumption coefficient

        # Communication Parameters
        self.bandwidth = 10e6  # Bandwidth in Hz (10 MHz)
        self.noise_power = 1  # Noise power in Watts
        self.path_loss_exponent = 4  # Path loss exponent (updated to 4)
        self.channel_fading = 1.0  # Fading coefficient (updated to 1.0)
        self.alpha = 1  # Uplink transmission overhead coefficient
        self.beta = 0.01  # Downlink transmission overhead coefficient

        # DDQN Algorithm Parameters
        self.memory_size = 1024  # Memory size for experience replay
        self.batch_size = 32  # Batch size for training
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.9  # Exploration rate
        self.learning_rate = 0.05  # Learning rate

        # Initialize vehicle and RSU properties
        self.initialize_vehicles()
        self.initialize_rsus()

        # Initialize distances between target vehicle and edge servers
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

    def calculate_edge_delay(self, f_edge):
        """
        Calculate the edge computing delay.

        Parameters:
            f_edge (float): CPU frequency of the selected edge server (in Hz).

        Returns:
            float: Edge computing delay (in seconds).
        """
        return (self.processing_complexity * self.input_data_size) / f_edge

    def calculate_ul_delay(self, transmission_rate):
        """
        Calculate the uplink (UL) transmission delay.

        Parameters:
            transmission_rate (float): Transmission rate (in bits per second).

        Returns:
            float: Uplink transmission delay (in seconds).
        """
        return self.alpha * (self.input_data_size / transmission_rate)

    def calculate_dl_delay(self, transmission_rate):
        """
        Calculate the downlink (DL) transmission delay.

        Parameters:
            transmission_rate (float): Transmission rate (in bits per second).

        Returns:
            float: Downlink transmission delay (in seconds).
        """
        return self.beta * (self.input_data_size / transmission_rate)

    def calculate_total_offloading_delay(self, D_UL, D_edge, D_DL):
        """
        Calculate the total offloading delay.

        Parameters:
            D_UL (float): Uplink transmission delay (in seconds).
            D_edge (float): Edge computing delay (in seconds).
            D_DL (float): Downlink transmission delay (in seconds).

        Returns:
            float: Total offloading delay (in seconds).
        """
        return D_UL + D_edge + D_DL

    def calculate_offloading_energy(self, P, transmission_rate):
        """
        Calculate the energy consumption during computation offloading.

        Parameters:
            P (float): Transmit power (in Watts).
            transmission_rate (float): Transmission rate (in bits per second).

        Returns:
            float: Energy consumption during offloading (in Joules).
        """
        return (P * self.processing_complexity * self.input_data_size) / transmission_rate

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
            total_energy += (self.vehicle_transmit_power * self.processing_complexity * (1 - mu_V[i]) * self.input_data_size) / R_VT_Bj

        # Calculate energy for local computing
        total_energy += n * (self.kappa * (2.5 * 1e9) ** 2 * self.processing_complexity * mu_loc * self.input_data_size)  # Example: f_loc = 2.5 GHz

        return total_energy

# Create an instance of the simulation parameters
sim_params = SimulationParameters()

# Example usage of local computing calculations
f_loc = 2.5 * 1e9  # Example CPU frequency of the target vehicle (2.5 GHz)
D_loc = sim_params.calculate_local_delay(f_loc)
E_loc = sim_params.calculate_local_energy(f_loc)

print(f"Local Computing Delay: {D_loc:.4f} seconds")
print(f"Local Energy Consumption: {E_loc:.4f} Joules")

# Example task fractions (mu_V, mu_B, mu_loc)
# Ensure that the sum of mu_V, mu_B, and mu_loc satisfies the constraint: sum(mu_V) + sum(mu_B) + n * mu_loc = 1
mu_V = [0.3 / sim_params.num_vehicles] * sim_params.num_vehicles  # 30% of the task offloaded to vehicles
mu_B = [0.2 / sim_params.num_rsus] * sim_params.num_rsus  # 20% of the task offloaded to RSUs
mu_loc = 0.1  # 10% of the task computed locally
n = 1  # Number of areas where the task is computed locally

# Verify the constraint
total_fraction = sum(mu_V) + sum(mu_B) + n * mu_loc
print(f"Total Fraction: {total_fraction:.2f} (Should be 1.0)")

# Calculate total delay and energy
D_tot = sim_params.calculate_total_delay(mu_V, mu_B, mu_loc, n)
E_tot = sim_params.calculate_total_energy(mu_V, mu_B, mu_loc, n)

print(f"Total Delay: {D_tot:.4f} seconds")
print(f"Total Energy Consumption: {E_tot:.4f} Joules")

# # Plotting the data
# plt.figure(figsize=(18, 12))

# # Plot 1: Local Computing Delay vs CPU Frequency
# plt.subplot(3, 2, 1)
# f_loc_range = np.linspace(1e9, 8e9, 100)  # CPU frequency range from 1 GHz to 8 GHz
# D_loc_range = [sim_params.calculate_local_delay(f) for f in f_loc_range]
# plt.plot(f_loc_range / 1e9, D_loc_range, color='blue')
# plt.xlabel('CPU Frequency (GHz)')
# plt.ylabel('Local Computing Delay (seconds)')
# plt.title('Local Computing Delay vs CPU Frequency')
# plt.grid(True)

# # Plot 2: Local Energy Consumption vs CPU Frequency
# plt.subplot(3, 2, 2)
# E_loc_range = [sim_params.calculate_local_energy(f) for f in f_loc_range]
# plt.plot(f_loc_range / 1e9, E_loc_range, color='green')
# plt.xlabel('CPU Frequency (GHz)')
# plt.ylabel('Local Energy Consumption (Joules)')
# plt.title('Local Energy Consumption vs CPU Frequency')
# plt.grid(True)

# # Plot 3: Edge Computing Delay vs CPU Frequency
# plt.subplot(3, 2, 3)
# f_edge_range = np.linspace(1e9, 16e9, 100)  # CPU frequency range from 1 GHz to 16 GHz
# D_edge_range = [sim_params.calculate_edge_delay(f) for f in f_edge_range]
# plt.plot(f_edge_range / 1e9, D_edge_range, color='red')
# plt.xlabel('CPU Frequency (GHz)')
# plt.ylabel('Edge Computing Delay (seconds)')
# plt.title('Edge Computing Delay vs CPU Frequency')
# plt.grid(True)

# # Plot 4: Total Offloading Delay vs Distance
# plt.subplot(3, 2, 4)
# distance_range = np.linspace(1e-9, sim_params.communication_range, 100)  # Distance range from 0 to 200 meters
# transmission_rate_range = [sim_params.calculate_transmission_rate(sim_params.vehicle_transmit_power, d) for d in distance_range]
# D_UL_range = [sim_params.calculate_ul_delay(rate) for rate in transmission_rate_range]
# D_DL_range = [sim_params.calculate_dl_delay(rate) for rate in transmission_rate_range]

# # Calculate D_edge for each distance (using a fixed edge server CPU frequency for simplicity)
# f_edge = 4.0 * 1e9  # Example CPU frequency of the selected edge server (4.0 GHz)
# D_edge = sim_params.calculate_edge_delay(f_edge)  # Fixed edge computing delay

# # Calculate total offloading delay for each distance
# D_off_range = [sim_params.calculate_total_offloading_delay(D_UL, D_edge, D_DL) for D_UL, D_DL in zip(D_UL_range, D_DL_range)]

# plt.plot(distance_range, D_off_range, color='purple')
# plt.xlabel('Distance (meters)')
# plt.ylabel('Total Offloading Delay (seconds)')
# plt.title('Total Offloading Delay vs Distance')
# plt.grid(True)

# # Plot 5: Transmission Rate vs Distance
# plt.subplot(3, 2, 5)
# plt.plot(distance_range, transmission_rate_range, color='orange')
# plt.xlabel('Distance (meters)')
# plt.ylabel('Transmission Rate (bps)')
# plt.title('Transmission Rate vs Distance')
# plt.grid(True)

# # Plot 6:

# # Plot 6: Offloading Energy Consumption vs Distance
# plt.subplot(3, 2, 6)
# E_off_range = [sim_params.calculate_offloading_energy(sim_params.vehicle_transmit_power, rate) for rate in transmission_rate_range]
# plt.plot(distance_range, E_off_range, color='brown')
# plt.xlabel('Distance (meters)')
# plt.ylabel('Offloading Energy Consumption (Joules)')
# plt.title('Offloading Energy Consumption vs Distance')
# plt.grid(True)

# # Display the plots
# plt.tight_layout()
# plt.show()