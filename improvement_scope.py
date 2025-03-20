def build_network(self):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(self.input_dim,)),  # Fixing input layer issue
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(self.action_dim, activation=None)
    ])
    return model

def calculate_transmission_rate(P, d):
    d = max(d, 1e-9)  # Prevent division by zero
    SNR = (P * (d ** -4)) / NOISE_POWER
    transmission_rate = BANDWIDTH * np.log2(1 + SNR)
    
    return max(transmission_rate, 1e-6)  # Ensure transmission rate is not zero

def run_ddqn_simulation():
    global global_total_latency_local, global_total_energy_local, global_count_local
    ddqn_episode_rewards = []
    for episode in range(EPISODES):
        total_reward = 0
        for mv in mission_vehicles:
            available_rsus = [r for r in rsus if abs(r.x - mv.x) <= COMMUNICATION_RANGE]
            available_vehicles = [v for v in mission_vehicles if v.vehicle_id != mv.vehicle_id and abs(v.x - mv.x) <= COMMUNICATION_RANGE]

            state = [len(available_rsus), mv.x, mv.y, mv.cpu, mv.task_size, mv.velocity, mv.direction]
            action = ddqn.act(state)

            if action == 0:  # Local Execution
                delay = mv.calculate_local_delay()
                energy = mv.calculate_local_energy()
            
            elif action == 1 and available_rsus:  # RSU Offloading
                selected_rsu = min(available_rsus, key=lambda r: np.sqrt((mv.x - r.x) ** 2 + (mv.y - r.y) ** 2))
                distance = np.sqrt((mv.x - selected_rsu.x) ** 2 + (mv.y - selected_rsu.y) ** 2)
                transmission_rate = calculate_transmission_rate(mv.vehicle_transmit_power, distance)
                
                delay = (PROCESSING_COMPLEXITY * mv.task_size) / selected_rsu.cpu
                delay_uplink = ALPHA * (mv.task_size / transmission_rate)
                delay_downlink = BETA * (mv.task_size / transmission_rate)
                total_delay = delay + delay_uplink + delay_downlink
                energy = (mv.vehicle_transmit_power * PROCESSING_COMPLEXITY * mv.task_size) / transmission_rate

            elif action == 2 and available_vehicles:  # V2V Offloading
                selected_vehicle = min(available_vehicles, key=lambda v: np.sqrt((mv.x - v.x) ** 2 + (mv.y - v.y) ** 2) / v.cpu)
                distance = np.sqrt((mv.x - selected_vehicle.x) ** 2 + (mv.y - selected_vehicle.y) ** 2)
                transmission_rate = calculate_transmission_rate(mv.vehicle_transmit_power, distance)

                delay = (PROCESSING_COMPLEXITY * mv.task_size) / selected_vehicle.cpu
                delay_uplink = ALPHA * (mv.task_size / transmission_rate)
                delay_downlink = BETA * (mv.task_size / transmission_rate)
                total_delay = delay + delay_uplink + delay_downlink
                energy = (mv.vehicle_transmit_power * PROCESSING_COMPLEXITY * mv.task_size) / transmission_rate
            else:  # Fallback to local execution
                delay = mv.calculate_local_delay()
                energy = mv.calculate_local_energy()

            reward = compute_reward(delay, energy)
            total_reward += reward

        ddqn_episode_rewards.append(total_reward)
        if episode % 50 == 0:
            print(f"Episode {episode} - Reward: {total_reward:.2f}")

    return ddqn, ddqn_episode_rewards
