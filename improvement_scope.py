def select_best_vehicle_for_offloading(mv, available_vehicles, task_deadline, task_size):
    best_vehicle = None
    best_score = float('inf')
    
    for v in available_vehicles:
        d = ((mv.x - v.x) ** 2 + (mv.y - v.y) ** 2) ** 0.5
        transmission_rate = calculate_transmission_rate(v.transmit_power, d)
        transmission_delay = task_size / transmission_rate
        computation_delay = (PROCESSING_COMPLEXITY * task_size) / v.cpu
        total_delay = transmission_delay + computation_delay
        
        if total_delay <= task_deadline:  # Ensure deadline is met
            score = total_delay / v.cpu  # Prioritize lower delay and higher CPU
            if score < best_score:
                best_score = score
                best_vehicle = v
    
    return best_vehicle