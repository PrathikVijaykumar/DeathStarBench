import random
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import os
from scipy.optimize import minimize
from scipy.stats import qmc  # Import for quasi-random sequence generation

#============================
### GLOBAL VARIABLES DECLARATION
#============================
latency_weight = 1.0
cpu_allocation_weight = 10.0

container_names = [
    "socialnetwork-compose-post-service-1",
    "socialnetwork-nginx-thrift-1",
    "socialnetwork-home-timeline-service-1",
    "socialnetwork-user-timeline-service-1",
    "socialnetwork-user-service-1",
    "socialnetwork-post-storage-service-1",
    "socialnetwork-media-service-1",
    "socialnetwork-user-mention-service-1",
    "socialnetwork-social-graph-service-1",
    "socialnetwork-text-service-1"
]

num_of_containers = len(container_names)


def update_container_resources(container_name, cpu_allocation, memory_allocation):
    """
    Update the CPU and memory allocation for a given container.
    
    Args:
        container_name (str): The name of the container to update.
        cpu_allocation (float): The CPU allocation value.
        memory_allocation (int): The memory allocation value in MB.
        
    Returns:
        None
    """
    cpu_period = 100000
    cpu_quota = int(cpu_allocation * cpu_period)
    hardcoded_password = "your_placeholder_password"  # Placeholder for actual password
    if memory_allocation != -1:
        cmd = f'echo "{hardcoded_password}" | dzdo -S docker update --cpus={cpu_allocation} --memory={memory_allocation}m --memory-swap={memory_allocation}m {container_name}'
    else:
        cmd = f'echo "{hardcoded_password}" | dzdo -S docker update --cpu-period={cpu_period} --cpu-quota={cpu_quota} {container_name}'
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_latency_jaeger():
    """
    Get the latency from Jaeger using a specific script.
    
    Args:
        None
        
    Returns:
        int: The total average duration in microseconds.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(script_dir, '..')
    os.chdir(relative_path)
    cmd = ["python3", "jaegergrpc_trial.py", "grpc", "1", "socialnetwork-nginx-thrift-1"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract total_average_duration from the output
    match = re.search(r"Total average duration across all services: (\d+\.\d+) microseconds", result.stdout)
    if match:
        total_average_duration = float(match.group(1))
        rounded_duration = round(total_average_duration)
        return rounded_duration
    else:
        return 0

def execute_docker_command(cmd):
    """
    Execute a Docker command and return its output.
    
    Args:
        cmd (str): The Docker command to execute.
        
    Returns:
        str: The output of the command.
    """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def estimate_gradient(container_name, current_cpu, epsilon=0.05, latency_weight=1.0, cpu_allocation_weight=10.0):
    """
    Estimate the gradient for a given container's CPU allocation.
    
    Args:
        container_name (str): The name of the container to update.
        current_cpu (float): The current CPU allocation value.
        epsilon (float): The perturbation value for gradient estimation.
        latency_weight (float): The weight for latency in the cost function.
        cpu_allocation_weight (float): The weight for CPU allocation in the cost function.
        
    Returns:
        tuple: A tuple containing the estimated gradient and the average cost.
    """
    # Increase CPU allocation
    update_container_resources(container_name, current_cpu + epsilon, -1)
    time.sleep(25)  # Wait for the system to stabilize
    latency_increase = get_latency_jaeger()
    cost_increase = latency_weight * (latency_increase / 1000) + cpu_allocation_weight * (current_cpu + epsilon)

    # Decrease CPU allocation
    update_container_resources(container_name, current_cpu - epsilon, -1)
    time.sleep(25)  # Wait for the system to stabilize
    latency_decrease = get_latency_jaeger()
    cost_decrease = latency_weight * (latency_decrease / 1000) + cpu_allocation_weight * (current_cpu - epsilon)

    # Estimate gradient based on cost
    estimated_gradient = (cost_increase - cost_decrease) / (2 * epsilon)
    average_cost = ((cost_increase + cost_decrease) / 2)
    
    return estimated_gradient, average_cost

def optimize_all_container_with_vanilla_sgd(container_names, initial_cpu=0.5, initial_learning_rate=0.01, max_iterations=10,decay_factor = 0.2):
    """
    Optimize resource allocation for all containers using Vanilla SGD.
    
    Args:
        container_names (list): List of container names.
        initial_cpu (float): Initial CPU allocation value.
        initial_learning_rate (float): Initial learning rate for SGD.
        max_iterations (int): Maximum number of iterations.
        
    Returns:
        tuple: A tuple containing lists of global costs and time stamps.
    """
    global_cpu_allocations = {name: initial_cpu for name in container_names}  # Starting point
    global_costs = []  # To track cost instead of latency
    learning_rate = initial_learning_rate
    time_stamps = []  # To record the elapsed time
    start_time = time.time()  # Start the timer

    for iteration in range(max_iterations):
        container_gradients = {}
        total_cost = 0  # Initialize total cost for this iteration
    
        # Estimate gradients for each container
        for container_name in container_names:
            gradient, cost = estimate_gradient(container_name, global_cpu_allocations[container_name])
            normalized_gradient = gradient / abs(gradient) if gradient != 0 else 0  # Normalized
            container_gradients[container_name] = normalized_gradient
            total_cost += cost  # Aggregate cost
        
        # Average cost for this iteration
        average_cost = total_cost / len(container_names)
        global_costs.append(average_cost)

        # Update global CPU allocations based on gradients
        for container_name in container_gradients:
            update_step = learning_rate * container_gradients[container_name]
            new_cpu = max(min(global_cpu_allocations[container_name] - update_step, 1.0), 0.1)
            global_cpu_allocations[container_name] = new_cpu
            update_container_resources(container_name, new_cpu, -1)
    
        # Measure global latency (could be average or total)
        current_time = time.time() - start_time
        time_stamps.append(current_time)  # Record the elapsed time
        learning_rate *= (1.0 / (1.0 + iteration * decay_factor))  # Example schedule: learning rate decay over iterations
    
    return global_costs, time_stamps

def plot_optimization_results(time_stamps, global_costs, initial_cpu):
    """
    Plot the optimization results.
    
    Args:
        time_stamps (list): List of time stamps.
        global_costs (list): List of global costs.
        initial_cpu (float): Initial CPU allocation value.
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, global_costs, marker='o', linestyle='-', color='b')
    plt.xlabel('Time')
    plt.ylabel('Global Cost')
    plt.grid(True)
    filename = f"normalized_global_cost_time_based_convergence_plot_0.7_variable_workload.png"
    plt.title(f'Cost over Time - normalized - initial cpu {initial_cpu}')
    plt.savefig(f"/your_placeholder_path/results_for_paper/vanilla/{filename}")  # Placeholder for actual path
    plt.show()

def store_optimization_data(iterations, global_latencies, timestamps):
    """
    Store the CPU allocations and latencies to a CSV file for future reference.
    
    Args:
        iterations (list): List of iteration numbers.
        global_latencies (list): List of global latencies.
        timestamps (list): List of time stamps.
        
    Returns:
        None
    """
    df = pd.DataFrame({
        'Iterations': iterations,
        'Latency': global_latencies,
        'timestamps': timestamps
    })
    filename = f"convergence_data_normalized_0.7_variable_workload.csv"
    filepath = f"/your_placeholder_path/results_for_paper/vanilla/{filename}"  # Placeholder for actual path
    df.to_csv(filepath, index=False)

def optimize_all_containers(epsilon):
    """
    Perform zeroth-order optimization for all containers.
    
    Args:
        epsilon (float): The perturbation value for gradient estimation.
        
    Returns:
        None
    """
    initial_cpu = 0.7
    initial_learning_rate = 0.04
    max_iterations = 13
    global_costs, time_stamps = optimize_all_container_with_vanilla_sgd(container_names, initial_cpu, initial_learning_rate, max_iterations)
    plot_optimization_results(time_stamps, global_costs, initial_cpu)
    store_optimization_data(range(max_iterations), global_costs, time_stamps)

def main():
    """
    Main function to run the optimization and stop all Docker containers.
    
    Args:
        None
        
    Returns:
        None
    """
    optimize_all_containers(0)
    hardcoded_password = "your_placeholder_password"  # Placeholder for actual password
    cmd = f'echo "{hardcoded_password}" | dzdo docker stop $(dzdo docker ps -a -q)'  
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
