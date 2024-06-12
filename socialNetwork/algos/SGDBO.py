import random
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import os
import pandas as pd
import random
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize

space = [
    (0.1, 0.18)
]
global_costs_tracking = []
time_stamps_tracking = []
call_counter = 0
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
initial_allocation = 0.7
current_allocation = [initial_allocation] * len(container_names)

def update_container_resources(container_name, cpu_allocation, memory_allocation):
    """
    Update the CPU allocation for a given container.
    
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
    # print(f"NOW EXECUTING {cmd}")
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
    cmd = ["python3", "jaegergrpc.py", "grpc", "1", "socialnetwork-nginx-thrift-1"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract total_average_duration from the output
    match = re.search(r"Total average duration across all services: (\d+\.\d+) microseconds", result.stdout)
    if match:
        total_average_duration = float(match.group(1))
        rounded_duration = round(total_average_duration)
        return rounded_duration
    else:
        # print("Total average duration not found in the script output.")
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

def run_sgd(container_names):
    """
    Run SGD to estimate the gradients for each container.
    
    Args:
        container_names (list): List of container names.
        
    Returns:
        np.ndarray: The normalized gradient vector.
    """
    global current_allocation
    gradient_vector = []
      
    for i in range(len(container_names)):
        gradient, _ = estimate_gradient(container_names[i], current_allocation[i])
        gradient_vector.append(gradient)
    
    normalized_gradient_vector = gradient_vector / np.linalg.norm(gradient_vector)

    return normalized_gradient_vector

def optimize_hyperparameters_with_bo(container_names, n_calls=10):
    """
    Optimize hyperparameters using Bayesian Optimization.
    
    Args:
        container_names (list): List of container names.
        n_calls (int): Number of optimization calls.
        
    Returns:
        float: The optimized cost.
    """
    global current_allocation
    normalized_gradient_vector = run_sgd(container_names)

    def objective_function(x):
        """
        Objective function for Bayesian Optimization.
        
        Args:
            x (list): List of hyperparameter values.
            
        Returns:
            float: The cost for the given hyperparameters.
        """
        global current_allocation
        curr_allocation = current_allocation.copy()
        curr_allocation += normalized_gradient_vector * x[0]
        for i in range(len(container_names)):
            update_container_resources(container_names[i], current_allocation[i], -1)
        time.sleep(25)
        latency = get_latency_jaeger()
        cost = latency_weight * (latency / 1000) + cpu_allocation_weight * (sum(curr_allocation) / len(curr_allocation))
        return cost

    result = gp_minimize(
        func=objective_function,
        dimensions=space,
        acq_func="LCB",
        n_calls=n_calls,
        random_state=0
    )

    current_allocation += normalized_gradient_vector * (result.x)[0]
    for i in range(len(current_allocation)):
        update_container_resources(container_names[i], current_allocation[i], -1)
    time.sleep(25)
    latency = get_latency_jaeger()
    cost = latency_weight * (latency / 1000) + cpu_allocation_weight * (sum(current_allocation) / len(current_allocation))

    return cost

def plot_cost_vs_time(cost_list):
    """
    Plot the cost vs. time.
    
    Args:
        cost_list (list): List of costs.
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_list, marker='', linestyle='-', color='b')  # No marker for the line
    plt.title("Cost vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    filename = f"sgdo_global_convergence_plot_0.7_variable_workload_TTT.png"
    plt.savefig(f"/your_placeholder_path/results_for_paper/sgdbo/{filename}")  # Placeholder for actual path
    plt.show()
    return 

def store_convergence_data(cost_list, time_stamps):
    """
    Store the convergence data in a CSV file.
    
    Args:
        cost_list (list): List of costs.
        time_stamps (list): List of time stamps.
        
    Returns:
        None
    """
    iteration = [i for i in range(1, len(cost_list) + 1)]
    data = pd.DataFrame({
        'iteration': iteration,
        'cost observed': cost_list,
        'timestamps': time_stamps
    })
    filename = f"sgbod_all_convergence_data_0.7_variable_workload_TTT.csv"
    filepath = f"/your_placeholder_path/results_for_paper/sgdbo/{filename}"  # Placeholder for actual path
    data.to_csv(filepath, index=False)

def optimize_all_containers(max_iterations=15):
    """
    Perform zeroth-order optimization for all containers.
    
    Args:
        max_iterations (int): Maximum number of iterations.
        
    Returns:
        tuple: A tuple containing lists of cost values and time stamps.
    """
    cost_list = []
    time_stamps = []  # To record the elapsed time
    start_time = time.time()  # Start the timer
    for i in range(max_iterations):
        cost = optimize_hyperparameters_with_bo(container_names, n_calls=10)
        cost_list.append(cost)
        current_time = time.time() - start_time
        time_stamps.append(current_time)  # Record the elapsed time
    # print(res_gp)
    return cost_list, time_stamps

# Example usage to optimize all containers:
cost_list, time_stamps = optimize_all_containers()
plot_cost_vs_time(cost_list)
store_convergence_data(cost_list, time_stamps)

hardcoded_password = "your_placeholder_password"  # Placeholder for actual password
cmd = f'echo "{hardcoded_password}" | dzdo docker stop $(dzdo docker ps -a -q)'  
# print(f"NOW EXECUTING {cmd}")
subprocess.run(cmd, shell=True)
