import random
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import os
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc  # Import for quasi-random sequence generation

#============================
### GLOBAL VARIABLES DECLARATION
#============================
latency_weight = 1.0
cpu_allocation_weight = 10.0
top_n_containers = 8

# Container names for the experiment
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

def execute_docker_command(cmd):
    """
    Executes a Docker command and returns its output.

    Args:
        cmd (str): The Docker command to execute.

    Returns:
        str: The output of the command.
    """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

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
    Gets the latency from Jaeger using a specific script.

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
        print("Total average duration not found in the script output.")
        return 0

def objective(x):
    """
    Objective function to minimize the L1 norm of x.

    Args:
        x (np.ndarray): The input array.

    Returns:
        float: The L1 norm of x.
    """
    return np.sum(np.abs(x))

def constraints(x, A, y, eta):
    """
    Constraints ensuring that the Euclidean norm of the residuals is within eta.

    Args:
        x (np.ndarray): The input array.
        A (np.ndarray): The measurement matrix.
        y (np.ndarray): The observations.
        eta (float): The tolerance value.

    Returns:
        float: The difference between eta and the Euclidean norm of the residuals.
    """
    return eta - np.linalg.norm(np.dot(A.T, x) - y)

def sigmoid_adjustment(values, scale=5):
    """
    Apply a sigmoid function to adjust CPU allocations within a soft limit, avoiding extreme adjustments for low values.

    Args:
        values (np.ndarray): The input values.
        scale (int): The scale factor for the sigmoid function.

    Returns:
        np.ndarray: The adjusted values.
    """
    return 1 / (1 + np.exp(-scale * (values - 0.5)))

def optimize_resources(container_names, initial_cpu_allocations, iterations=20):
    """
    Optimize resource allocation for the given containers.

    Args:
        container_names (list): List of container names.
        initial_cpu_allocations (list): List of initial CPU allocation values.
        iterations (int): Number of iterations for optimization.

    Returns:
        tuple: Optimized CPU allocations, cost list, allocation list, gradient history, and time stamps.
    """
    allocation_list = []
    cost_list = []
    time_stamps = []
    cpu_allocations = np.array(initial_cpu_allocations[:num_of_containers])
    np.random.seed(48)
    measurement_matrix = np.random.uniform(low=-1.0, high=1.0, size=(num_of_containers, top_n_containers))
    gradient_history = []
    start_time = time.time()  # Start the timer

    for iteration in range(iterations):
        print("------ITERATION--------{}----------------".format(iteration))
        cost_neutral = [measure_cost(cpu_allocations)] * top_n_containers
        sobol_generator = qmc.Sobol(d=top_n_containers, scramble=True)
        random_vector = sobol_generator.random_base2(m=int(2**np.ceil(np.log2(top_n_containers))))
        random_vector = random_vector[:top_n_containers] * 0.2
        perturbation = np.dot(measurement_matrix, random_vector)
        delta = 0.2 / np.max(perturbation)
        perturbation = perturbation * delta
        cpu_allocations_reshaped = cpu_allocations.reshape(-1, 1)
        perturbed_allocations = cpu_allocations_reshaped + perturbation
        perturbed_allocations = np.clip(perturbed_allocations, 0.1, 1.0)  # Ensuring allocations stay within bounds

        cost_measurements = []
        perturbed_allocations_transposed = np.transpose(perturbed_allocations)
        for alloc_set in perturbed_allocations_transposed:
            cost = measure_cost(alloc_set)
            cost_measurements.append(cost)

        new_list = [(cost_measurements[i] - cost_neutral[i]) / np.squeeze(random_vector[i]) for i in range(top_n_containers)]
        sum_list = [sum(x) for x in zip(*new_list)]
        observations = [x / top_n_containers for x in sum_list]

        eta = 0.2 / (1 + iteration * 0.05)
        cons = {'type': 'ineq', 'fun': constraints, 'args': (measurement_matrix, observations, eta)}
        result = minimize(objective, np.zeros(num_of_containers), method='SLSQP', constraints=[cons])
        estimated_gradient = result.x
        estimated_gradient = estimated_gradient / np.max(np.abs(estimated_gradient))

        learning_rate = 0.6 / (1 + iteration * 0.5)
        cpu_allocations -= learning_rate * estimated_gradient
        cpu_allocations = sigmoid_adjustment(cpu_allocations)  # Use sigmoid adjustment

        gradient_history.append(estimated_gradient)
        cost = measure_cost(cpu_allocations)
        cost_list.append(cost)
        allocation_list.append(str(cpu_allocations))
        current_time = time.time() - start_time
        time_stamps.append(current_time)

    return cpu_allocations, cost_list, allocation_list, gradient_history, time_stamps

def measure_cost(cpu_allocations):
    """
    Measure the cost for given CPU allocations.

    Args:
        cpu_allocations (list): List of CPU allocation values.

    Returns:
        float: The measured cost.
    """
    for index in range(len(container_names)):
        update_container_resources(container_names[index], cpu_allocations[index], -1)
    time.sleep(25)
    latency = get_latency_jaeger()
    cost = latency_weight * (latency / 1000) + cpu_allocation_weight * (sum(cpu_allocations) / len(cpu_allocations))
    return cost

def plot_latency_vs_iteration(costs):
    """
    Plot the cost versus iteration.

    Args:
        costs (list): List of costs.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs, marker='', linestyle='-', color='b')
    plt.title("Cost vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    filename = "CONGO_costs_vs_iterations.png"
    plt.savefig("/your_placeholder/path_to_save_csv/{}".format(filename))
    plt.show()

def store_convergence_data(allocation_list, costs, gradient_history, time_stamps):
    """
    Store convergence data in a CSV file.

    Args:
        allocation_list (list): List of CPU allocations.
        costs (list): List of observed costs.
        gradient_history (list): List of gradient history.
        time_stamps (list): List of time stamps.

    Returns:
        None
    """
    data = pd.DataFrame({
        'cpu allocation': allocation_list,
        'cost observed': costs,
        'gradients': gradient_history,
        'time_stamps': time_stamps
    })
    filename = "CONGO_data.csv"
    filepath = "/your_placeholder/path_to_save_plot/{}".format(filename)
    data.to_csv(filepath, index=False)

def main():
    """
    Main function to run the optimization and stop all Docker containers.

    Args:
        None

    Returns:
        None
    """
    initial_cpu_allocations = [0.7] * 10  # Starting point for CPU allocations

    # Optimize resources
    optimized_cpu_allocations, cost_list, allocation_list, gradient_history, time_stamps = optimize_resources(container_names, initial_cpu_allocations)

    # Plot and store results
    plot_latency_vs_iteration(cost_list)
    store_convergence_data(allocation_list, cost_list, gradient_history, time_stamps)

    # Stop all Docker containers
    hardcoded_password = "your_placeholder_password"  # Placeholder for actual password
    cmd = f'echo "{hardcoded_password}" | dzdo docker stop $(dzdo docker ps -a -q)'
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
