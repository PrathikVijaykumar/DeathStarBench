import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import os
import pandas as pd
import random

import gym
from stable_baselines3 import PPO
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback

#============================
### GLOBAL VARIABLES DECLARATION
#============================
CPU_COST_FACTOR = 10.0
latency_weight = 1.0
container_names_copy = [
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
num_of_containers = len(container_names_copy)

class StopTrainingOnEpisodeDoneCallback(BaseCallback):
    """
    A custom callback that stops the training when a single episode is done.
    """
    def __init__(self, verbose=0):
        super(StopTrainingOnEpisodeDoneCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        """
        This method is called after each call to `env.step()` in the training loop.
        If `done` is True for any environment, stop the training by returning False.
        
        Args:
            None
            
        Returns:
            bool: False if any environment is done, True otherwise.
        """
        if any(self.locals['dones']):  # Check if 'done' is True for any environment
            return False  # Returning False stops the training
        return True  # Continue training

def update_container_resources(container_names, cpu_allocations, memory_allocation):
    """
    Update the CPU allocations for multiple containers.
    
    Args:
        container_names (list): List of container names.
        cpu_allocations (list): List of CPU allocation values.
        memory_allocation (int): Memory allocation value in MB.
        
    Returns:
        None
    """
    cpu_period = 100000
    for i, container_name in enumerate(container_names):
        cpu_quota = int(round(cpu_allocations[i], 5) * cpu_period)
        hardcoded_password = "your_placeholder_password"  # Placeholder for actual password
        if memory_allocation != -1:
            cmd = f'echo "{hardcoded_password}" | dzdo -S docker update --cpus={cpu_allocations[i]} --memory={memory_allocation}m --memory-swap={memory_allocation}m {container_name}'
        else:
            cmd = f'echo "{hardcoded_password}" | dzdo -S docker update --cpu-period={cpu_period} --cpu-quota={cpu_quota} {container_name}'
        subprocess.run(cmd, shell=True)

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
        return 0

def cost_function(latency, cpu_allocations, cpu_cost_factor):
    """
    Calculate the cost function based on latency and CPU allocations.
    
    Args:
        latency (float): The measured latency.
        cpu_allocations (list): List of CPU allocation values.
        cpu_cost_factor (float): The cost factor for CPU usage.
        
    Returns:
        float: The total cost.
    """
    latency_cost = latency_weight * (latency / 1000)
    resource_cost = (sum(cpu_allocations) / len(cpu_allocations)) * cpu_cost_factor
    total_cost = latency_cost + resource_cost
    return total_cost

class ContainerEnv(gym.Env):
    """
    Custom Environment for multiple containers' CPU Allocation using OpenAI Gym.
    """
    def __init__(self):
        super(ContainerEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([3] * num_of_containers)  # 3 actions for each of 10 containers
        self.observation_space = spaces.Box(low=np.array([0]*num_of_containers), high=np.array([np.inf]*num_of_containers), dtype=np.float32)
        self.max_iterations = 30
        self.iteration_count = 0
        self.container_names = [
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
        self.cpu_allocations = [0.7] * num_of_containers  # Initial CPU allocations for each container
        self.latencies = []
        self.costs = []
        self.time_stamps = []  # To record the elapsed time
        self.start_time = time.time()  # Start the timer
    
    def render(self, mode='console'):
        """
        Render the environment.
        
        Args:
            mode (str): The mode to render with.
            
        Returns:
            None
        """
        if mode == 'console':
            print(f"Current Latency: {self.current_latency}, CPU Allocation: {self.cpu_allocations}")
        else:
            super(ContainerEnv, self).render(mode=mode)

    def reset(self):
        """
        Reset the environment to an initial state.
        
        Args:
            None
            
        Returns:
            np.ndarray: The initial observation.
        """
        self.iteration_count = 0
        self.cpu_allocations = [0.7] * num_of_containers
        update_container_resources(self.container_names, self.cpu_allocations, -1)
        time.sleep(25)
        self.current_latency = get_latency_jaeger()
        cost = cost_function(self.current_latency, self.cpu_allocations, CPU_COST_FACTOR)
        self.cost_value = cost
        self.start_time = time.time() 
        return np.array(self.cpu_allocations).astype(np.float32)

    def step(self, actions):
        """
        Execute one time step within the environment.
        
        Args:
            actions (np.ndarray): The actions to take.
            
        Returns:
            tuple: A tuple containing the new observation, reward, done flag, and info dictionary.
        """
        self.iteration_count += 1
        done = self.iteration_count >= self.max_iterations
        self._take_action(actions)
        time.sleep(25)
        self.current_latency = get_latency_jaeger()
        
        self.latencies.append(self.current_latency)  # Store the current latency
        cost = cost_function(self.current_latency, self.cpu_allocations, CPU_COST_FACTOR)
        self.cost_value = cost
        self.costs.append(cost)
        current_time = time.time() - self.start_time
        self.time_stamps.append(current_time)  # Record the elapsed time
        reward = -cost

        return np.array(self.cpu_allocations).astype(np.float32), reward, done, {}

    def _take_action(self, actions):
        """
        Take the specified actions within the environment.
        
        Args:
            actions (np.ndarray): The actions to take.
            
        Returns:
            None
        """
        for i, action in enumerate(actions):
            if action == 0:  # Decrease CPU allocation
                self.cpu_allocations[i] = max(self.cpu_allocations[i] - 0.05, 0.1)
            elif action == 2:  # Increase CPU allocation
                self.cpu_allocations[i] = min(self.cpu_allocations[i] + 0.05, 1.0)

        update_container_resources(self.container_names, self.cpu_allocations, -1)    
        time.sleep(25)    

    def _get_reward(self):
        """
        Get the reward based on the current state.
        
        Args:
            None
            
        Returns:
            float: The reward value.
        """
        return -self.cost_value
    
    def get_latencies(self):
        """
        Get the recorded latencies.
        
        Args:
            None
            
        Returns:
            list: List of latencies.
        """
        return self.latencies
    
    def get_costs(self):
        """
        Get the recorded costs.
        
        Args:
            None
            
        Returns:
            list: List of costs.
        """
        return self.costs
    
    def get_time_stamps(self):
        """
        Get the recorded time stamps.
        
        Args:
            None
            
        Returns:
            list: List of time stamps.
        """
        return self.time_stamps

def plot_cost_vs_time(costs):
    """
    Plot the cost vs. time.
    
    Args:
        costs (list): List of costs.
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs, marker='o', linestyle='-', color='b')
    plt.title("Cost vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("cost = latency + cost of using resource ")
    filename = f"ppo_plot_COST_variable_workload.png"
    plt.grid(True)
    plt.savefig(f"/your_placeholder_path/results_for_paper/ppo/{filename}")  # Placeholder for actual path
    plt.show()

def store_convergence_data(time_stamps, costs):
    """
    Store the convergence data in a CSV file.
    
    Args:
        time_stamps (list): List of time stamps.
        costs (list): List of costs.
        
    Returns:
        None
    """
    iterations = list(range(1, len(costs) + 1))
    data = pd.DataFrame({
        'iterations': iterations,
        'Cost': costs,
        'Time': time_stamps
    })
    
    filename = f"ppo_convergence_data_COST_variable_workload.csv"
    filepath = f"/your_placeholder_path/results_for_paper/ppo/{filename}"  # Placeholder for actual path
    data.to_csv(filepath, index=False)

def main():
    """
    Main function to run the optimization, training, and cleanup.
    
    Args:
        None
        
    Returns:
        None
    """
    # Create and wrap the environment
    env = ContainerEnv()

    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1, 
        n_steps=2048,  # Adjust number of steps
        gamma=0.99,  # Adjust discount factor
        learning_rate=0.015,  # Adjust learning rate
        gae_lambda=0.95,  # Adjust GAE lambda
        ent_coef=0.01,  # Adjust entropy coefficient
        vf_coef=0.5,  # Adjust value function coefficient
        max_grad_norm=0.5  # Adjust maximum gradient norm
    )

    # Train the agent
    callback = StopTrainingOnEpisodeDoneCallback()
    model.learn(total_timesteps=35, callback=callback)

    # Test the trained agent
    obs = env.reset()
    done = False  # Initialize the done flag
    for i in range(30):
        if done:  # Check if the episode is done
            break  # Exit the loop if the episode is complete
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()  # or print relevant information

    latencies = env.get_latencies()  # Get the recorded latencies
    costs = env.get_costs()  # Get the recorded costs
    time_stamps = env.get_time_stamps()  # Get the recorded time stamps
    plot_cost_vs_time(costs)
    store_convergence_data(time_stamps, costs)

    hardcoded_password = "your_placeholder_password"  # Placeholder for actual password
    cmd = f'echo "{hardcoded_password}" | dzdo docker stop $(dzdo docker ps -a -q)'  
    subprocess.run(cmd, shell=True)
    cmd = f'echo "{hardcoded_password}" | dzdo docker rm -f $(dzdo docker ps -a -q)'  
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
