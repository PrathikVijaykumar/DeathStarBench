import subprocess
import random
import time

trail = []

def run_benchmark(rps):
    command = f"../wrk2/wrk -D exp -t 4 -c 8 -d 22m -L -s ./wrk2/scripts/social-network/compose-post.lua http://localhost:8080/wrk2-api/post/compose -R {rps}"
    print(f"Running benchmark with {rps} requests per second")
    subprocess.run(command, shell=True, check=True)
    time.sleep(2)


def main():
    rps_values = [2000, 1900, 1500, 1600 , 1800]  # Add more RPS values as needed
    for rps in rps_values:
        trail.append(rps)
        run_benchmark(rps)
    print(trail)
    for rps in rps_values:
        trail.append(rps)
        run_benchmark(rps)
    with open('Workload_values.txt', 'a') as file:
        for value in trail:
            file.write(f"{value}\n")

if __name__ == "__main__":
    main()
