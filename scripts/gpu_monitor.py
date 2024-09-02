import os
import time
import pynvml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def capture_gpu(output_file='gpu_memory_usage.log', interval=0.1, num_chunks=7):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count < 1:
        raise RuntimeError("No GPU devices found.")

    start_time = time.time()

    target_memory_usage = [np.random.uniform(300, 500) for _ in range(num_chunks)]
    
    amplitude = 50
    frequency = 0.1
    noise_level = 20
    ramp_up_duration = 10

    with open(output_file, 'w') as f:
        f.write('Time')
        for i in range(num_chunks):
            f.write(f',sGPU_{i}_Memory_Usage(MB)')
        f.write('\n')

        try:
            while not all(os.path.exists(f'training_complete_chunk_{i}.flag') for i in range(num_chunks)):
                current_time = time.time() - start_time
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory = memory_info.used / 1024 ** 2

                simulated_memory_usage = []
                for i, target_usage in enumerate(target_memory_usage):
                    ramp_up_factor = min(current_time / ramp_up_duration, 1)
                    base_usage = target_usage * ramp_up_factor
                    
                    sine_component = amplitude * np.sin(2 * np.pi * frequency * current_time + i)
                    noise_component = np.random.normal(0, noise_level)
                    usage = base_usage + (sine_component + noise_component) * ramp_up_factor
                    
                    usage = max(0, min(usage, 2000))
                    simulated_memory_usage.append(usage)

                print(f'Current Time: {current_time:.2f}')
                print(f'Total GPU Memory Usage: {total_memory:.2f}')
                print(f'Simulated GPU Memory Usage: {[f"{usage:.2f}" for usage in simulated_memory_usage]}')

                f.write(f'{current_time:.2f}')
                for usage in simulated_memory_usage:
                    f.write(f',{usage:.2f}')
                f.write('\n')
                f.flush()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Capture interrupted by user.")
        finally:
            pynvml.nvmlShutdown()

def plot_gpu_memory(input_file='gpu_memory_usage.log', num_chunks=7):
    try:
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The input file {input_file} was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The input file {input_file} is empty.")

    plt.figure(figsize=(12, 8))
    for i in range(num_chunks):
        plt.plot(data['Time'], data[f'sGPU_{i}_Memory_Usage(MB)'], label=f'sGPU {i}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('GPU Memory Usage (MB)')
    plt.title('Simulated GPU Memory Usage Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('gpu_memory_usage.png')
    plt.show()
