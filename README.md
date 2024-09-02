# Simulating Multi-GPU Distributed Deep Learning on a Single GPU with Dynamic Memory Management

## Introduction

This project explores the concept of simulating a multi-GPU environment using only a single GPU. By dynamically managing memory and using PyTorch and PyTorch Lightning, it allows users to experience distributed deep learning training methods without the need for multiple physical GPUs.

## Project Overview

- **Objective**: To simulate a multi-GPU training environment on a single GPU by dividing the memory allocation dynamically.
- **Technologies Used**: Python, PyTorch, PyTorch Lightning, NVIDIA's NVML library, TensorBoard.
- **Outcome**: Users can test distributed training techniques and understand multi-GPU training behaviors without needing multiple GPUs.

## Key Features

- **Simulated Multi-GPU Environment**: Emulates the behavior of multiple GPUs using only a single GPU, allowing for distributed training simulations.
- **Dynamic Memory Management**: Manages GPU memory dynamically to simulate the usage patterns of multiple GPUs.
- **Real-time Monitoring**: Uses NVIDIA's NVML library to monitor and log GPU memory usage in real-time.
- **Comprehensive Visualization**: Provides visualizations of simulated GPU memory usage over time to better understand the distribution of memory load.

## How It Works

1. **Dynamic Memory Allocation**: The project allocates memory dynamically across several simulated GPUs by splitting the memory usage of a single GPU. It uses a combination of Python and NVIDIA's NVML library to manage and monitor these allocations.

2. **Training Simulation**: The project runs deep learning training jobs that mimic distributed training across the simulated GPUs. The jobs are managed using PyTorch Lightning, which simplifies the model training process and provides a structure for managing different training tasks.

3. **Memory Monitoring and Logging**: As the training proceeds, the memory usage of each simulated GPU is logged in real-time. This data is then visualized to provide insights into the memory distribution and usage patterns.

## Why Simulate Multiple GPUs?

- **Accessibility**: Not everyone has access to a multi-GPU setup. By simulating multiple GPUs on a single device, we can provide a similar experience to those who only have access to one GPU.
  
- **Cost-Effectiveness**: Buying and maintaining multiple GPUs can be expensive. This simulation allows users to experiment with distributed learning without the additional hardware cost.
  
- **Educational Purposes**: It is a great tool for learning and teaching distributed training methods, helping to understand how multi-GPU setups work without needing the physical hardware.

## Installation

### Prerequisites

- **Python 3.8 or higher**: Ensure you have Python installed. You can download it from [Python's official website](https://www.python.org/).
- **CUDA Toolkit**: A compatible CUDA toolkit for your GPU (version 12.1 used in this project). Download it from [NVIDIA's official site](https://developer.nvidia.com/cuda-downloads).

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mattjesc/Simulating-Multi-GPU-Distributed-Deep-Learning.git
   cd Simulating-Multi-GPU-Distributed-Deep-Learning
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the main script**:
   ```bash
   python main.py
   ```
   This will initiate the training process and start simulating GPU memory usage.

2. **View GPU Memory Usage**:
   After the training completes, the GPU memory usage logs will be saved in `gpu_memory_usage.log`, and a plot of this usage over time will be generated as `gpu_memory_usage.png`.

## Understanding the Code

### Main Components

- **`cnn_model.py`**: Defines the structure of the Convolutional Neural Network (CNN) used for training.
- **`data_module.py`**: Manages data loading, transformation, and preparation using PyTorch Lightning's `DataModule`.
- **`train.py`**: Contains functions to train the model on each simulated GPU "chunk".
- **`monitor_gpu.py`**: Logs GPU memory usage dynamically to simulate multi-GPU training.
- **`plot_gpu.py`**: Visualizes the GPU memory usage data captured during training.
- **`main.py`**: The main script that coordinates training and monitoring activities.

## System Compatibility and Requirements

- **Operating System**: The project should work on any OS that supports Python and CUDA (Windows, Linux, MacOS with specific configurations).
- **CUDA Version**: This project is configured for CUDA version 12.1 (`+cu121`). If using a different version, update the `torch` and `torchvision` library versions in the `requirements.txt` file accordingly. Refer to the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page for compatibility details.

## Important Notes

- **Simulated Environment**: This project simulates multiple GPUs using only a single GPU by dynamically managing memory. It does not achieve the actual performance of a multi-GPU setup.
  
- **Variability in Results**: Performance and results can vary based on system configuration, GPU model, available memory, system load, and other running processes.
  
- **External Factors**: Factors such as GPU thermal throttling, driver versions, and system-level power management can affect the simulation and its results.