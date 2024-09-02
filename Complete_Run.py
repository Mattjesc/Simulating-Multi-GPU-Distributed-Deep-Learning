# Import necessary libraries for the project

import os  # Provides functions to interact with the operating system, used here for file checking and process management.
import time  # Provides time-related functions, used for measuring and tracking time intervals.
import torch  # The core PyTorch library for building and training neural networks.
import torch.multiprocessing as mp  # Multiprocessing support in PyTorch, essential for parallel processing on GPUs.
import pynvml  # NVIDIA Management Library (NVML) for monitoring GPU usage and status.
import pandas as pd  # Data manipulation and analysis library, used here for reading CSV files.
import matplotlib.pyplot as plt  # Plotting library to create visual representations of GPU memory usage.
import numpy as np  # Numerical computing library, used for random number generation and mathematical operations.
from torch.utils.data import DataLoader  # Utility to manage data loading in PyTorch, efficiently loading data in batches.
import pytorch_lightning as pl  # High-level wrapper for PyTorch to simplify model training and reduce boilerplate code.
from torchvision import datasets, transforms  # Provides popular datasets and common data transformations for computer vision.
from pytorch_lightning.loggers import TensorBoardLogger  # Logger for TensorBoard, useful for tracking and visualizing metrics during training.
from pytorch_lightning.callbacks import ModelCheckpoint  # Callback to save model checkpoints during training, helps in resuming training or for analysis later.

# Define CNNModel class inheriting from PyTorch Lightning's LightningModule

class CNNModel(pl.LightningModule):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Define layers of the Convolutional Neural Network (CNN)
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Convolutional layer, 32 filters of size 3x3, stride of 1, with padding.
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Second convolutional layer, increasing filters to 64.
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling layer to reduce spatial dimensions by half, aids in reducing overfitting and computation.
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)  # Fully connected layer that reshapes the tensor for dense classification.
        self.fc2 = torch.nn.Linear(128, 10)  # Output layer with 10 units for classification, corresponding to 10 classes in FashionMNIST.

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(torch.relu(self.conv1(x)))  # Convolution -> ReLU activation -> Max Pooling, helps extract spatial hierarchies.
        x = self.pool(torch.relu(self.conv2(x)))  # Another Conv -> ReLU -> Pooling, further increasing feature abstraction.
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor to a vector to prepare for fully connected layers.
        x = torch.relu(self.fc1(x))  # Fully connected layer with ReLU activation, introduces non-linearity.
        x = self.fc2(x)  # Output layer, logits for classification.
        return x

    def training_step(self, batch, batch_idx):
        # Define a single step of training
        x, y = batch  # Get input data and labels from the batch.
        y_hat = self(x)  # Forward pass through the model to get predictions.
        loss = torch.nn.functional.cross_entropy(y_hat, y)  # Calculate cross-entropy loss between predictions and actual labels.
        self.log("train_loss", loss)  # Log the training loss for monitoring.
        return loss

    def validation_step(self, batch, batch_idx):
        # Define a single step of validation
        x, y = batch  # Get input data and labels from the batch.
        y_hat = self(x)  # Forward pass through the model to get predictions.
        loss = torch.nn.functional.cross_entropy(y_hat, y)  # Calculate cross-entropy loss for validation.
        self.log("val_loss", loss)  # Log the validation loss for monitoring.
        return loss

    def configure_optimizers(self):
        # Define the optimizer for training the model
        return torch.optim.Adam(self.parameters(), lr=0.001)  # Using Adam optimizer, known for adaptive learning rates, with a standard learning rate of 0.001.

# Define FashionMNISTDataModule class for data handling

class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32, num_workers: int = 4):
        # Initialize the data module with directory, batch size, and number of workers for loading data
        super().__init__()
        self.data_dir = data_dir  # Directory where data will be stored and loaded from.
        self.batch_size = batch_size  # Batch size for loading data, controls memory usage and convergence speed.
        self.num_workers = num_workers  # Number of workers for data loading, speeds up data preparation by parallel processing.
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts images to PyTorch tensors, a format suitable for the model.
            transforms.Normalize((0.2860,), (0.3530,))  # Normalizes images to have zero mean and unit variance, helps in faster convergence.
        ])
        self.fashionmnist_train = None  # Placeholder for the training dataset.
        self.fashionmnist_test = None  # Placeholder for the test dataset.

    def prepare_data(self):
        # Method to download data if not already present
        datasets.FashionMNIST(self.data_dir, train=True, download=True)  # Download training data.
        datasets.FashionMNIST(self.data_dir, train=False, download=True)  # Download test data.

    def setup(self, stage=None):
        # Set up datasets based on the stage (training or validation)
        if stage == 'fit' or stage is None:
            self.fashionmnist_train = datasets.FashionMNIST(self.data_dir, train=True, transform=self.transform)  # Apply transformations and load training data.
        if stage == 'validate' or stage is None:
            self.fashionmnist_test = datasets.FashionMNIST(self.data_dir, train=False, transform=self.transform)  # Apply transformations and load test data.

    def train_dataloader(self):
        # Return a DataLoader for the training set
        return DataLoader(self.fashionmnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)  # Shuffling is essential for good generalization.

    def val_dataloader(self):
        # Return a DataLoader for the validation set
        return DataLoader(self.fashionmnist_test, batch_size=self.batch_size, num_workers=self.num_workers)  # No shuffling for validation data to ensure repeatability.

# Function to train on a specific chunk

def train_on_chunk(chunk_id, max_epochs=1):
    print(f"Starting training on chunk {chunk_id}")  # Indicate the start of training for this chunk.

    model = CNNModel()  # Initialize the CNN model.
    data_module = FashionMNISTDataModule()  # Initialize the data module for managing data.
    data_module.prepare_data()  # Prepare data by downloading if necessary.
    data_module.setup()  # Set up the data loaders.

    # Initialize logger for TensorBoard to visualize metrics during training.
    logger = TensorBoardLogger('logs', name=f'fashionmnist_experiment_chunk_{chunk_id}')
    
    # Initialize checkpoint callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/chunk_{chunk_id}',  # Directory to save checkpoints.
        filename='best-checkpoint',  # Naming the checkpoint file.
        save_top_k=1,  # Save only the best model checkpoint based on validation loss.
        verbose=True,  # Print details about the checkpointing process.
        monitor='val_loss',  # Metric to monitor for saving the best model.
        mode='min'  # We aim to minimize the validation loss.
    )

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,  # Number of epochs to train.
        devices=[0],  # Specify which GPU to use, in this case, GPU 0.
        accelerator='gpu',  # Use GPU acceleration for training.
        logger=logger,  # Use TensorBoard logger for tracking metrics.
        callbacks=[checkpoint_callback]  # Include callback for saving checkpoints.
    )

    trainer.fit(model, data_module)  # Start training the model with the data module.
    print(f"Training completed on chunk {chunk_id}")  # Indicate that training has finished for this chunk.

    # Create a flag file to indicate completion of training for this chunk
    open(f'training_complete_chunk_{chunk_id}.flag', 'w').close()

# Function to capture GPU memory usage with added variation

def capture_gpu(output_file='gpu_memory_usage.log', interval=0.1, num_chunks=7):
    # Initialize NVIDIA Management Library (NVML) to access GPU info
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()  # Get the number of GPU devices available.
    if device_count < 1:
        raise RuntimeError("No GPU devices found.")  # Ensure that there is at least one GPU available.

    start_time = time.time()  # Record the start time to measure elapsed time.

    # Initialize target memory usage for each simulated GPU
    target_memory_usage = [np.random.uniform(300, 500) for _ in range(num_chunks)]  # Random initial memory usage between 300-500 MB for each simulated GPU.
    
    # Parameters for memory usage simulation
    amplitude = 50  # Maximum deviation from base memory, simulates fluctuations in GPU memory usage.
    frequency = 0.1  # Frequency of the sine wave used for simulating periodic changes in memory usage.
    noise_level = 20  # Standard deviation of random noise added to simulate irregularities in memory usage.
    ramp_up_duration = 10  # Time in seconds over which memory usage ramps up from 0 to target value, simulating gradual increase in usage.

    with open(output_file, 'w') as f:
        # Write header for the log file
        f.write('Time')
        for i in range(num_chunks):
            f.write(f',sGPU_{i}_Memory_Usage(MB)')
        f.write('\n')

        try:
            while not all(os.path.exists(f'training_complete_chunk_{i}.flag') for i in range(num_chunks)):
                # Continue capturing GPU memory usage until all training chunks are complete
                current_time = time.time() - start_time  # Calculate the elapsed time.
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get handle for the first GPU.
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # Fetch current memory usage information.
                total_memory = memory_info.used / 1024 ** 2  # Convert memory usage from bytes to MB.

                simulated_memory_usage = []
                for i, target_usage in enumerate(target_memory_usage):
                    # Calculate ramp-up factor (from 0 to 1 over ramp_up_duration)
                    ramp_up_factor = min(current_time / ramp_up_duration, 1)
                    
                    # Calculate base memory usage based on the ramp-up factor
                    base_usage = target_usage * ramp_up_factor
                    
                    # Simulate memory usage with a sine wave (periodic variation) and random noise (irregularities)
                    sine_component = amplitude * np.sin(2 * np.pi * frequency * current_time + i)
                    noise_component = np.random.normal(0, noise_level)
                    usage = base_usage + (sine_component + noise_component) * ramp_up_factor
                    
                    # Ensure simulated memory usage is within realistic bounds (0 to 2000 MB)
                    usage = max(0, min(usage, 2000))
                    simulated_memory_usage.append(usage)  # Append the calculated memory usage for this simulated GPU.

                # Print current time and GPU memory usage for debugging purposes
                print(f'Current Time: {current_time:.2f}')
                print(f'Total GPU Memory Usage: {total_memory:.2f}')
                print(f'Simulated GPU Memory Usage: {[f"{usage:.2f}" for usage in simulated_memory_usage]}')

                # Write the current time and simulated memory usage to the log file
                f.write(f'{current_time:.2f}')
                for usage in simulated_memory_usage:
                    f.write(f',{usage:.2f}')
                f.write('\n')
                f.flush()  # Flush output to ensure it is written to file.
                time.sleep(interval)  # Pause for a short interval before capturing the next set of data.
        except KeyboardInterrupt:
            print("Capture interrupted by user.")  # Gracefully handle user interruption.
        finally:
            pynvml.nvmlShutdown()  # Clean up NVML resources.

# Function to plot GPU memory usage

def plot_gpu_memory(input_file='gpu_memory_usage.log', num_chunks=7):
    try:
        # Read the GPU memory usage log file
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The input file {input_file} was not found.")  # Handle case where the log file does not exist.
    except pd.errors.EmptyDataError:
        raise ValueError(f"The input file {input_file} is empty.")  # Handle case where the log file is empty.

    plt.figure(figsize=(12, 8))  # Set the size of the plot.
    for i in range(num_chunks):
        # Plot memory usage for each simulated GPU over time
        plt.plot(data['Time'], data[f'sGPU_{i}_Memory_Usage(MB)'], label=f'sGPU {i}')
    
    plt.xlabel('Time (seconds)')  # Label the x-axis.
    plt.ylabel('GPU Memory Usage (MB)')  # Label the y-axis.
    plt.title('Simulated GPU Memory Usage Over Time')  # Set the title of the plot.
    plt.legend()  # Add a legend to distinguish between different simulated GPUs.
    plt.grid(True)  # Add a grid for better readability.
    
    plt.savefig('gpu_memory_usage.png')  # Save the plot as a PNG file.
    plt.show()  # Display the plot on the screen.

# Main function to run the script

def main():
    num_chunks = 7  # Define the number of simulated GPUs.
    max_epochs = 1  # Define the number of epochs for training each chunk.

    # Start GPU monitoring in a separate process
    monitor_process = mp.Process(target=capture_gpu, args=('gpu_memory_usage.log', 0.1, num_chunks))
    monitor_process.start()  # Start the GPU monitoring process.

    # Create and start training processes for each simulated GPU
    processes = []
    for i in range(num_chunks):
        p = mp.Process(target=train_on_chunk, args=(i, max_epochs))  # Create a new process for each chunk.
        p.start()  # Start the process.
        processes.append(p)  # Add the process to the list.

    # Wait for all training processes to complete
    for p in processes:
        p.join()  # Wait for the process to finish.

    # Terminate GPU monitoring process after training completes
    monitor_process.terminate()  # Stop the GPU monitoring process.
    monitor_process.join()  # Wait for the GPU monitoring process to terminate.

    # Plot the GPU memory usage after all processes are complete
    plot_gpu_memory('gpu_memory_usage.log', num_chunks)

# Run the main function if this script is executed

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Set the start method for multiprocessing to 'spawn' for compatibility across platforms.
    main()  # Execute the main function.
