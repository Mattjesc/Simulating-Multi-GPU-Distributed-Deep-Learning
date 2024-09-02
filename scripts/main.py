import multiprocessing as mp
from training import train_on_chunk
from gpu_monitor import capture_gpu, plot_gpu_memory

def main():
    num_chunks = 7
    max_epochs = 1

    monitor_process = mp.Process(target=capture_gpu, args=('gpu_memory_usage.log', 0.1, num_chunks))
    monitor_process.start()

    processes = []
    for i in range(num_chunks):
        p = mp.Process(target=train_on_chunk, args=(i, max_epochs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    monitor_process.terminate()
    monitor_process.join()

    plot_gpu_memory('gpu_memory_usage.log', num_chunks)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
