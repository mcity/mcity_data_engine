import logging
import subprocess
import time

import psutil
import torch
import torch.multiprocessing as mp


class Distributer:
    def __init__(self):
        self.cpu_cores = psutil.Process().cpu_affinity()
        self.n_cpu_cores = len(self.cpu_cores)
        self.n_gpus = torch.cuda.device_count()

        logging.info(f"CPU cores: {self.n_cpu_cores}. GPU count: {self.n_gpus}")

        # Check if GPUs are in necessary default compute mode (shared):
        gpu_modes = self.get_gpu_compute_modes()
        for i, mode in enumerate(gpu_modes):
            if not "default" in mode.lower():
                logging.error(f"GPU {i} is in {mode}. Needs to be 'Default'.")

    def get_gpu_compute_modes(self):
        """
        Retrieves the compute modes of all available GPUs using the `nvidia-smi` command.

        Returns:
            list: A list of strings where each string represents the compute mode of a GPU.
                  If an error occurs while running the `nvidia-smi` command, an empty list is returned.

        Raises:
            subprocess.CalledProcessError: If the `nvidia-smi` command fails to execute.
        """
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=compute_mode", "--format=csv,noheader"],
                                    capture_output=True, text=True, check=True)
            modes = result.stdout.strip().split("\n")
            return modes
        except subprocess.CalledProcessError as e:
            logging.error("Error running nvidia-smi:", e)
            return []

    def distribute_cpu_cores(self, cpu_cores, n_processes):
        """
        Distributes a list of CPU cores among a specified number of processes.

        Args:
            n_processes (int): The number of processes to distribute the CPU cores among.

        Returns:
            list: A list of lists, where each sublist contains the CPU cores assigned to a process.
        """
        n_cores = len(cpu_cores)

        chunk_size = n_cores // n_processes
        remainder = n_cores % n_processes

        cpu_cores_per_process = []
        start = 0
        for i in range(n_processes):
            # Determine the end index for this chunk
            end = start + chunk_size + (1 if i < remainder else 0)
            cpu_cores_per_process.append(cpu_cores[start:end])
            start = end

        return cpu_cores_per_process


class ZeroShotDistributer(Distributer):
    def __init__(self, config, n_samples, dataset_info, detector):
        super().__init__()      # Call the parent class's __init__ method
        self.config = config
        self.n_samples = n_samples
        self.dataset_info = dataset_info
        self.detector = detector

    def distribute_and_run(self):
        dataset_name = self.dataset_info["name"]
        models_dict = self.config["hf_models_zeroshot_objectdetection"]
        n_post_worker = self.config["n_post_processing_worker_per_inference_worker"]

        runs = []
        run_id = 0


        for model_name in models_dict:
            batch_size = models_dict[model_name]["batch_size"]
            n_chunks = models_dict[model_name]["n_dataset_chunks"]

            # Calculate the base split size and leftover samples
            chunk_size, leftover_samples = divmod(self.n_samples, n_chunks)

            chunk_index_start = 0
            chunk_index_end = None
            for split_id in range(n_chunks):

                # Prepare torch subsets
                if n_chunks == 1:
                    is_subset = False
                else:
                    is_subset = True
                    chunk_size += (leftover_samples if split_id == n_chunks - 1 else 0)
                    chunk_index_end = chunk_index_start + chunk_size

                # Add entry to runs
                runs.append({
                        "run_id": run_id,
                        "model_name": model_name,
                        "is_subset": is_subset,
                        "chunk_index_start": chunk_index_start,
                        "chunk_index_end": chunk_index_end,
                        "batch_size": batch_size,
                        "dataset_name": dataset_name
                    })


                # Update start index for next chunk
                if n_chunks > 1:
                    chunk_index_start += chunk_size

                run_id += 1

        n_runs = len(runs)

        logging.info(f"Running with multiprocessing on {self.n_gpus} GPUs.")
        n_parallel_processes = min(self.n_gpus, n_runs)

        n_post_processing_workers = int(n_post_worker * n_parallel_processes)
        n_total_workers = n_parallel_processes + n_post_processing_workers

        # Create results queues, max one per GPU
        result_queues = []
        max_queue_size = 3                                      # Balance between flexibility and available GPU memory for inference (data stays on GPU for post-processing)
        for index in range(n_parallel_processes):
            results_queue = mp.Queue(maxsize=max_queue_size)    # Ensure that no GPU memory overflow occurs
            result_queues.append(results_queue)

        # Dedicated worker that continuously measures the lengths of the result queues
        result_queues_sizes = mp.Manager().list([0] * n_parallel_processes)
        largest_queue_index = mp.Value('i', -1)
        size_updater = mp.Process(target=self.detector.update_queue_sizes_worker, args=(result_queues, result_queues_sizes, largest_queue_index, max_queue_size))
        size_updater.start()

        # Create queue with all planned runs
        task_queue = mp.Queue()
        for run_id, run_metadata in enumerate(runs):
            logging.info(f"Run {run_id} Metadata: {run_metadata}")
            task_queue.put(run_metadata)

        # Flags for synchronizations
        inference_finished = mp.Value('b', False)
        post_processing_finished = mp.Value('b', False)

        # Distribute CPU cores
        if (n_parallel_processes + n_post_processing_workers) > len(self.cpu_cores):
            logging.error(f"Launching {n_parallel_processes + n_post_processing_workers} processes with only {len(self.cpu_cores)} CPU cores.")

        cpu_cores_post_processing = self.cpu_cores[:n_post_processing_workers]
        cpu_cores_inference = self.cpu_cores[n_post_processing_workers:]
        cpu_cores_per_process = self.distribute_cpu_cores(cpu_cores_inference, n_parallel_processes)

        # Create post-processing worker processes
        post_processing_processes = []
        for i in range(n_post_processing_workers):
            p = mp.Process(target=self.detector.process_outputs_worker, args=(result_queues, result_queues_sizes, largest_queue_index, inference_finished))
            post_processing_processes.append(p)
            p.start()
            logging.info(f"Started post-processing worker {index}")

        # Create worker processes, max one per GPU
        inference_processes = []
        gpu_worker_done_events = [mp.Event() for _ in range(n_parallel_processes)]  # Signals when a GPU worker is done

        for index in range(n_parallel_processes):
            gpu_id = index
            cpu_cores_for_run = cpu_cores_per_process[gpu_id]
            results_queue = result_queues[index]
            done_event = gpu_worker_done_events[index]
            p = mp.Process(target=self.detector.gpu_worker, args=(gpu_id, cpu_cores_for_run, task_queue, results_queue, done_event, post_processing_finished))
            inference_processes.append(p)
            p.start()
            logging.info(f"Started inference worker {index} for GPU {gpu_id}")

        logging.info(f"Started {len(post_processing_processes)} post-processing workers and {len(inference_processes)} GPU inference workers.")

        # Wait for all inference tasks to complete
        while not all(worker_event.is_set() for worker_event in gpu_worker_done_events):
            continue
        logging.info("All workers have finished inference tasks.")
        inference_finished.value = True

        # Wait for results processing to finish
        for p in post_processing_processes:
            p.join()
        logging.info("Results processing worker has shut down.")
        post_processing_finished.value = True

        # Wait for workers to finish
        for p in inference_processes:
            p.join()
        logging.info("All inference workers have shut down.")

        # Process updating queue sizes
        size_updater.terminate()

        # Close queues
        task_queue.close()
        results_queue.close()
        logging.info("All multiprocessing queues are closed.")
