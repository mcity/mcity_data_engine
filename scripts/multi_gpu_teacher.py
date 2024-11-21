import os
import sys

sys.path.append("..")
import fiftyone as fo
import psutil

import wandb

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from torch.utils.data import DataLoader, Subset, Dataset, RandomSampler
from torchvision.models import resnet18, vgg16
from concurrent.futures import ProcessPoolExecutor, wait
import time
import datetime
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
from utils.data_loader import FiftyOneTorchDatasetCOCO

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, AutoConfig

class ZeroShotInferenceCollateFn:
    def __init__(self, hf_model_config_name, hf_processor, batch_size, object_classes, batch_classes):
        self.hf_model_config_name = hf_model_config_name
        self.processor = hf_processor
        self.batch_size = batch_size
        self.object_classes = object_classes
        self.batch_classes = batch_classes

    def __call__(self, batch):
        try:
            images, labels = zip(*batch)

            # Adjustments for final batch
            n_images = len(images)
            if n_images < self.batch_size:
                if self.hf_model_config_name == "OmDetTurboConfig":
                    self.batch_classes = [self.object_classes] * n_images
                elif self.hf_model_config_name == "Owlv2Config":
                    self.batch_classes = self.object_classes * n_images

            # Apply PIL transformation for specific models
            if self.hf_model_config_name == "OmDetTurboConfig":
                images = [to_pil_image(image) for image in images]

            inputs = self.processor(
                text=self.batch_classes,
                images=images,
                return_tensors="pt",
            )

            return inputs, labels
        except Exception as e:
            print(f"Error in collate function of DataLoader: {e}")
            traceback.print_exc()

def _terminate_processes(processes):
    """Helper to terminate all processes gracefully."""
    for p in processes:
        if p.is_alive():
            p.terminate()
        p.join()
    print("All processes terminated")

# Load CIFAR-10 dataset
def _get_cifar10(max_samples=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    if max_samples:
        dataset = Subset(dataset, range(max_samples))
    return dataset

def _get_v51(max_samples=None):
    dataset_v51_orig = fo.load_dataset("dbogdollumich/mcity_fisheye_v51")
    if max_samples:
        dataset_v51 = dataset_v51_orig.take(max_samples)
    else:
        dataset_v51 = dataset_v51_orig
    dataset = FiftyOneTorchDatasetCOCO(dataset_v51)
    return dataset

def _process_output(output):
    print("Process outputs")
    return True

def _distribute_cpu_cores(cpu_cores, n_processes):
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


# Function to perform inference with a specific model on a specific GPU
def run_inference(cpu_cores: list, dataset: torch.utils.data.Dataset, metadata: dict, runs_in_parallel: bool, num_workers:int = 4, prefetch_factor:int = 4):
    # Optional: Set CPU affinity to pin this process to specific cores
    # psutil.Process().cpu_affinity(cpu_cores)
    max_n_cpus = len(cpu_cores)
    
    # Set GPU
    gpu_id = metadata["gpu_id"]
    device = f"cuda:{gpu_id}"
    print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(device)

    wandb_run = None
    writer = None
    run_successfull = True
    try: 
        # Metadata
        model_name = metadata["model_name"]
        dataset_name = metadata["dataset_name"]
        is_subset = metadata["is_subset"]
        batch_size = metadata["batch_size"]
        
        print(f"Process ID: {os.getpid()}, Device: {device}, Model: {model_name}, Parallel: {runs_in_parallel}")        

        # Load the model
        print("Loading model")
        object_classes = ["pedestrian", "vehicle", "cyclist"]
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        model = model.to(device)
        hf_model_config = AutoConfig.from_pretrained(model_name)
        hf_model_config_name = type(hf_model_config).__name__
        
        if hf_model_config_name == "OmDetTurboConfig":
            batch_classes = [object_classes] * batch_size
        elif hf_model_config_name == "Owlv2Config":
            batch_classes = object_classes * batch_size
        else:
            raise ValueError("Invalid model name")

        # Dataloader
        print("Generating dataloader")
        if is_subset:
            chunk_index_start = metadata["chunk_index_start"]
            chunk_index_end = metadata["chunk_index_end"]
            print(f"Length of dataset: {len(dataset)}")
            print(f"Subset start index: {chunk_index_start}")
            print(f"Subset stop index: {chunk_index_end}")
            dataset = Subset(dataset, range(chunk_index_start, chunk_index_end))
        
        zero_shot_inference_preprocessing = ZeroShotInferenceCollateFn(hf_model_config_name=hf_model_config_name, hf_processor=processor, object_classes=object_classes, batch_size=batch_size, batch_classes=batch_classes)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, collate_fn=zero_shot_inference_preprocessing)

        # Weights and Biases
        wandb_run = wandb.init(
            name=f"{model_name}_{device}",
            sync_tensorboard=True,
            job_type="inference",
            project="Teacher Dev",
        )

        #Tensorboard logging
        log_dir_root = f"logs/tensorboard/teacher_zeroshot/{dataset_name}/"
        experiment_name = f"{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{device}"
        log_directory = os.path.join(log_dir_root, experiment_name)
        writer = SummaryWriter(log_dir=log_directory)

        # Inference Loop
        print("Starting inference")
        n_processed_images = 0
        for inputs, labels in dataloader:
            time_start = time.time()
            n_images = len(labels)  # inputs is already processed
            inputs.to(device)

            with torch.amp.autocast("cuda"), torch.no_grad():
                outputs = model(**inputs)

            #with ProcessPoolExecutor(max_workers=max_n_cpus) as executor:
            #    futures = [executor.submit(_process_output, output) for output in outputs]

            time_end = time.time()
            duration = time_end - time_start
            batches_per_second = 1 / duration
            frames_per_second = batches_per_second * n_images
            n_processed_images += n_images
            writer.add_scalar(f'inference/frames_per_second', frames_per_second, n_processed_images)

        wandb_run.finish(exit_code=0)

    except Exception as e:
        print(f"Error in Process {os.getpid()}: {e}")
        wandb_run.finish(exit_code=1)
        run_successfull = False

    finally:
        print(f"Finished run. Closing SummaryWriter.")
        if writer:
            writer.close()
        torch.cuda.empty_cache()
        return run_successfull

if __name__ == "__main__":
    mp.set_start_method("spawn")                    # https://pytorch.org/docs/stable/notes/multiprocessing.html
    torch.backends.cudnn.benchmark = True           # https://pytorch.org/docs/stable/backends.html
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
    
    #Hardware configuration
    n_cpus_os = os.cpu_count()  # https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
    n_cpus_mp = mp.cpu_count()
    n_cpus_cluster = len(psutil.Process().cpu_affinity())   # As requested in Lighthouse job
    n_cpus = min(n_cpus_os, n_cpus_mp, n_cpus_cluster)               
    n_gpus = torch.cuda.device_count()
    print(f"OS CPU count: {n_cpus_os}. MP CPU count: {n_cpus_mp}. Cluster CPU count: {n_cpus_cluster}. Utilized CPU cores: {n_cpus}. GPU count: {n_gpus}")

    dataset_name = "cifar_10"

    # Load dataset and dataloader
    if dataset_name == "cifar_10":
        dataset = _get_cifar10(max_samples=None)  
    elif dataset_name == "v51":
        dataset = _get_v51(max_samples=None) 

    n_samples = len(dataset)
    print(f"Dataset has {n_samples} samples.")

    # Define models and dataset chunks per model (1 = whole dataset)
    models_splits_dict = {
        "omlab/omdet-turbo-swin-tiny-hf": {"batch_size": 256, "dataset_chunks": 1},
        "google/owlv2-base-patch16": {"batch_size": 32, "dataset_chunks": 2}
    }

    # Prepare dictionary for process execution
    runs_dict = {}
    run_counter = 0
    for model_name in models_splits_dict:
        batch_size = models_splits_dict[model_name]["batch_size"]
        n_chunks = models_splits_dict[model_name]["dataset_chunks"]
        
        # Calculate the base split size and leftover samples
        chunk_size, leftover_samples = divmod(n_samples, n_chunks)
        
        chunk_index_start = 0
        chunk_index_end = None
        for split_id in range(n_chunks):
            if run_counter >= n_gpus:
                print(f"Run {run_counter} will fail. Only {n_gpus} GPUs available.")

            # Generate torch subsets
            if n_chunks == 1:
                is_subset = False
            else:
                is_subset = True
                chunk_size += (leftover_samples if split_id == n_chunks - 1 else 0)
                chunk_index_end = chunk_index_start + chunk_size
            
            # Add entry to runs_dict
            runs_dict[run_counter] = {
                    "model_name": model_name,
                    "gpu_id": run_counter,
                    "is_subset": is_subset,
                    "chunk_index_start": chunk_index_start,
                    "chunk_index_end": chunk_index_end,
                    "batch_size": batch_size,
                    "dataset_name": dataset_name
                }

            # Update start index for next chunk
            if n_chunks > 1:
                chunk_index_start = chunk_size

            run_counter += 1
    
    # Create processes for each GPU (model + dataset split)
    n_processes = len(runs_dict)

    # Distribute CPU cores
    cpu_cores = psutil.Process().cpu_affinity()
    cpu_cores_per_process = _distribute_cpu_cores(cpu_cores, n_processes)

    processes = []
    test_inference = False

    time_start = time.time()
    try:
        for run_id, run_metadata in runs_dict.items():
            if test_inference:  # All CPU cores
                test_result = run_inference(cpu_cores, dataset, run_metadata, False)
                print(f"Test ran successful: {test_result}")
            else:
                cpu_cores_for_run = cpu_cores_per_process[run_id]
                p = mp.Process(target=run_inference, args=(cpu_cores_for_run, dataset, run_metadata, True))
                processes.append(p)
                p.start()

        # Wait for all tasks to complete
        for p in processes:
            p.join()
    except Exception as e:
        print(f"Error during multiprocessing: {e}")
        traceback.print_exc()
    finally:
        _terminate_processes(processes)

        time_end = time.time()
        duration = time_end - time_start
        print(f"All processes complete after {duration} seconds")