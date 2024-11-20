import os
import sys

sys.path.append("..")
import fiftyone as fo

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

# Load CIFAR-10 dataset
def _get_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return dataset

def _get_v51():
    dataset_v51_orig = fo.load_dataset("dbogdollumich/mcity_fisheye_v51")
    dataset = FiftyOneTorchDatasetCOCO(dataset_v51)
    return dataset

def _collate_fn(batch):
    return list(zip(*batch))

# Function to perform inference with a specific model on a specific GPU
def run_inference(dataset: torch.utils.data.Dataset, metadata: dict, max_n_cpus: int, runs_in_parallel: bool):
    print("Job started")
    wandb_run = None
    writer = None
    run_successfull = True
    try:        
        # Metadata
        model_name = metadata["model_name"]
        dataset_name = metadata["dataset_name"]
        gpu_id = metadata["gpu_id"]
        is_subset = metadata["is_subset"]
        batch_size = metadata["batch_size"]
        device  = f"cuda:{gpu_id}"
        print(f"Process ID: {os.getpid()}, Device: {device}, Model: {model_name}, Parallel: {runs_in_parallel}")        
        
        # Dataloader
        if is_subset:
            chunk_index_start = metadata["chunk_index_start"]
            chunk_index_end = metadata["chunk_index_end"]
            print(f"Length of dataset: {len(dataset)}")
            print(f"Subset start index: {chunk_index_start}")
            print(f"Subset stop index: {chunk_index_end}")
            dataset = Subset(dataset, range(chunk_index_start, chunk_index_end))
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=max_n_cpus, pin_memory=True, collate_fn=_collate_fn)

        torch.cuda.set_device(device)
        object_classes = ["pedestrian", "vehicle", "cyclist"]

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

        # Load the model
        print("Loading model")
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        model = model.to(device)

        hf_model_config = AutoConfig.from_pretrained(model_name)
        if type(hf_model_config).__name__ == "OmDetTurboConfig":
            batch_classes = [object_classes] * batch_size
        elif type(hf_model_config).__name__ == "Owlv2Config":
            batch_classes = object_classes * batch_size
        else:
            raise ValueError("Invalid model name")
        
        
        print("Starting inference")
        n_processed_images = 0
        with torch.no_grad():
            for images, labels in dataloader:
                time_start = time.time()
                n_images = len(images)

                # Adjustments for final batch
                if n_images < batch_size:
                    if type(hf_model_config).__name__ == "OmDetTurboConfig":
                        batch_classes = [object_classes] * n_images
                    elif type(hf_model_config).__name__ == "Owlv2Config":
                        batch_classes = object_classes * n_images

                #OmDet
                if type(hf_model_config).__name__ == "OmDetTurboConfig":
                    images = [to_pil_image(image) for image in images]
                else:
                    images = [(image).to(device, non_blocking=True) for image in images]

                inputs = processor(
                    text=batch_classes,
                    images=images,
                    return_tensors="pt",
                        ).to(device)

                outputs = model(**inputs)
                
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
    mp.set_start_method("forkserver")               # https://pytorch.org/docs/stable/notes/multiprocessing.html
    os.environ["TOKENIZERS_PARALLELISM"] = "true"   # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
    
    #Hardware configuration
    n_cpus = os.cpu_count() # https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
    n_gpus = torch.cuda.device_count()
    print(f"CPU count: {n_cpus}. GPU count: {n_gpus}")

    dataset_name = "cifar_10"

    # Load dataset and dataloader
    if dataset_name == "cifar_10":
        dataset = _get_cifar10()  
    elif dataset_name == "v51":
        dataset = _get_v51() 

    n_samples = len(dataset)
    print(f"Dataset has {n_samples} samples.")

    # Define models and dataset splits per model (1 = whole dataset)
    models_splits_dict = {
        "omlab/omdet-turbo-swin-tiny-hf": {"batch_size": 256, "dataset_chunks": 1},
        "google/owlv2-base-patch16": {"batch_size": 32, "dataset_chunks": 2}
    }

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
    test_inference = False
    max_n_cpus = n_cpus // len(runs_dict)
    print(f"Max CPU per process: {max_n_cpus}")
    with ProcessPoolExecutor() as executor:
        time_start = time.time()
        futures = []
        for run_id, run_metadata in runs_dict.items():
            print(f"Launch job {run_id}: {run_metadata}")
            if test_inference:
                run_successfull = run_inference(dataset, run_metadata, max_n_cpus, False)
                print(f"Test run successfull: {run_successfull}")
            else:
                future = executor.submit(run_inference, dataset, run_metadata, max_n_cpus, True)
                futures.append(future)

        # Wait for all tasks to complete
        wait(futures)

    time_end = time.time()
    duration = time_end - time_start
    print(f"All processes complete after {duration} seconds")