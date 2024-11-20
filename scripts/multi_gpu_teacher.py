import os
import sys

sys.path.append("..")
import fiftyone as fo

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from torch.utils.data import DataLoader, Subset, Dataset, SubsetRandomSampler
from torchvision.models import resnet18, vgg16
from concurrent.futures import ProcessPoolExecutor, wait
import time
import datetime
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
from utils.data_loader import FiftyOneTorchDatasetCOCO

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Load CIFAR-10 dataset
def _get_cifar10_dataloader(batch_size=128, num_workers=24):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataset, dataloader

# Load CIFAR-100 dataset
def _get_cifar100_dataloader(batch_size=128, num_workers=24):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataset, dataloader

def _get_v51_dataloader(batch_size=8, max_samples=101, num_workers=24, start_index_chunk=None, stop_index_chunk=None):
    dataset_v51_orig = fo.load_dataset("dbogdollumich/mcity_fisheye_v51")
    dataset_v51 = dataset_v51_orig.take(max_samples)

    dataset = FiftyOneTorchDatasetCOCO(dataset_v51)
    if start_index_chunk and stop_index_chunk:  # Get subset of dataset
        dataset = Subset(dataset, range(start_index_chunk, stop_index_chunk))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=lambda batch: list(zip(*batch)))
    return dataset, dataloader


# Function to perform inference with a specific model on a specific GPU
def run_inference(device, model_name, dataset_name, dataloader, batch_size, index_run, runs_in_parallel):
    print(f"Process ID: {os.getpid()}, Run: {index_run}, Device: {device}, Model: {model_name}, Parallel: {runs_in_parallel}")
    run_successfull = True
    try:
        torch.cuda.set_device(device)
        object_classes = ["pedestrian", "vehicle", "cyclist"]

        #Tensorboard logging
        log_dir_root = f"../logs/tensorboard/teacher_zeroshot/{dataset_name}/"
        experiment_name = f"{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{device}"
        log_directory = os.path.join(log_dir_root, experiment_name)
        writer = SummaryWriter(log_dir=log_directory)

        # Load the model
        print("Loading model")
        if model_name == "omlab/omdet-turbo-swin-tiny-hf":
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            batch_classes = [object_classes] * batch_size
        else:
            raise ValueError("Invalid model name")
        
        model = model.to(device)

        print("Starting inference")
        with torch.no_grad():
            for step, (images, labels) in enumerate(dataloader):
                time_start = time.time()

                if len(images) < batch_size:
                    batch_classes = [object_classes] * len(images)

                #OmDet
                images = [to_pil_image(image) for image in images]
                inputs = processor(
                            text=batch_classes,
                            images=images,
                            return_tensors="pt",
                        ).to(device)

                outputs = model(**inputs)
                
                time_end = time.time()
                duration = time_end - time_start
                batches_per_second = 1/duration
                frames = len(images)
                frames_per_second = batches_per_second * frames
                writer.add_scalar(f'inference/frames_per_second', frames_per_second, step)

    except Exception as e:
        print(f"Error in Process {os.getpid()}: {e}")
        run_successfull = False

    finally:
        print(f"Finished run. Closing SummaryWriter.")
        writer.close()
        return run_successfull

if __name__ == "__main__":
    mp.set_start_method("forkserver")  # https://pytorch.org/docs/stable/notes/multiprocessing.html
    os.environ["TOKENIZERS_PARALLELISM"] = "true"   # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
    torch.cuda.empty_cache()
    
    #Hardware configuration
    n_cpus = os.cpu_count() # https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
    n_gpus = torch.cuda.device_count()
    print(f"CPU count: {n_cpus}. GPU count: {n_gpus}")

    batch_size = 128
    dataset_name = "cifar_10"
    test_inference = False

    # Load dataset and dataloader
    if dataset_name == "cifar_10":
        dataset, dataloader = _get_cifar10_dataloader(batch_size=batch_size)  
    elif dataset_name == "cifar_100":
        dataset, dataloader = _get_cifar100_dataloader(batch_size=batch_size)
    elif dataset_name == "v51":
        dataset, dataloader = _get_v51_dataloader(batch_size=batch_size) 

    n_samples = len(dataset)
    print(f"Dataset has {n_samples} samples.")

    # Test 'run_inference'
    if test_inference:
        sampler = SubsetRandomSampler(torch.randperm(len(dataset))[:batch_size*2])
        dataloader_test = DataLoader(dataset, batch_size=1, sampler=sampler)
        run_successfull = run_inference("cuda:0", "omlab/omdet-turbo-swin-tiny-hf", dataset_name, dataloader_test, batch_size, 0, False)
        if run_successfull == False:
            print("Test run not successful")

    # Define models and dataset splits per model (1 = whole dataset)
    models_splits_dict = {
        "omlab/omdet-turbo-swin-tiny-hf": {"batch_size": 256, "splits": 1},
        "omlab/omdet-turbo-swin-tiny-hf": {"batch_size": 256, "splits": 2}
    }

    runs_dict = {}

    
    run_counter = 0
    for model_name, n_splits in models_splits_dict.items():
        # Calculate the base split size and leftover samples
        split_size, leftover_samples = divmod(n_samples, n_splits)
        
        split_index_start = 0
        for split_id in range(n_splits):
            if run_counter >= n_gpus:
                print(f"Run {run_counter} will fail. Only {n_gpus} GPUs available.")

            # Generate torch subsets
            split_size = split_size + (leftover_samples if split_id == n_splits - 1 else 0)
            #subset = Subset(dataset, range(split_index_start, split_index_start + split_size))
            #dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True, collate_fn=lambda batch: list(zip(*batch)))
            split_index_start = split_size
            
            # Add entry to runs_dict
            runs_dict[run_counter] = {
                "model_name": model_name,
                "gpu_id": run_counter,
                "dataloader": dataloader,
                "split_id": split_id,
                "split_length": split_size,
            }
            run_counter += 1
    
    print(runs_dict)

    # Create processes for each GPU (model + dataset split)
    with ProcessPoolExecutor() as executor:
        time_start = time.time()
        futures = []
        for run_id, run_metadata in runs_dict.items():
            index = run_id
            device  = f"cuda:{run_id}"
            model_name = run_metadata["model_name"]
            dataloader = run_metadata["dataloader"]
            print(f"Launch job {index} - {device} - {model_name}")
            # run_successfull = run_inference(device, model_name, dataloader, batch_size, index, False)
            future = executor.submit(run_inference, device, model_name, dataset_name, dataloader, batch_size, index, True)
            futures.append(future)

        # Wait for all tasks to complete
        wait(futures)

    time_end = time.time()
    duration = time_end - time_start
    print(f"All processes complete after {duration} seconds")

