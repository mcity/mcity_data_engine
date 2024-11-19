import os
import sys

sys.path.append("..")
import fiftyone as fo

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.models import resnet18, vgg16
from concurrent.futures import ProcessPoolExecutor, wait
import time
import datetime
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import FiftyOneTorchDatasetCOCO

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Function to perform inference with a specific model on a specific GPU
def run_inference(device, model_name, dataloader, batch_size, index_run):
    print(f"Process ID: {os.getpid()}, Device: {device}, Model: {model_name}")
    run_successfull = True
    try:
        torch.cuda.set_device(device)
        object_classes = ["pedestrian", "vehicle", "cyclist"]

        #Tensorboard logging
        log_dir_root = "../logs/tensorboard/teacher_zeroshot"
        experiment_name = f"{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        log_directory = os.path.join(log_dir_root, experiment_name)
        writer = SummaryWriter(log_dir=log_directory)

        # Load the model
        if model_name == "omlab/omdet-turbo-swin-tiny-hf":
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            batch_classes = [object_classes] * batch_size
        else:
            raise ValueError("Invalid model name. Choose 'ResNet' or 'VGG'.")
        
        model = model.to(device)

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
        writer.close()
        return run_successfull

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"   # Warning from huggingface/tokenizers
    torch.cuda.empty_cache()
    
    #Hardware configuration
    n_cpus = os.cpu_count() # https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
    n_gpus = torch.cuda.device_count()
    print(f"CPU count: {n_cpus}. GPU count: {n_gpus}")

    # Dataset and configurations
    dataset_v51_orig = fo.load_dataset("dbogdollumich/mcity_fisheye_v51")
    dataset_v51 = dataset_v51_orig.take(101)

    dataset = FiftyOneTorchDatasetCOCO(dataset_v51)
    batch_size=8

    dataloader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True, collate_fn=lambda batch: list(zip(*batch)))

    # Test 'run_inference'
    run_successfull = run_inference("cuda:0", "omlab/omdet-turbo-swin-tiny-hf", dataloader_test, batch_size, 0)
    if run_successfull == False:
        print("Test run not successful")

    # Define models and dataset splits per model (1 = whole dataset)
    models_splits_dict = {
        "omlab/omdet-turbo-swin-tiny-hf": 2
    }

    runs_dict = {}

    n_samples = len(dataset)
    print(f"Dataset has {n_samples} samples.")
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
            subset = Subset(dataset, range(split_index_start, split_index_start + split_size))
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True, collate_fn=lambda batch: list(zip(*batch)))
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
            print(f"Launch job {run_id}")
            index = run_id
            device  = f"cuda:{run_id}"
            model_name = run_metadata["model_name"]
            dataloader = run_metadata["dataloader"]
            future = executor.submit(run_inference, device, model_name, dataloader, batch_size, index)
            futures.append(future)

        # Wait for all tasks to complete
        wait(futures)

        time_end = time.time()
        duration = time_end - time_start
        print(f"All processes complete after {duration} seconds")

