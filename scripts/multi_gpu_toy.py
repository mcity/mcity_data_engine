import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.models import resnet18, vgg16
from concurrent.futures import ProcessPoolExecutor, wait
import time
from torch.utils.tensorboard import SummaryWriter

# Function to perform inference with a specific model on a specific GPU
def run_inference(device, model_name, dataloader, index_run):
    torch.cuda.set_device(device)
    
    #Tensorboard logging
    writer = SummaryWriter(log_dir=f"logs/tensorboard/teacher_zeroshot_dev/{index_run}")

    # Load the model
    if model_name == "resnet":
        model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif model_name == "vgg":
        model = vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    else:
        raise ValueError("Invalid model name. Choose 'ResNet' or 'VGG'.")
    
    # Send model to the specified GPU
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()  # Example criterion to calculate loss
    total_loss = 0.0
    correct = 0
    total = 0

    print(f"Running {model_name} on GPU {device}...")

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(dataloader):
            time_start = time.time()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            time_end = time.time()
            duration = time_end - time_start
            batches_per_second = 1/duration
            writer.add_scalar(f'inference/batches_per_second', batches_per_second, step)

    print(f"{model_name} on GPU {device}: Accuracy: {100 * correct / total:.2f}%, Loss: {total_loss:.4f}")
    writer.close()

# Load CIFAR-100 dataset
def load_cifar100():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return dataset



if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    #Hardware configuration
    n_cpus = os.cpu_count() # https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
    n_gpus = torch.cuda.device_count()
    print(f"CPU count: {n_cpus}. GPU count: {n_gpus}")

    # Dataset and configurations
    dataset = load_cifar100()
    batch_size=128

    # Define models and dataset splits per model (1 = whole dataset)
    models_splits_dict = {
        "resnet": 1,
        "vgg": 2
    }

    runs_dict = {}

    n_samples = len(dataset)
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
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=24)
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

    # Create processes for each GPU/model
    with ProcessPoolExecutor() as executor:
        time_start = time.time()
        futures = []
        for run_id, run_metadata in runs_dict.items():
            index = run_id
            device  = f"cuda:{run_id}"
            model_name = run_metadata["model_name"]
            dataloader = run_metadata["dataloader"]
            future = executor.submit(run_inference, device, model_name, dataloader, index)
            futures.append(future)

        # Wait for all tasks to complete
        wait(futures)

        time_end = time.time()
        duration = time_end - time_start
        print(f"All processes complete after {duration} seconds")

