import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, vgg16
from concurrent.futures import ProcessPoolExecutor, wait
import time
from torch.utils.tensorboard import SummaryWriter

# Function to perform inference with a specific model on a specific GPU
def run_inference(device_id, model_name, dataloader, index_run):
    torch.cuda.set_device(device_id)
    
    #Tensorboard logging
    writer = SummaryWriter(log_dir=f"logs/tensorboard/teacher_zeroshot_dev/{index_run}")

    # Load the model
    if model_name == "ResNet":
        model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif model_name == "VGG":
        model = vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    else:
        raise ValueError("Invalid model name. Choose 'ResNet' or 'VGG'.")
    
    # Send model to the specified GPU
    model = model.to(device_id)
    model.eval()

    criterion = nn.CrossEntropyLoss()  # Example criterion to calculate loss
    total_loss = 0.0
    correct = 0
    total = 0

    print(f"Running {model_name} on GPU {device_id}...")

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(dataloader):
            time_start = time.time()

            inputs, labels = inputs.to(device_id), labels.to(device_id)
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

    print(f"{model_name} on GPU {device_id}: Accuracy: {100 * correct / total:.2f}%, Loss: {total_loss:.4f}")
    writer.close()

# Load CIFAR-10 dataset
def load_cifar10(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Dataset and configurations
    dataloader = load_cifar10()

    n_cpus = os.cpu_count() # https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
    n_gpus = torch.cuda.device_count()

    print(f"CPU count: {n_cpus}. GPU count: {n_gpus}")

    models_splits_dict = {
        "resnet": None,
        "vgg": 2
    }

    gpus = [0, 1]  # Two GPUs
    models = ["ResNet", "VGG"]

    # Create processes for each GPU/model
    with ProcessPoolExecutor() as executor:
        time_start = time.time()
        futures = []
        for index, (gpu, model_name) in enumerate(zip(gpus, models)):
            future = executor.submit(run_inference, gpu, model_name, dataloader, index)
            futures.append(future)

        # Wait for all tasks to complete
        wait(futures)

        time_end = time.time()
        duration = time_end - time_start
        print(f"All processes complete after {duration} seconds")

