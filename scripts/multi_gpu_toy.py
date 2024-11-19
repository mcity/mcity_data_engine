import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, vgg16
from multiprocessing import Process

# Function to perform inference with a specific model on a specific GPU
def run_inference(device_id, model_name, dataloader):
    torch.cuda.set_device(device_id)
    
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
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device_id), labels.to(device_id)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    print(f"{model_name} on GPU {device_id}: Accuracy: {100 * correct / total:.2f}%, Loss: {total_loss:.4f}")

# Load CIFAR-10 dataset
def load_cifar10(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Dataset and configurations
    dataloader = load_cifar10()
    gpus = [0, 1]  # Two GPUs
    models = ["ResNet", "VGG"]

    # Create processes for each GPU/model
    processes = []
    for gpu, model_name in zip(gpus, models):
        p = Process(target=run_inference, args=(gpu, model_name, dataloader))
        p.start()
        processes.append(p)

    # Join processes
    for p in processes:
        p.join()