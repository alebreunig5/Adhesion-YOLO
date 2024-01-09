"""
This script performs training using YOLOv5 models on a dataset and logs the results using Weights & Biases (W&B).

The script imports necessary modules, handles warnings, and retrieves system information. It then initializes W&B and defines the parameters for YOLOv5 training. The script iterates through a list of models, configures W&B for each model, and trains the model using the specified parameters. The results are logged using W&B.

Parameters:
    - warnings: The 'warnings' module is imported to handle warnings.
    - YOLO: The YOLO class from the 'ultralytics' module is imported for working with YOLOv5.
    - wandb: The 'wandb' module is imported for integrating Weights & Biases into the project.
    - time: The 'time' module is imported for handling time-related operations.
    - GPUtil: The 'GPUtil' module is imported for retrieving GPU information.
    - platform: The 'platform' module is imported for retrieving system information.

Returns:
    - None

Note:
    - Ensure that the necessary modules and dependencies are installed before running the script.
    - Weights & Biases API key should be provided in the WANDB_API_KEY variable.

Instructions to install the necessary modules:

1. Open a terminal or command prompt on your operating system.

2. Ensure that you have Python installed on your system. You can check by running the following command:

   python --version

   If you don't have Python installed, you can download and install it from the official Python website (https://www.python.org).

3. Once you have Python installed, you can install the necessary modules using the pip package manager. Run the following commands one by one:

   pip install ultralytics
   pip install wandb
   pip install gputil

   These commands will install the 'ultralytics', 'wandb', and 'gputil' modules required to run the code.

4. After the installation is complete, you can run the code without any issues.

Please note that you may need administrator or superuser permissions to install the modules on your system. If you are using a virtual environment, make sure to activate it before running the installation commands.
"""

# Import the 'warnings' module to handle warnings.
import warnings

# Ignore warnings to prevent them from being printed to the output.
warnings.filterwarnings("ignore")

# Import the YOLO class from the 'ultralytics' module. This is for working with YOLOv5.
from ultralytics import YOLO

# Import the 'wandb' module, which is used to integrate Weights & Biases into the project.
import wandb

# Import el module to load training options from training_config.yaml file
import yaml

# Import GPUtil module for retrieving GPU information.
import GPUtil

# Import the 'platform' module for retrieving system information.
import platform

# Import os module
import os

# Import sys module
import sys

# Print a message to indicate the successful loading of modules.
print('Modules loaded successfully')

# Get information about the operating system.
operating_system = platform.system()
system_version = platform.version()

# Get information about the machine's name.
machine_name = platform.node()

# Get information about the machine's architecture.
architecture = platform.architecture()

# Get information about the processor.
processor = platform.processor()

# Print the obtained information.
print('=== System Information ===')
print(f"Operating System: {operating_system} ({system_version})")
print(f"Machine Name: {machine_name}")
print(f"Architecture: {architecture[0]} {architecture[1]}")
print(f"Processor: {processor}")

# Get a list of all available GPUs on the computer.
gpus = GPUtil.getGPUs()

# Check if any GPUs were found.
if len(gpus) == 0:
    print("No GPUs were found on this computer.")
else:
    # Get detailed information about each GPU.
    for i, gpu in enumerate(gpus):
        print(f"GPU {i + 1}:")
        print(f"Name: {gpu.name}")
        print(f"ID: {gpu.id}")
        print(f"Driver: {gpu.driver}")
        print(f"GPU Usage: {gpu.load * 100}%")
        print(f"Total Memory: {gpu.memoryTotal} MB")
        print(f"Available Memory: {gpu.memoryFree} MB")
        print(f"Used Memory: {gpu.memoryUsed} MB")
        print(f"Temperature: {gpu.temperature}°C")
        print("\n")

# Print a message to indicate the successful loading of modules.
print('Modules loaded successfully')

# Define a default configuration with generic options
default_config = {
    'wandb_api_key': 'YOUR_API_KEY',
    'img_size': 320,
    'epochs': 100,
    'batch_size': 8,
    'patience': 10,
    'verbose': True,
    'data': 'dataset/data.yaml',
    'model_list': [
        'yolov3u', 'yolov3-tiny',
        'yolov5nu.pt', 'yolov5su.pt', 'yolov5mu.pt', 'yolov5lu.pt', 'yolov5xu.pt',
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
    ],
    'dataset': 'Dataset by Experts',
    'running_pc': 'GPU',
    'subtitle': 'With Augmentation',
    'architecture': 'You Only Look Once',
    'project': 'New Paper',
    'wandb_mode': 'online'
}

# Check if the YAML configuration file exists
if not os.path.exists('training_config.yaml'):
    # If it doesn't exist, create it with the default configuration
    with open('training_config.yaml', 'w') as config_file:
        yaml.dump(default_config, config_file, default_flow_style=False)
        print('Config file created with dummy default options. Please configure it acording to your requeriments and re-run this script.')
    sys.exit()  # Exit the script after creating the config file


# Load variables from the YAML file
with open('training_config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Log in to Weights & Biases (W&B).
wandb_api_key = config['wandb_api_key']
print('Logging in with WANDB')
wandb.login(key=wandb_api_key)
print('WANDB login completed')

# Define the parameters for YOLO.
img_size = config['img_size']
epochs = config['epochs']
batch_size = config['batch_size']
patience = config['patience']
verbose = config['verbose']
data = config['data']
model_list = config['model_list']
dataset = config['dataset']
running_pc = config['running_pc']
subtitle = config['subtitle']
architecture = config['architecture']

# Define the parameters for W&B.
project = config['project']
wandb_mode = config['wandb_mode']

# Configure and start a new W&B run.
# Iterate through the model list.
for model in model_list:
    wandb.init(
        project=project,
        name=f"Architecture {architecture}, Model {model} {subtitle}",
        save_code=True,
        notes=f"Using {model} on {running_pc}",
        group=f"{architecture}_{subtitle}",
        mode=wandb_mode,
        settings=wandb.Settings(start_method="fork"),
        config={
            "architecture": architecture,
            "model": model,
            "dataset": dataset,
            "epochs": epochs,
            "data_yaml": data,
            "image_size": img_size,
            "batch_size": batch_size,
            "patience": patience,
            "augmentation": "Yes",
            "computer": running_pc,
            "Operating System": f"{operating_system} {system_version}",
            "Machine Name": machine_name,
            "Architecture": f"{architecture[0]} {architecture[1]}",
            "Processor": processor,
        }
    )
    # Print a message to indicate the successful loading of modules and W&B configuration.
    print('Module loading and W&B configuration completed')

    # Create an instance of the YOLOv8x model by loading a pre-trained model (recommended for training).
    model_yolo = YOLO(model)

    # Configure the training parameters and train the model.
    results = model_yolo.train(
        data=data,                # Path to the YAML file with data configuration.
        epochs=epochs,            # Number of training epochs.
        imgsz=img_size,           # Input image size.
        patience=patience,        # Number of patience epochs.
        batch_size=batch_size,    # Batch size.
        cache=True,               # Do not cache training data.
        val=True,                 # Perform validation.
        verbose=verbose,          # Show detailed information during training.
        save_json=True,           # Save results in JSON format.
        save=True,                # Save the trained model.
        optimizer='Adam',         # Optimizer to use (Adam in this case).
        warmup_epochs=20.0,       # Number of warm-up epochs.
        pretrained=True,          # Use a pre-trained model.
        task='detect',            # Model task (in this case, 'detect').
        mode='train',             # Training mode.
        iou=0.5,                  # IoU threshold for detection.
        plots=True,               # Generate plots during training.
        save_txt=True,            # Save results in text files.
        save_conf=True,           # Save configuration files.
        save_crop=True,           # Save cropped images.
        show_labels=True,         # Show labels.
        line_width=1,             # Line width for visualization. Can't make it smaller.
        visualize=True,           # Visualize during training.
        augment=True,             # Apply data augmentation. This generate a new YAML with the techniques applyed.
        boxes=True,               # Train detection boxes.
        optimize=True,            # Optimize the model.
        lr0=0.001,                # Initial learning rate. We should be optimizing this in the future
        lrf=0.1,                  # Learning rate reduction factor. Need optimization.
        nbs=64,                   # Batch size during training. Naeed optimization.
    )

    # Finish the Weights & Biases (W&B) run. Apparently this is only necessary when running on Jupyter
    wandb.finish()

# Print a message to indicate the completion of the execution. 
print('Execution completed')
