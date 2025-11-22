# Cloud-Edge Framework for Neural Network Pruning

A distributed cloud-edge framework for automatic optimization of neural networks through intelligent model dispatching and structured pruning. The system supports both Large Language Models (LLMs) and Convolutional Neural Networks (CNNs) with specialized pruning strategies.

## Overview

This framework implements a cloud-edge architecture that enables the optimization of pre-trained models from Hugging Face through adaptive pruning techniques. The system features an intelligent dispatcher that routes models to specialized pruning engines based on their architecture type.

### Architecture
```
┌─────────────┐         ┌──────────────────────────────┐
│   Client    │────────>│       Cloud Service          │
│   (Edge)    │<────────│                              │
└─────────────┘         │  ┌────────────────────────┐  │
     │                  │  │       Dispatcher       │  │
     │                  │  │    (Model Detection)   │  │
     │                  │  └──────────┬─────────────┘  │
     │                  │             │                │
     │                  │      ┌──────┴──────┐         │
     │                  │      │             │         │
     │                  │  ┌───▼───┐    ┌────▼─────┐   │
     │                  │  │  SiRE │    │ImproveNet│   │
     │                  │  │ (LLM) │    │  (CNN)   │   │
     │                  │  └───────┘    └──────────┘   │
     │                  │                              │
     │                  │         JSON Export          │
     │                  └──────────────────────────────┘
     │
     └─ Apply Pruning Config
```

## Features

- **Intelligent Model Dispatcher**: Automatically detects model architecture (LLM vs CNN)
- **SiRE Engine**: Specialized pruning for Transformer-based LLMs (attention heads and feed-forward neurons)
- **ImproveNet Engine**: Optimized pruning for CNNs (filters and channels)
- **Automatic loading** of models from Hugging Face Hub
- **Structured output** in JSON format with precise pruning indices
- **Flexible configuration** for different tasks and architectures

## Dispatcher Logic

The dispatcher module analyzes the model architecture and routes it to the appropriate pruning engine:
```python
Model Input → Dispatcher
                │
                ├─ If Transformer/LLM → SiRE Engine
                │   (BERT, GPT, T5, RoBERTa, etc.)
                │
                └─ If CNN → ImproveNet Engine
                    (ResNet, VGG, EfficientNet, etc.)
```

### Detection Criteria

**LLM Detection:**
- Presence of attention layers
- Transformer architecture components
- Model types: BERT, GPT, T5, RoBERTa, ELECTRA, etc.

**CNN Detection:**
- Convolutional layers
- Pooling layers
- Model types: ResNet, VGG, EfficientNet, MobileNet, etc.

## Output Format

### LLM Pruning (SiRE)
```json
{
  "model_name": "bert-base-uncased",
  "model_type": "llm",
  "task": "text-classification",
  "pruning_engine": "SiRE",
  "pruning_config": {
    "attention_heads": [
      {
        "layer": 0,
        "head": 3
      },
      {
        "layer": 2,
        "head": 7
      }
    ],
    "feed_forward_neurons": [
      {
        "layer": 1,
        "neuron_indices": [45, 127, 256, 892]
      },
      {
        "layer": 5,
        "neuron_indices": [12, 98, 345]
      }
    ]
  },
  "pruning_ratio": {
    "heads": 0.15,
    "neurons": 0.20
  },
  "metadata": {
    "original_parameters": 110000000,
    "pruned_parameters": 88000000,
    "compression_ratio": 0.20
  }
}
```

### CNN Pruning (ImproveNet)
```json
{
  "model_name": "resnet50",
  "model_type": "cnn",
  "task": "image-classification",
  "pruning_engine": "ImproveNet",
  "pruning_config": {
    "filters": [
      {
        "layer": "conv1",
        "filter_indices": [5, 12, 23, 45]
      },
      {
        "layer": "layer2.0.conv1",
        "filter_indices": [8, 15, 27, 33, 41]
      }
    ],
    "channels": [
      {
        "layer": "layer3.1.conv2",
        "channel_indices": [10, 25, 67, 89, 102]
      }
    ]
  },
  "pruning_ratio": {
    "filters": 0.25,
    "channels": 0.18
  },
  "metadata": {
    "original_parameters": 25500000,
    "pruned_parameters": 19500000,
    "compression_ratio": 0.235,
    "original_flops": "4.1B",
    "pruned_flops": "3.1B"
  }
}
```

## Requirements

### Cloud Service
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- torchvision
- Flask/FastAPI
- NumPy

### Edge Device
- Python 3.8+
- Requests
- PyTorch (for applying pruning)
- Transformers
- torchvision

## Installation
```bash
# Clone the repository
git clone https://github.com/your-username/cloud-edge-pruning-framework.git
cd cloud-edge-pruning-framework

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Starting the Cloud Service
```bash
python cloud_service.py --port 8000
```

### Request from Edge Device
```python
import requests

# Request parameters (LLM example)
payload = {
    "model_name": "bert-base-uncased",
    "task": "text-classification"
}

# Send request to cloud
response = requests.post(
    "http://cloud-service:8000/api/prune",
    json=payload
)

# Receive pruning configuration
pruning_config = response.json()
print(f"Detected model type: {pruning_config['model_type']}")
print(f"Using pruning engine: {pruning_config['pruning_engine']}")
```
```python
# CNN example
payload = {
    "model_name": "resnet50",
    "task": "image-classification"
}

response = requests.post(
    "http://cloud-service:8000/api/prune",
    json=payload
)

pruning_config = response.json()
```

### Applying Pruning Configuration

#### For LLMs (SiRE output)
```python
from pruning_utils import apply_llm_pruning

# Load original model
model = AutoModel.from_pretrained("bert-base-uncased")

# Apply pruning configuration from SiRE
pruned_model = apply_llm_pruning(model, pruning_config)

# Save pruned model
pruned_model.save_pretrained("./pruned_bert")
```

#### For CNNs (ImproveNet output)
```python
from pruning_utils import apply_cnn_pruning
import torchvision.models as models

# Load original model
model = models.resnet50(pretrained=True)

# Apply pruning configuration from ImproveNet
pruned_model = apply_cnn_pruning(model, pruning_config)

# Save pruned model
torch.save(pruned_model.state_dict(), "./pruned_resnet50.pth")
```

## API Endpoints

### POST `/api/prune`

Analyzes a model, dispatches to appropriate engine, and generates the pruning configuration.

**Request Body:**
```json
{
  "model_name": "string",
  "task": "string",
  "pruning_ratio": 0.2,  // optional, default: 0.2
  "pruning_method": "magnitude"  // optional
}
```

**Response:**
```json
{
  "model_name": "string",
  "model_type": "llm|cnn",
  "pruning_engine": "SiRE|ImproveNet",
  "task": "string",
  "pruning_config": {...},
  "metadata": {...}
}
```

### GET `/api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "engines": {
    "SiRE": "active",
    "ImproveNet": "active"
  }
}
```

### GET `/api/supported-models`

List supported model types and architectures.

**Response:**
```json
{
  "llm": ["LLAMA", "MISTRAL", "QWEN"],
  "cnn": ["VGG16", "VGG19", "AlexNet", "Convolutional Autoencoder", "Linear AutoEncoder"]
}
```

## Supported Tasks

### LLM Tasks (SiRE)
- `question-answering`
- `text-generation`

### CNN Tasks (ImproveNet)
- `image-classification`
- `object-detection`
- `signal reconstruction`

## Pruning Engines

### SiRE (Structured Importance-based Reduction Engine)

Specialized for Large Language Models:

- **Attention Head Pruning**: Identifies redundant attention heads based on importance scores
- **Feed-Forward Neuron Pruning**: Removes less important neurons in FFN layers
- **Methods**: 
  - Magnitude-based
  - Gradient-based
  - Taylor expansion
  - Attention pattern analysis

### ImproveNet

Optimized for Convolutional Neural Networks:

- **Filter Pruning**: Removes entire convolutional filters
- **Channel Pruning**: Prunes input/output channels
- **Methods**:
  - L1/L2 norm-based
  - Geometric median
  - Feature map correlation
  - Network slimming

## Project Structure
```
cloud-edge-pruning-framework/
├── cloud_service/
│   ├── app.py
│   ├── dispatcher.py                 # Model type detection and routing
│   └── utils.py
├── edge_client/
│   ├── client.py
│   ├── pruning_utils.py
│   └── model_handler.py
├── examples/
│   ├── prune_llm.py
│   └── prune_cnn.py
├── requirements.txt
├── setup.py
└── README.md
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Hugging Face Transformers library
- PyTorch and torchvision frameworks
- SiRE pruning methodology for LLMs
- ImproveNet pruning methodology for CNNs
