# Visual Attention Networks

A CNN with Channel and Spatial Attention mechanisms for image classification
on CIFAR-10, with visualizations of where the model focuses.

## Attention Mechanisms
- **Channel Attention (SE Block)**: which feature maps matter most?
- **Spatial Attention**: which spatial locations matter most?
- Combined at each layer to progressively refine focus

## Architecture
ResNet-like CNN (3 conv blocks with stride) enhanced with attention blocks
after each conv stage.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py
python src/visualize_attention.py
```

## Results
- `results/training_curves.png` — loss and accuracy
- `results/attention_visualization.png` — spatial attention heatmaps per layer

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology
