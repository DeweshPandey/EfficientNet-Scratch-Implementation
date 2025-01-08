# EfficientNet from Scratch using PyTorch

This repository contains a scratch implementation of EfficientNet using PyTorch, inspired by the original research paper:
**"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"** by Mingxing Tan and Quoc V. Le.

EfficientNet introduces a new scaling method called **Compound Scaling** to systematically scale up the model's width, depth, and resolution, achieving state-of-the-art accuracy with fewer parameters and FLOPS.

## Key Concepts

### Compound Scaling
EfficientNet scales a baseline network using:
- **Depth (α^phi)**: Number of layers.
- **Width (β^phi)**: Number of channels per layer.
- **Resolution (γ^phi)**: Input image resolution.

Constraints:
- α × (β^2) × (γ^2) ≈ 2
- α, β, γ ≥ 1

### Key Techniques
EfficientNet incorporates:
1. **Mobile Inverted Residual Blocks**
2. **Squeeze-and-Excitation Optimization**
3. **Stochastic Depth**
4. **Depthwise Separable Convolutions**

These techniques improve efficiency and accuracy.

## Implementation Details

### Model Architecture
The baseline architecture is scaled using predefined `phi_values` for different versions (b0 to b7). Each version adjusts the model's depth, width, and resolution accordingly.

#### Baseline Model
```python
base_model = [
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]
```

### Classes and Components
#### 1. `CNNBlock`
Implements a convolutional block with:
- Convolution layer (supports depthwise separable convolution).
- Batch Normalization.
- SiLU activation.

#### 2. `SqueezeExcitation`
Applies channel-wise attention using:
- Adaptive Average Pooling.
- Two convolution layers for reduction and expansion.

#### 3. `InvertedResidualBlock`
A building block that:
- Expands input channels.
- Applies depthwise separable convolution.
- Reduces channels using pointwise convolution.
- Incorporates residual connections and stochastic depth.

#### 4. `EfficientNet`
The main model class that:
- Scales the baseline model using width, depth, and resolution factors.
- Constructs the feature extractor and classifier.

### `phi_values`
Defines scaling factors for each version:
```python
phi_values = {
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}
```

### Testing the Model
The test function validates the implementation by:
- Initializing an EfficientNet-b0 model.
- Forwarding a random input tensor of shape `(batch_size, 3, 224, 224)`.
- Displaying the output shape and model summary.

```python
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "b0"
    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes = 4, 10
    x = torch.rand((num_examples, 3, res, res)).to(device)
    model = EfficientNet(version=version, num_classes=num_classes).to(device)

    print(model(x).shape)  # (num_examples, num_classes)
    print(summary(model=model, input_size=(3, 224, 224), device="cuda"))

test()
```

## Dependencies
- PyTorch
- TorchSummary

Install required libraries using:
```bash
pip install torch torchvision torchsummary
```

## Running the Code
1. Clone the repository.
2. Install dependencies.
3. Run the script to test the EfficientNet implementation:
   ```bash
   python EfficientNet.py
   ```

## Output
- Prints the model summary and output tensor shape for a given input.

## References
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Future Work
- Add support for pre-trained weights.
- Implement training and evaluation scripts for real-world datasets.
- Benchmark model performance on various tasks.

