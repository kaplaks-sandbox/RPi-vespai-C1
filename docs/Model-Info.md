

## 🔬 Model Information

VespAI uses a specialized YOLOv26 model trained specifically for hornet detection:

- **Model**: `yolov26` (14MB)
- **Classes**: 
  - **0**: Unknown insects
  - **1**: Bees
  - **2**: Vespa crabro (European hornet)
  - **3**: Vespa velutina (Asian hornet - invasive)
- **Research**: Based on Communications Biology 2024 paper
- **Accuracy**: Optimized for hornet species differentiation

For better performance, I recommend to use the ncnn format on RPi

### Model Performance
- **Input size**: 640x640 pixels
- **Parameters**: ~7M parameters
- **Speed**: ~15-30 FPS (depending on hardware)
- **Accuracy**: >95% on hornet detection task