# Qwen Vision-Language Model Processing for Apple Silicon

This repository contains optimized scripts for running vision-language models on Apple Silicon (M1/M2) Macs, with a focus on speed and efficiency.

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install torch torchvision pillow accelerate transformers onnx onnxruntime
```

### Fastest Option (Recommended)
```bash
python qwen_ultra_fast.py
```
- **Processing Time:** ~0.8 seconds per image
- **Model:** microsoft/git-base (110M parameters)
- **Features:** Basic image captioning with logging

## 📁 Scripts Overview

### 1. `qwen_ml_silicon.py` - Original Script
- **Purpose:** Full Qwen2-VL-7B-Instruct model with interior/exterior classification
- **Processing Time:** 10-30 seconds per image
- **Memory Usage:** High (requires disk offloading)
- **Features:** Detailed image descriptions, interior/exterior classification

### 2. `qwen_ultra_fast.py` - Optimized Version ⭐
- **Purpose:** Ultra-fast image processing with logging
- **Processing Time:** ~0.8 seconds per image
- **Model:** microsoft/git-base (110M parameters)
- **Features:** 
  - Comprehensive logging
  - Interior/exterior classification
  - Optimized for MacBook Pro M1
  - Automatic model selection

### 3. `qwen_small_multimodal.py` - Multimodal Support
- **Purpose:** Supports both image and text inputs
- **Models:** Multiple small models with fallback options
- **Features:** Image captioning + text conversations

### 4. `convert_to_onnx.py` - ONNX Conversion (Complex)
- **Purpose:** Convert Qwen2-VL-7B-Instruct to ONNX format
- **Status:** ⚠️ May fail due to model complexity
- **Use Case:** Advanced users who need ONNX optimization

### 5. `convert_simple_onnx.py` - Simple ONNX Conversion
- **Purpose:** Convert smaller models to ONNX for speed
- **Model:** microsoft/git-base (110M parameters)
- **Features:** Creates `simple_onnx_inference.py` for ONNX inference

## 🎯 Performance Comparison

| Script | Model Size | Processing Time | Memory Usage | Features |
|--------|------------|-----------------|--------------|----------|
| `qwen_ml_silicon.py` | 7B params | 10-30 sec | High | Full multimodal |
| `qwen_ultra_fast.py` | 110M params | ~0.8 sec | Low | Fast + logging |
| `qwen_small_multimodal.py` | 110M-990M | 1-5 sec | Medium | Image + text |
| ONNX versions | 110M params | ~0.5 sec | Low | Ultra-fast |

## 🔧 Usage Instructions

### Basic Image Processing
```bash
# Enter image path when prompted
python qwen_ultra_fast.py
```

### ONNX Conversion (Optional)
```bash
# Convert to ONNX for maximum speed
python convert_simple_onnx.py

# Use the ONNX model
python simple_onnx_inference.py
```

## 📊 Logging

All scripts include comprehensive logging:
- **Log Files:** Timestamped files (e.g., `qwen_processing_20250128_163015.log`)
- **Console Output:** Real-time processing information
- **Performance Metrics:** Processing times, model loading, errors

### Log Information Includes:
- Model loading attempts and success/failure
- Image processing times
- Classification results (INTERIOR/EXTERIOR)
- Error messages with context
- Session start/end timestamps

## 🎨 Features

### Interior/Exterior Classification
- **Prompt:** "Classify this image as INTERIOR or EXTERIOR only. Then briefly describe what you see."
- **Output Format:** 
  ```
  📋 Classification: 🏠 INTERIOR
  📝 Description: This is an INTERIOR image showing...
  ```

### Image Processing Optimizations
- **Resizing:** Automatic resizing to optimal dimensions
- **Memory Management:** Efficient memory usage for Apple Silicon
- **Batch Processing:** Support for multiple images

## 🛠️ Technical Details

### Hardware Requirements
- **Recommended:** MacBook Pro M1/M2 with 16GB+ RAM
- **Minimum:** 8GB RAM (may use disk offloading)

### Model Options
1. **microsoft/git-base** (110M params) - Fastest
2. **microsoft/git-base-coco** (110M params) - Good quality
3. **Salesforce/blip-image-captioning-base** (990M params) - Better quality
4. **Qwen/Qwen2-VL-7B-Instruct** (7B params) - Best quality, slowest

### ONNX Runtime Benefits
- **2-5x faster inference** than PyTorch
- **Better memory management**
- **Optimized for Apple Silicon**
- **Reduced latency**

## 🔍 Troubleshooting

### Common Issues

1. **"Invalid buffer size" Error**
   - **Solution:** Use smaller models or increase available RAM
   - **Workaround:** Close other applications

2. **Slow Processing**
   - **Solution:** Use `qwen_ultra_fast.py` instead of original
   - **Alternative:** Convert to ONNX using `convert_simple_onnx.py`

3. **Model Loading Failures**
   - **Solution:** Check internet connection for model downloads
   - **Alternative:** Use offline models if available

### Performance Tips
- Close unnecessary applications to free RAM
- Use smaller image dimensions for faster processing
- Enable "Fast Mode" in scripts for maximum speed
- Consider ONNX conversion for production use

## 📝 Example Output

```
🚀 Loading model on mps device...
⚡ Ultra-fast mode enabled - maximum speed optimizations
✅ Successfully loaded microsoft/git-base

Enter the path to your image or 'q' to quit: /path/to/image.jpg

📋 Classification: 🏠 INTERIOR
📝 Description: This is an INTERIOR image showing a modern living room with comfortable furniture and warm lighting.

⚡ Ultra-fast processing time: 0.84 seconds
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is for educational and research purposes.

---

**💡 Pro Tip:** For the best balance of speed and quality, use `qwen_ultra_fast.py` with the microsoft/git-base model. It provides fast processing (~0.8 seconds) while maintaining good image understanding capabilities. 