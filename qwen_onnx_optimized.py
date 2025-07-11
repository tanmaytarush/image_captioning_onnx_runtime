import torch
import onnxruntime as ort
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from pathlib import Path
import sys
import time
import numpy as np
import os

OUTPUT_FULL_RESPONSE = False
FAST_MODE = True
USE_ONNX = True  # Enable ONNX Runtime

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_image_path():
    """Prompt user for an image path until a valid path is provided or 'q' to quit."""
    while True:
        path_input = input("\nEnter the path to your image or 'q' to quit: ").strip().strip("'\"")
        if path_input.lower() == 'q':
            return None
        path = Path(path_input)
        if path.exists():
            return path
        print("The file does not exist. Please try again.")

def convert_to_onnx(model, tokenizer, processor, onnx_path="qwen2vl_model.onnx"):
    """Convert the PyTorch model to ONNX format."""
    print("🔄 Converting model to ONNX format...")
    
    # Create dummy inputs for ONNX conversion
    dummy_text = "Describe this image briefly: What do you see?"
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    messages = [{"role": "user", "content": [{"type": "image", "image": dummy_image}, {"type": "text", "text": dummy_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[dummy_image], padding=True, return_tensors="pt")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values']),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask', 'pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'pixel_values': {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print(f"✅ Model converted and saved to {onnx_path}")
    return onnx_path

def load_model_and_tools():
    """Load the model using ONNX Runtime if available, otherwise fallback to PyTorch."""
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    onnx_path = "qwen2vl_model.onnx"
    
    if USE_ONNX and os.path.exists(onnx_path):
        print("🚀 Loading ONNX model for faster inference...")
        try:
            # Configure ONNX Runtime for Apple Silicon
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            return session, tokenizer, processor, True  # True indicates ONNX mode
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            print("Falling back to PyTorch...")
    
    # Fallback to PyTorch
    print("📦 Loading PyTorch model...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Convert to ONNX if not already done
        if USE_ONNX and not os.path.exists(onnx_path):
            convert_to_onnx(model, tokenizer, processor, onnx_path)
        
        return model, tokenizer, processor, False  # False indicates PyTorch mode
    except Exception as e:
        print(f"Failed to load the model: {e}")
        sys.exit(1)

def resize_image(image, max_height, max_width):
    """Resize the image only if it exceeds the specified dimensions."""
    original_width, original_height = image.size
    
    if original_width > max_width or original_height > max_height:
        aspect_ratio = original_width / original_height
        if original_width > original_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    else:
        return image

def process_image_onnx(image_path: Path, session, tokenizer, processor):
    """Process image using ONNX Runtime for faster inference."""
    try:
        start_time = time.time()
        
        with Image.open(image_path) as img:
            # Use smaller dimensions for faster processing
            max_height = 600 if FAST_MODE else 800
            max_width = 600 if FAST_MODE else 800
            img = resize_image(img, max_height, max_width)
            
            # Shorter prompt for faster generation
            prompt = "Describe this image briefly: What do you see?" if FAST_MODE else "Describe this image in detail."
            
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
            
            # Convert to numpy for ONNX Runtime
            onnx_inputs = {
                'input_ids': inputs['input_ids'].numpy(),
                'attention_mask': inputs['attention_mask'].numpy(),
                'pixel_values': inputs['pixel_values'].numpy()
            }
            
            # Run inference with ONNX Runtime
            outputs = session.run(None, onnx_inputs)
            
            # Decode the outputs
            logits = outputs[0]
            predicted_ids = np.argmax(logits, axis=-1)
            
            # Decode tokens to text
            response = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
            # Extract the assistant's response
            if '<|im_start|>assistant' in response:
                response = response.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0].strip()
            
            print(f"\n{response}")
            
            end_time = time.time()
            print(f"\n⏱️  ONNX Processing time: {end_time - start_time:.2f} seconds")
            
    except Exception as e:
        print(f"Error processing image: {e}")

def process_image_pytorch(image_path: Path, model, tokenizer, processor):
    """Process image using PyTorch (fallback method)."""
    try:
        start_time = time.time()
        
        with Image.open(image_path) as img:
            max_height = 800 if FAST_MODE else 1260
            max_width = 800 if FAST_MODE else 1260
            img = resize_image(img, max_height, max_width)
            
            prompt = "Describe this image briefly: What do you see?" if FAST_MODE else "Describe this image in detail."
            
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
            
            # Optimized generation parameters
            if FAST_MODE:
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100,  # Further reduced
                    temperature=0.8,
                    repetition_penalty=1.02,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=200,
                    temperature=0.7,
                    repetition_penalty=1.05
                )
            
            response_ids = outputs[0]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
            
            if OUTPUT_FULL_RESPONSE:
                print(f"\n{response_text}")
            else:
                response = response_text.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0].strip()
                print(f"\n{response}")
            
            end_time = time.time()
            print(f"\n⏱️  PyTorch Processing time: {end_time - start_time:.2f} seconds")
            
    except Exception as e:
        print(f"Error processing image: {e}")

def main():
    """Main function to handle model loading and image processing loop."""
    print(f"Loading model on {device.type} device...")
    if FAST_MODE:
        print("🚀 Fast mode enabled - using optimized settings for speed")
    if USE_ONNX:
        print("⚡ ONNX Runtime mode enabled for faster inference")
    
    model_or_session, tokenizer, processor, is_onnx = load_model_and_tools()

    while True:
        image_path = get_image_path()
        if image_path is None:
            print("Exiting...")
            break
        
        if is_onnx:
            process_image_onnx(image_path, model_or_session, tokenizer, processor)
        else:
            process_image_pytorch(image_path, model_or_session, tokenizer, processor)

    print("Goodbye!")

if __name__ == "__main__":
    main() 