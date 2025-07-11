import torch
import onnx
import onnxruntime as ort
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import numpy as np
import time
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration."""
    log_filename = f"onnx_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"🔄 Starting ONNX Conversion for Qwen2-VL-7B-Instruct")
    return log_filename

def load_model_and_tools():
    """Load the Qwen2-VL model for ONNX conversion."""
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    
    logging.info(f"📦 Loading model: {model_name}")
    print(f"📦 Loading model: {model_name}")
    
    try:
        # Load model with optimizations for conversion
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        logging.info("✅ Model loaded successfully")
        print("✅ Model loaded successfully")
        return model, tokenizer, processor
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        print(f"❌ Failed to load model: {e}")
        raise

def create_dummy_inputs(processor, tokenizer):
    """Create dummy inputs for ONNX conversion."""
    logging.info("🔄 Creating dummy inputs for ONNX conversion")
    
    # Create dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Create dummy text
    dummy_text = "Describe this image briefly."
    
    # Create messages format for Qwen2-VL
    messages = [{"role": "user", "content": [{"type": "image", "image": dummy_image}, {"type": "text", "text": dummy_text}]}]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(text=[text], images=[dummy_image], padding=True, return_tensors="pt")
    
    logging.info(f"📏 Input shapes - input_ids: {inputs['input_ids'].shape}, attention_mask: {inputs['attention_mask'].shape}, pixel_values: {inputs['pixel_values'].shape}")
    
    return inputs

def convert_to_onnx(model, tokenizer, processor, onnx_path="qwen2vl_model.onnx"):
    """Convert the PyTorch model to ONNX format."""
    logging.info("🔄 Starting ONNX conversion...")
    print("🔄 Starting ONNX conversion...")
    
    try:
        # Create dummy inputs
        dummy_inputs = create_dummy_inputs(processor, tokenizer)
        
        # Set model to evaluation mode
        model.eval()
        
        # Define dynamic axes for variable input sizes
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'pixel_values': {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
        
        # Export to ONNX
        logging.info("📤 Exporting model to ONNX format...")
        torch.onnx.export(
            model,
            (dummy_inputs['input_ids'], dummy_inputs['attention_mask'], dummy_inputs['pixel_values']),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask', 'pixel_values'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        logging.info(f"✅ Model successfully converted and saved to: {onnx_path}")
        print(f"✅ Model successfully converted and saved to: {onnx_path}")
        
        # Validate ONNX model
        logging.info("🔍 Validating ONNX model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logging.info("✅ ONNX model validation passed")
        print("✅ ONNX model validation passed")
        
        return onnx_path
        
    except Exception as e:
        logging.error(f"❌ ONNX conversion failed: {e}")
        print(f"❌ ONNX conversion failed: {e}")
        raise

def test_onnx_model(onnx_path, processor, tokenizer):
    """Test the converted ONNX model."""
    logging.info("🧪 Testing ONNX model...")
    print("🧪 Testing ONNX model...")
    
    try:
        # Create ONNX Runtime session
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Create test inputs
        test_image = Image.new('RGB', (224, 224), color='blue')
        test_text = "What's in this image?"
        
        messages = [{"role": "user", "content": [{"type": "image", "image": test_image}, {"type": "text", "text": test_text}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[test_image], padding=True, return_tensors="pt")
        
        # Convert to numpy for ONNX Runtime
        onnx_inputs = {
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy(),
            'pixel_values': inputs['pixel_values'].numpy()
        }
        
        # Run inference
        start_time = time.time()
        outputs = session.run(None, onnx_inputs)
        inference_time = time.time() - start_time
        
        logging.info(f"✅ ONNX model test successful - Inference time: {inference_time:.2f} seconds")
        print(f"✅ ONNX model test successful - Inference time: {inference_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ ONNX model test failed: {e}")
        print(f"❌ ONNX model test failed: {e}")
        return False

def create_onnx_inference_script(onnx_path):
    """Create a script for ONNX inference."""
    script_content = f'''import onnxruntime as ort
import torch
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import numpy as np
import time

def load_onnx_model():
    """Load the ONNX model for inference."""
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession("{onnx_path}", providers=providers)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return session, tokenizer, processor

def process_image_onnx(image_path, session, tokenizer, processor):
    """Process image using ONNX model."""
    try:
        start_time = time.time()
        
        # Load and resize image
        img = Image.open(image_path).resize((224, 224))
        
        # Create prompt
        prompt = "Classify this image as INTERIOR or EXTERIOR only. Then briefly describe what you see."
        messages = [{{"role": "user", "content": [{{"type": "image", "image": img}}, {{"type": "text", "text": prompt}}]}}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
        
        # Convert to numpy
        onnx_inputs = {{
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy(),
            'pixel_values': inputs['pixel_values'].numpy()
        }}
        
        # Run inference
        outputs = session.run(None, onnx_inputs)
        
        # Decode output
        logits = outputs[0]
        predicted_ids = np.argmax(logits, axis=-1)
        response = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        
        processing_time = time.time() - start_time
        
        print(f"📷 Result: {{response}}")
        print(f"⚡ ONNX Processing time: {{processing_time:.2f}} seconds")
        
    except Exception as e:
        print(f"Error: {{e}}")

if __name__ == "__main__":
    print("🚀 Loading ONNX model...")
    session, tokenizer, processor = load_onnx_model()
    print("✅ ONNX model loaded successfully!")
    
    while True:
        image_path = input("\\nEnter image path or 'q' to quit: ").strip()
        if image_path.lower() == 'q':
            break
        process_image_onnx(image_path, session, tokenizer, processor)
'''
    
    with open("onnx_inference.py", "w") as f:
        f.write(script_content)
    
    logging.info("📝 Created onnx_inference.py for ONNX model usage")
    print("📝 Created onnx_inference.py for ONNX model usage")

def main():
    """Main function to convert model to ONNX."""
    log_filename = setup_logging()
    
    try:
        # Load model
        model, tokenizer, processor = load_model_and_tools()
        
        # Convert to ONNX
        onnx_path = convert_to_onnx(model, tokenizer, processor)
        
        # Test ONNX model
        if test_onnx_model(onnx_path, processor, tokenizer):
            # Create inference script
            create_onnx_inference_script(onnx_path)
            
            logging.info("🎉 ONNX conversion completed successfully!")
            print("\n🎉 ONNX conversion completed successfully!")
            print(f"📁 ONNX model saved as: {onnx_path}")
            print("📝 Inference script created as: onnx_inference.py")
            print("\n🚀 To use the ONNX model, run: python onnx_inference.py")
        else:
            logging.error("❌ ONNX conversion failed during testing")
            print("❌ ONNX conversion failed during testing")
            
    except Exception as e:
        logging.error(f"❌ Conversion process failed: {e}")
        print(f"❌ Conversion process failed: {e}")

if __name__ == "__main__":
    main() 