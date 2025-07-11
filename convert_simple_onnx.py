import torch
import onnx
import onnxruntime as ort
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from PIL import Image
import numpy as np
import time
import logging
from datetime import datetime
import os

def setup_logging():
    """Setup logging configuration."""
    log_filename = f"simple_onnx_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"🔄 Starting Simple ONNX Conversion")
    return log_filename

def load_simple_model():
    """Load a simple model that's easier to convert to ONNX."""
    model_name = "microsoft/git-base"  # 110M params - much easier to convert
    
    logging.info(f"📦 Loading simple model: {model_name}")
    print(f"📦 Loading simple model: {model_name}")
    
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        logging.info("✅ Simple model loaded successfully")
        print("✅ Simple model loaded successfully")
        return model, tokenizer, processor
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        print(f"❌ Failed to load model: {e}")
        raise

def convert_simple_to_onnx(model, tokenizer, processor, onnx_path="simple_model.onnx"):
    """Convert the simple model to ONNX format."""
    logging.info("🔄 Starting simple ONNX conversion...")
    print("🔄 Starting simple ONNX conversion...")
    
    try:
        # Create dummy inputs
        dummy_image = Image.new('RGB', (224, 224), color='red')
        inputs = processor(images=dummy_image, return_tensors="pt")
        
        # Set model to evaluation mode
        model.eval()
        
        # Define dynamic axes
        dynamic_axes = {
            'pixel_values': {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
        
        # Export to ONNX
        logging.info("📤 Exporting simple model to ONNX format...")
        torch.onnx.export(
            model,
            inputs['pixel_values'],
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        logging.info(f"✅ Simple model successfully converted and saved to: {onnx_path}")
        print(f"✅ Simple model successfully converted and saved to: {onnx_path}")
        
        # Validate ONNX model
        logging.info("🔍 Validating ONNX model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logging.info("✅ ONNX model validation passed")
        print("✅ ONNX model validation passed")
        
        return onnx_path
        
    except Exception as e:
        logging.error(f"❌ Simple ONNX conversion failed: {e}")
        print(f"❌ Simple ONNX conversion failed: {e}")
        raise

def test_simple_onnx_model(onnx_path, processor, tokenizer):
    """Test the converted simple ONNX model."""
    logging.info("🧪 Testing simple ONNX model...")
    print("🧪 Testing simple ONNX model...")
    
    try:
        # Create ONNX Runtime session
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Create test inputs
        test_image = Image.new('RGB', (224, 224), color='blue')
        inputs = processor(images=test_image, return_tensors="pt")
        
        # Convert to numpy for ONNX Runtime
        onnx_inputs = {
            'pixel_values': inputs['pixel_values'].numpy()
        }
        
        # Run inference
        start_time = time.time()
        outputs = session.run(None, onnx_inputs)
        inference_time = time.time() - start_time
        
        logging.info(f"✅ Simple ONNX model test successful - Inference time: {inference_time:.2f} seconds")
        print(f"✅ Simple ONNX model test successful - Inference time: {inference_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Simple ONNX model test failed: {e}")
        print(f"❌ Simple ONNX model test failed: {e}")
        return False

def create_simple_onnx_inference_script(onnx_path):
    """Create a script for simple ONNX inference."""
    script_content = f'''import onnxruntime as ort
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import numpy as np
import time

def load_simple_onnx_model():
    """Load the simple ONNX model for inference."""
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession("{onnx_path}", providers=providers)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/git-base")
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    return session, tokenizer, processor

def process_image_simple_onnx(image_path, session, tokenizer, processor):
    """Process image using simple ONNX model."""
    try:
        start_time = time.time()
        
        # Load and resize image
        img = Image.open(image_path).resize((224, 224))
        
        # Process image
        inputs = processor(images=img, return_tensors="pt")
        
        # Convert to numpy
        onnx_inputs = {{
            'pixel_values': inputs['pixel_values'].numpy()
        }}
        
        # Run inference
        outputs = session.run(None, onnx_inputs)
        
        # Decode output
        logits = outputs[0]
        predicted_ids = np.argmax(logits, axis=-1)
        caption = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        
        processing_time = time.time() - start_time
        
        print(f"📷 Caption: {{caption}}")
        print(f"⚡ ONNX Processing time: {{processing_time:.2f}} seconds")
        
    except Exception as e:
        print(f"Error: {{e}}")

if __name__ == "__main__":
    print("🚀 Loading simple ONNX model...")
    session, tokenizer, processor = load_simple_onnx_model()
    print("✅ Simple ONNX model loaded successfully!")
    
    while True:
        image_path = input("\\nEnter image path or 'q' to quit: ").strip()
        if image_path.lower() == 'q':
            break
        process_image_simple_onnx(image_path, session, tokenizer, processor)
'''
    
    with open("simple_onnx_inference.py", "w") as f:
        f.write(script_content)
    
    logging.info("📝 Created simple_onnx_inference.py for ONNX model usage")
    print("📝 Created simple_onnx_inference.py for ONNX model usage")

def main():
    """Main function to convert simple model to ONNX."""
    log_filename = setup_logging()
    
    try:
        # Load simple model
        model, tokenizer, processor = load_simple_model()
        
        # Convert to ONNX
        onnx_path = convert_simple_to_onnx(model, tokenizer, processor)
        
        # Test ONNX model
        if test_simple_onnx_model(onnx_path, processor, tokenizer):
            # Create inference script
            create_simple_onnx_inference_script(onnx_path)
            
            logging.info("🎉 Simple ONNX conversion completed successfully!")
            print("\n🎉 Simple ONNX conversion completed successfully!")
            print(f"📁 ONNX model saved as: {onnx_path}")
            print("📝 Inference script created as: simple_onnx_inference.py")
            print("\n🚀 To use the ONNX model, run: python simple_onnx_inference.py")
            print("💡 This model provides basic image captioning with ultra-fast ONNX inference!")
        else:
            logging.error("❌ Simple ONNX conversion failed during testing")
            print("❌ Simple ONNX conversion failed during testing")
            
    except Exception as e:
        logging.error(f"❌ Simple conversion process failed: {e}")
        print(f"❌ Simple conversion process failed: {e}")

if __name__ == "__main__":
    main() 