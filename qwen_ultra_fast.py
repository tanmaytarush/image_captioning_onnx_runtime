import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from PIL import Image
from pathlib import Path
import sys
import time
import logging
from datetime import datetime

OUTPUT_FULL_RESPONSE = False
FAST_MODE = True

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def setup_logging():
    """Setup logging configuration."""
    log_filename = f"qwen_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"🚀 Starting Qwen Ultra Fast Image Processing")
    logging.info(f"💻 Device: {device.type}")
    logging.info(f"📁 Log file: {log_filename}")
    return log_filename

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

def load_model_and_tools():
    """Load a fast vision-language model for quick image description."""
   
    model_options = [
        "Qwen/Qwen2-VL-7B-Instruct",   # Best for specific prompts
        "microsoft/git-base",           
        "microsoft/git-base-coco",    
        "nlpconnect/vit-gpt2-image-captioning"  
    ]
    
    for model_name in model_options:
        logging.info(f"🔄 Attempting to load model: {model_name}")
        print(f"🔄 Trying to load {model_name}...")
        try:
            if "Qwen" in model_name:
                
                from transformers import Qwen2VLForConditionalGeneration
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16, 
                    device_map="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                processor = AutoProcessor.from_pretrained(model_name)
            else:
             
                model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                processor = AutoProcessor.from_pretrained(model_name)
            
            logging.info(f"✅ Successfully loaded model: {model_name}")
            print(f"✅ Successfully loaded {model_name}")
            return model, tokenizer, processor, model_name
        except Exception as e:
            logging.error(f"❌ Failed to load model {model_name}: {e}")
            print(f"❌ Failed to load {model_name}: {e}")
            continue
    
    logging.error("❌ Failed to load any model")
    print("❌ Failed to load any model")
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

def process_image_qwen(image_path: Path, model, tokenizer, processor):
    """Process image using Qwen2-VL model."""
    try:
        start_time = time.time()
        logging.info(f"🖼️  Processing image: {image_path}")
        logging.info(f"📏 Image dimensions: {Image.open(image_path).size}")
        
        with Image.open(image_path) as img:
            max_height = 512 if FAST_MODE else 800
            max_width = 512 if FAST_MODE else 800
            img = resize_image(img, max_height, max_width)
            
            # Custom prompt for interior/exterior classification
            prompt = "Classify this image as INTERIOR or EXTERIOR only. Then briefly describe the main elements." if FAST_MODE else "Classify this image as INTERIOR or EXTERIOR only. Then provide a detailed description of what you see."
            
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
            
            if FAST_MODE:
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=50,
                    temperature=0.9,
                    repetition_penalty=1.0,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    num_beams=1
                )
            else:
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100,
                    temperature=0.7,
                    repetition_penalty=1.02,
                    do_sample=True,
                    top_p=0.9
                )
            
            response_ids = outputs[0]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
            
            if OUTPUT_FULL_RESPONSE:
                print(f"\n{response_text}")
            else:
                response = response_text.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0].strip()
                
                            # Extract classification and format output
            if "INTERIOR" in response.upper():
                classification = "🏠 INTERIOR"
            elif "EXTERIOR" in response.upper():
                classification = "🏢 EXTERIOR"
            else:
                classification = "❓ UNCLEAR"
            
            processing_time = end_time - start_time
            
            # Log results
            logging.info(f"📋 Classification: {classification}")
            logging.info(f"📝 Description: {response}")
            logging.info(f"⏱️  Processing time: {processing_time:.2f} seconds")
            
            print(f"\n📋 Classification: {classification}")
            print(f"📝 Description: {response}")
            print(f"\n⏱️  Processing time: {processing_time:.2f} seconds")
            
    except Exception as e:
        logging.error(f"❌ Error processing image {image_path}: {e}")
        print(f"Error processing image: {e}")

def process_image_standard(image_path: Path, model, tokenizer, processor):
    """Process image using standard vision-language models."""
    try:
        start_time = time.time()
        logging.info(f"🖼️  Processing image with standard model: {image_path}")
        
        with Image.open(image_path) as img:
            max_height = 224 if FAST_MODE else 384 
            max_width = 224 if FAST_MODE else 384
            img = resize_image(img, max_height, max_width)
            
            # Process image for standard models
            inputs = processor(images=img, return_tensors="pt").to(device)
            
            # Generate caption
            if FAST_MODE:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,  
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    num_beams=1
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode the generated text
            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            processing_time = end_time - start_time
            
            # Log results
            logging.info(f"📷 Basic Caption: {caption}")
            logging.info(f"⏱️  Processing time: {processing_time:.2f} seconds")
            
            print(f"\n📷 Basic Caption: {caption}")
            print("💡 Note: This model provides basic captioning. For precise interior/exterior classification, the Qwen model will be used.")
            print(f"\n⏱️  Processing time: {processing_time:.2f} seconds")
            
    except Exception as e:
        logging.error(f"❌ Error processing image with standard model {image_path}: {e}")
        print(f"Error processing image: {e}")

def main():
    """Main function to handle model loading and image processing loop."""
    # Setup logging
    log_filename = setup_logging()
    
    print(f"Loading model on {device.type} device...")
    if FAST_MODE:
        print("🚀 Fast mode enabled - using ultra-optimized settings for maximum speed")
        logging.info("🚀 Fast mode enabled")
    
    model, tokenizer, processor, model_name = load_model_and_tools()

    while True:
        image_path = get_image_path()
        if image_path is None:
            logging.info("👋 User requested exit")
            print("Exiting...")
            break
        
        # Use appropriate processing function based on model type
        if "Qwen" in model_name:
            process_image_qwen(image_path, model, tokenizer, processor)
        else:
            process_image_standard(image_path, model, tokenizer, processor)

    logging.info("🏁 Session ended")
    print("Goodbye!")

if __name__ == "__main__":
    main() 