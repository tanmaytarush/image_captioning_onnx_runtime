import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from pathlib import Path
import sys
import time

OUTPUT_FULL_RESPONSE = False
FAST_MODE = True  # Enable faster inference mode

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

def load_model_and_tools():
    """Load the Qwen2-VL model, tokenizer, and processor for Apple Silicon."""
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    try:
        # For Apple Silicon, use device_map="auto" without additional .to(device)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, tokenizer, processor
    except Exception as e:
        print(f"Failed to load the model: {e}")
        print("Trying alternative loading method...")
        try:
            # Fallback: load without device_map and manually move to device
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float16
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            return model, tokenizer, processor
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
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


def process_image(image_path: Path, model, tokenizer, processor):
    """Process the image and generate a description using the MPS device if available."""
    try:
        start_time = time.time()
        
        with Image.open(image_path) as img:
            # Resize the image if necessary - use smaller dimensions for faster processing
            if FAST_MODE:
                max_height = 800  # Reduced from 1260
                max_width = 800   # Reduced from 1260
            else:
                max_height = 1260
                max_width = 1260
                
            img = resize_image(img, max_height, max_width)
            
            # Shorter prompt for faster generation
            if FAST_MODE:
                prompt = "Describe this image briefly: What do you see?"
            else:
                prompt = "Describe this image in detail as prose: Identify the type of image (photo, diagram, etc.). Describe each person if any, using specific terms like 'man', 'woman', 'boy', 'girl' etc. and include details about their attire and actions. Guess the location. Include sensory details and emotions. Don't be gender neutral. Avoid saying 'an individual', instead be specific about any people."
            
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
            
            # Optimized generation parameters for speed
            if FAST_MODE:
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=150,  # Reduced from 400
                    temperature=0.7,     # Slightly higher for faster generation
                    repetition_penalty=1.05,  # Reduced from 1.1
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=400, 
                    temperature=0.5, 
                    repetition_penalty=1.1
                )
            
            # Decode the outputs using tokenizer
            response_ids = outputs[0]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
            
            if OUTPUT_FULL_RESPONSE:
                print(f"\n{response_text}")
            else:
                response = response_text.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0].strip()
                print(f"\n{response}")
            
            end_time = time.time()
            print(f"\n⏱️  Processing time: {end_time - start_time:.2f} seconds")
            
    except FileNotFoundError:
        print(f"Error: The image file '{image_path}' was not found.")
    except Image.UnidentifiedImageError:
        print(f"Error: The file '{image_path}' is not a valid image file or is corrupted.")
    except torch.cuda.OutOfMemoryError:
        print("Error: Ran out of GPU memory. Try using a smaller image or freeing up GPU resources.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{image_path.name}': {e}")
        print("Please check your input and try again.")

def main():
    """Main function to handle model loading and image processing loop."""
    print(f"Loading model on {device.type} device...")
    if FAST_MODE:
        print("🚀 Fast mode enabled - using optimized settings for speed")
    model, tokenizer, processor = load_model_and_tools()

    while True:
        image_path = get_image_path()
        if image_path is None:
            print("Exiting...")
            break
        process_image(image_path, model, tokenizer, processor)

    print("Goodbye!")

if __name__ == "__main__":
    main()