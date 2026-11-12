import os
import json
import time
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def describe_all_frames(frames_dir, output_json_path):
    client = genai.Client()

    # Get and sort all frames to process in chronological order
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    print(f"Found {len(frame_files)} frames in '{frames_dir}' to describe.")

    descriptions = {}
    
    # Check if we have progress saved previously, to resume if interrupted
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r') as f:
                descriptions = json.load(f)
            print(f"Resuming: Loaded {len(descriptions)} existing descriptions.")
        except Exception:
            print("Could not read existing descriptions. Starting fresh.")

    # Highly specific system prompt for spatial awareness
    prompt = (
        "Describe every physical object in this image, its color, and its spatial relationship "
        "to other objects (e.g., 'A red coffee cup is sitting on a wooden desk to the left of a "
        "black laptop'). Be highly detailed but concise."
    )

    for i, filename in enumerate(frame_files):
        # Skip if already processed (helpful if the script crashes or runs out of quota)
        if filename in descriptions:
            continue
            
        filepath = os.path.join(frames_dir, filename)
        
        # Extract timestamp from filename 'frame_MM_SS.jpg'
        # e.g., 'frame_00_05.jpg' -> '00:05'
        parts = filename.split('_')
        timestamp = f"{parts[1]}:{parts[2].replace('.jpg', '')}"

        print(f"[{i+1}/{len(frame_files)}] Describing {filename} (Timestamp: {timestamp}):")
        
        try:
            # Load image using PIL
            img = Image.open(filepath)
            
            # Call Gemini Flash
            response = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=[img, prompt]
            )
            
            # Save mapping: Filename -> Data
            descriptions[filename] = {
                "image_path": filepath,
                "timestamp": timestamp,
                "description": response.text.strip()
            }
            
            # Save progress iteratively
            with open(output_json_path, 'w') as f:
                json.dump(descriptions, f, indent=4)
                
            # Sleep to prevent hitting the 15 RPM rate limit (1 request per 4.5 seconds)
            time.sleep(4.5)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            print("Pausing for 10 seconds before retrying...")
            time.sleep(10)

    print(f"Descriptions saved to {output_json_path}")

if __name__ == "__main__":
    FRAMES_DIR = "extracted_frames"
    OUTPUT_JSON = "descriptions.json"
    
    describe_all_frames(FRAMES_DIR, OUTPUT_JSON)
