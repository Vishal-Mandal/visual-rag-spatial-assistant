import os
import cv2

def extract_frames(video_path, output_dir):
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 3. Get video metadata (FPS and Total Frames)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Avoid division by zero if metadata is missing
    if fps == 0 or total_frames == 0:
        print("Error: Could not read video metadata (FPS or total frames is 0).")
        return
        
    duration_seconds = total_frames / fps
    
    print(f"Video Loaded: {video_path}")
    print(f"FPS: {fps:.2f} | Total Frames: {total_frames} | Duration: {duration_seconds:.2f} seconds")

    # 4. Extract 1 frame per second
    for second in range(int(duration_seconds)):
        # Calculate time in milliseconds
        time_ms = second * 1000
        
        # Seek the video reader to this exact millisecond
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        
        # Read the frame at this position
        success, frame = cap.read()
        
        if success:
            # Format the filename as frame_MM_SS.jpg (e.g. frame_00_05.jpg)
            minutes = second // 60
            seconds = second % 60
            filename = f"frame_{minutes:02d}_{seconds:02d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save the frame as a JPEG image
            cv2.imwrite(filepath, frame)
            print(f"Extracted: {filename}")
        else:
            print(f"Failed to extract frame at second {second}")

    # 5. Clean up
    cap.release()
    print("Frame extraction complete!")

if __name__ == "__main__":
    # Settings
    VIDEO_FILE = "room.mp4" 
    OUTPUT_FOLDER = "extracted_frames"
    
    extract_frames(VIDEO_FILE, OUTPUT_FOLDER)
