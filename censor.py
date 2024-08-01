import cv2
import pytesseract
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray

def detect_text(image, keywords):
    config = r'--oem 3 --psm 6'  
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    
    boxes = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:
            text = data['text'][i].strip().lower()
            if any(keyword.lower() in text for keyword in keywords):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                boxes.append((x, y, w, h))
    
    return boxes

def merge_boxes(boxes, threshold=10):
    if not boxes:
        return []
    
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = [sorted_boxes[0]]
    
    for box in sorted_boxes[1:]:
        last_box = merged[-1]
        if (box[0] <= last_box[0] + last_box[2] + threshold and
            box[1] <= last_box[1] + last_box[3] + threshold):
            # Merge boxes
            new_x = min(last_box[0], box[0])
            new_y = min(last_box[1], box[1])
            new_w = max(last_box[0] + last_box[2], box[0] + box[2]) - new_x
            new_h = max(last_box[1] + last_box[3], box[1] + box[3]) - new_y
            merged[-1] = (new_x, new_y, new_w, new_h)
        else:
            merged.append(box)
    
    return merged

def censor_frame(frame, keywords, prev_boxes):
    frame_copy = np.array(frame, dtype=np.uint8)
    gray = preprocess_frame(frame_copy)
    
    boxes = detect_text(gray, keywords)
    boxes.extend(prev_boxes)
    boxes = merge_boxes(boxes)
    
    for (x, y, w, h) in boxes:
        x = max(0, x - 5)
        y = max(0, y - 5)
        w += 10
        h += 10
        frame_copy = cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 0), -1)
    
    return frame_copy, boxes

def process_frame_chunk(frames, keywords, initial_prev_boxes, chunk_index):
    processed_frames = []
    prev_boxes = initial_prev_boxes
    
    for frame in frames:
        censored_frame, prev_boxes = censor_frame(frame, keywords, prev_boxes)
        processed_frames.append(censored_frame)
    
    return processed_frames, prev_boxes, chunk_index

def censor_video(input_path, output_path, keywords, tesseract_path, display_frames, display_nth_frame):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    print("Loading video...")
    clip = VideoFileClip(input_path)
    
    total_frames = int(clip.fps * clip.duration)
    fps = clip.fps
    print(f"Total frames to process: {total_frames}")
    
    chunk_size = 100  
    censored_frames = [None] * total_frames  
    prev_boxes = []
    
    if display_frames:
        plt.figure(figsize=(12, 8))
        plt.ion()
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_chunk = {}
        frames_processed = tqdm(total=total_frames, desc="Processing frames")
        
        for i in range(0, total_frames, chunk_size):
            chunk_frames = list(clip.iter_frames(fps=fps))[i:i+chunk_size]
            future = executor.submit(process_frame_chunk, chunk_frames, keywords, prev_boxes, i)
            future_to_chunk[future] = i
        
        for future in as_completed(future_to_chunk):
            chunk_frames, prev_boxes, chunk_index = future.result()
            censored_frames[chunk_index:chunk_index+len(chunk_frames)] = chunk_frames
            
            frames_processed.update(len(chunk_frames))
            
            if display_frames:
                for j in range(chunk_index, chunk_index + len(chunk_frames), display_nth_frame):
                    if j < len(censored_frames) and censored_frames[j] is not None:
                        plt.clf()
                        plt.imshow(censored_frames[j])
                        plt.title(f"Frame {j}")
                        plt.axis('off')
                        plt.pause(0.01)
    
    frames_processed.close()
    
    if display_frames:
        plt.ioff()
        plt.close()
    
    censored_frames = [f for f in censored_frames if f is not None]
    
    print("Creating censored video clip...")
    censored_clip = concatenate_videoclips([ImageClip(f, duration=1/fps) for f in censored_frames], method="compose")
    
    print("Adding audio to censored video...")
    final_clip = censored_clip.set_audio(clip.audio)
    
    print("Writing final video...")
    final_clip.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac')
    
    print("Video processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Censoring Tool")
    parser.add_argument("input_path", help="Path to the input video file")
    parser.add_argument("output_path", help="Path for the output video file")
    parser.add_argument("keywords", nargs='+', help="Keywords to censor")
    parser.add_argument("--tesseract_path", default=r"C:\Program Files\Tesseract-OCR\tesseract.exe", help="Path to Tesseract executable")
    parser.add_argument("--display", action="store_true", help="Display frames while processing")
    parser.add_argument("--nth_frame", type=int, default=1, help="Display every nth frame (default: 1)")
    
    args = parser.parse_args()

    censor_video(args.input_path, args.output_path, args.keywords, args.tesseract_path, args.display, args.nth_frame)
