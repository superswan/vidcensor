# VidCensor

This tool allows you to censor specific keywords in a video by using OCR to detect and cover text in each frame.

## Requirements

- Python 3.x
- OpenCV
- pytesseract
- moviepy
- tqdm
- matplotlib

## Installation

1. Install requirements 
    ```bash
    pip install opencv-python pytesseract moviepy tqdm matplotlib
    ```

2. Download and install Tesseract OCR from [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).

## Usage

```bash
python censor_video.py <input_path> <output_path> <keywords> [--tesseract_path <path_to_tesseract>] [--display] [--nth_frame <n>]
```
- <input_path>: Path to the input video file.
- <output_path>: Path for the output video file.
- <keywords>: Keywords to censor (separated by spaces).
- --tesseract_path: Optional. Path to the Tesseract executable. Default is C:\Program Files\Tesseract-OCR\tesseract.exe.
- --display: Optional. Display frames while processing.
- --nth_frame <N>: Optional. Display every nth frame (default: 1).

Example


```bash
python vidcensor.py input.mp4 output.mp4 "keyword1" "keyword2" --display --nth_frame 10
```

it's really bad, but works for my purposes