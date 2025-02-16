# face-mask-detection

This project focuses on real-time face mask detection using YOLOv8, the latest iteration of the YOLO (You Only Look Once) object detection models. With the increasing importance of wearing masks in public spaces, this system can help monitor and enforce mask-wearing policies efficiently.

The model is trained to detect three categories:
✅ People wearing masks properly
❌ People without masks
⚠️ People wearing masks incorrectly

# How It Works
1. The input (image or video) is processed.
2. YOLOv8 detects faces and classifies them into one of the three categories.
3. The model draws bounding boxes around detected faces with labels.
4. The processed image/video is displayed with detections.

# How to Run
1. Clone the repository:
  git clone https://github.com/Dimitarbalo/face-mask-detection.git\n
  cd face-mask-detection\n
2. Install dependencies
   pip install -r requirements.txt
3. Run the model on an image or video:
   python detect_mask.py --source image.jpg  # For a single image  
   python detect_mask.py --source video.mp4  # For a video  


