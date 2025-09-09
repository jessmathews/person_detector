import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time

class YOLO11PeopleDetector:
    def __init__(self, model_size='n', conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize YOLO11 people detector
        
        Args:
            model_size (str): Model size - n (nano), s (small), m (medium), l (large), x (extra large)
            conf_threshold (float): Confidence threshold for detection
            iou_threshold (float): IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        model_path = f"yolo11{model_size}.pt"
        self.model = YOLO(model_path)
        
        self.class_names = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]
        
        print(f"Loaded YOLO11{model_size} model with confidence threshold {conf_threshold}")
    
    def detect_people(self, frame):
        """
        Detect people in a frame using YOLO11
        """
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold,
            classes=[0],  # Only detect people (class 0)
            verbose=False
        )
        
        person_count = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    confidence = float(box.conf[0])
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"Person: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    person_count += 1
        
        cv2.putText(frame, f"People: {person_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, person_count

def process_image(image_path, output_path=None, model_size='n', conf_threshold=0.5):
    """
    Process a single image for person detection
    """
    detector = YOLO11PeopleDetector(model_size=model_size, conf_threshold=conf_threshold)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    start_time = time.time()
    processed_image, count = detector.detect_people(image)
    end_time = time.time()
    
    cv2.imshow("People Detection - YOLO11", processed_image)
    print(f"Detected {count} people in the image")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    if output_path:
        cv2.imwrite(output_path, processed_image)
        print(f"Result saved to {output_path}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, output_path=None, model_size='n', conf_threshold=0.5):
    """
    Process a video file for person detection
    """
    detector = YOLO11PeopleDetector(model_size=model_size, conf_threshold=conf_threshold)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Processing video... Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, count = detector.detect_people(frame)
        
        cv2.imshow("People Detection - YOLO11", processed_frame)
        
        if output_path:
            out.write(processed_frame)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}, People detected: {count}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def process_webcam(output_path=None, model_size='n', conf_threshold=0.5):
    """
    Process live webcam feed for person detection
    """
    detector = YOLO11PeopleDetector(model_size=model_size, conf_threshold=conf_threshold)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Processing webcam feed... Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, count = detector.detect_people(frame)
        
        cv2.imshow("People Detection - YOLO11", processed_frame)
        
        if output_path:
            out.write(processed_frame)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}, People detected: {count}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='People Detection Application with YOLO11')
    parser.add_argument('-i', '--image', type=str, help='Path to input image')
    parser.add_argument('-v', '--video', type=str, help='Path to input video')
    parser.add_argument('-w', '--webcam', action='store_true', help='Use webcam')
    parser.add_argument('-o', '--output', type=str, help='Path to output file')
    parser.add_argument('-m', '--model_size', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n (nano), s (small), m (medium), l (large), x (extra large)')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, 
                       help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    if args.image:
        process_image(args.image, args.output, args.model_size, args.confidence)
    elif args.video:
        process_video(args.video, args.output, args.model_size, args.confidence)
    elif args.webcam:
        process_webcam(args.output, args.model_size, args.confidence)
    else:
        print("Please specify an input source: --image, --video, or --webcam")

if __name__ == "__main__":
    main()