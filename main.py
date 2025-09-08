import cv2
import numpy as np
import argparse
from imutils.object_detection import non_max_suppression
import imutils

class PeopleDetector:
    def __init__(self):
        
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
    def detect_people(self, frame):
        """
        Detect people in a frame and return bounding boxes and count
        """
        
        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        orig_frame = frame.copy()
        
        
        (rects, weights) = self.hog.detectMultiScale(
            frame, 
            winStride=(4, 4), 
            padding=(8, 8), 
            scale=1.05
        )
        
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        
        person_count = 0
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            person_count += 1
            
        cv2.putText(frame, f"People: {person_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, person_count

def process_image(image_path, output_path=None):
    """
    Process a single image for person detection
    """
    detector = PeopleDetector()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    processed_image, count = detector.detect_people(image)
    
    cv2.imshow("People Detection", processed_image)
    print(f"Detected {count} people in the image")
    
    if output_path:
        cv2.imwrite(output_path, processed_image)
        print(f"Result saved to {output_path}")
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, output_path=None):
    """
    Process a video file for person detection
    """
    detector = PeopleDetector()
    
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        
        processed_frame, count = detector.detect_people(frame)
        
        
        cv2.imshow("People Detection", processed_frame)
        
        
        if output_path:
            out.write(processed_frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def process_webcam(output_path=None):
    """
    Process live webcam feed for person detection
    """
    detector = PeopleDetector()
    
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, count = detector.detect_people(frame)
        
        cv2.imshow("People Detection", processed_frame)
        
        
        if output_path:
            out.write(processed_frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    
    parser = argparse.ArgumentParser(description='People Detection Application')
    parser.add_argument('-i', '--image', type=str, help='Path to input image')
    parser.add_argument('-v', '--video', type=str, help='Path to input video')
    parser.add_argument('-w', '--webcam', action='store_true', help='Use webcam')
    parser.add_argument('-o', '--output', type=str, help='Path to output file')
    
    args = parser.parse_args()
    
    if args.image:
        process_image(args.image, args.output)
    elif args.video:
        process_video(args.video, args.output)
    elif args.webcam:
        process_webcam(args.output)
    else:
        print("Please specify an input source: --image, --video, or --webcam")

if __name__ == "__main__":
    main()