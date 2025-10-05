"""
Forest Fire Detection - Detection Script
Detect fire in images, videos, or webcam!
"""

import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from PIL import Image
import argparse
import os

class FireDetector:
    """Fire Detection System"""
    
    def __init__(self, model_path):
        """Load the trained model"""
        print(f"Loading model: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully!\n")
    
    def preprocess_image(self, image_path):
        """Prepare image for prediction"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def detect_fire_image(self, image_path):
        """Detect fire in a single image"""
        
        print(f"🔍 Analyzing: {image_path}\n")
        
        # Preprocess
        img_array = self.preprocess_image(image_path)
        
        # Predict
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        # Interpret
        has_fire = prediction > 0.5
        confidence = prediction * 100 if has_fire else (1 - prediction) * 100
        
        # Display results
        print("="*60)
        if has_fire:
            print("🔥 FIRE DETECTED!")
            print("⚠️  ALERT: Fire found in image")
            print(f"🎯 Confidence: {confidence:.2f}%")
        else:
            print("✅ NO FIRE DETECTED")
            print("✓  Status: Safe")
            print(f"🎯 Confidence: {confidence:.2f}%")
        print("="*60 + "\n")
        
        # Show image with result
        self.show_result(image_path, has_fire, confidence)
        
        return has_fire, confidence
    
    def show_result(self, image_path, has_fire, confidence):
        """Display image with detection result"""
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Draw banner
        banner_height = 80
        if has_fire:
            color = (255, 0, 0)  # Red
            text = "FIRE DETECTED!"
        else:
            color = (0, 255, 0)  # Green
            text = "NO FIRE"
        
        # Create overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), color, -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        # Add text
        cv2.putText(img, text, (20, 50), 
                   cv2.FONT_HERSHEY_BOLD, 1.5, (255, 255, 255), 3)
        
        # Add confidence
        conf_text = f"Confidence: {confidence:.1f}%"
        cv2.putText(img, conf_text, (20, h-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Convert back to BGR for display
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save result
        output_path = 'results/detection_result.jpg'
        os.makedirs('results', exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"💾 Result saved: {output_path}")
        
        # Display
        cv2.imshow('Fire Detection Result', img)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def detect_fire_video(self, video_source, save_output=True):
        """Detect fire in video or webcam"""
        
        if video_source == 0:
            print("📹 Starting webcam detection...")
            print("Press 'q' to quit\n")
        else:
            print(f"📹 Processing video: {video_source}\n")
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open video source")
            return
        
        # Video writer setup
        if save_output and video_source != 0:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            os.makedirs('results', exist_ok=True)
            out = cv2.VideoWriter('results/output.avi',
                                cv2.VideoWriter_fourcc(*'XVID'),
                                fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame temporarily
            temp_path = 'temp_frame.jpg'
            cv2.imwrite(temp_path, frame)
            
            # Detect
            img_array = self.preprocess_image(temp_path)
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            has_fire = prediction > 0.5
            confidence = prediction * 100 if has_fire else (1 - prediction) * 100
            
            # Draw results
            if has_fire:
                color = (0, 0, 255)  # Red
                text = f"FIRE DETECTED! ({confidence:.1f}%)"
            else:
                color = (0, 255, 0)  # Green
                text = f"No Fire ({confidence:.1f}%)"
            
            # Add banner
            cv2.rectangle(frame, (10, 10), (500, 60), color, -1)
            cv2.putText(frame, text, (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Fire Detection', frame)
            
            # Save frame
            if save_output and video_source != 0:
                out.write(frame)
            
            frame_count += 1
            
            # Quit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_output and video_source != 0:
            out.release()
            print(f"\n💾 Output video saved: results/output.avi")
        cv2.destroyAllWindows()
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"\n✓ Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔥 FOREST FIRE DETECTION SYSTEM")
    print("="*60 + "\n")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found at '{args.model}'")
        print("\nTrain a model first:")
        print("  python src/train.py --model transfer --epochs 15")
        return
    
    # Create detector
    detector = FireDetector(args.model)
    
    # Detect based on input
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Error: Image not found at '{args.image}'")
            return
        detector.detect_fire_image(args.image)
    
    elif args.video:
        if not os.path.exists(args.video):
            print(f"❌ Error: Video not found at '{args.video}'")
            return
        detector.detect_fire_video(args.video)
    
    elif args.webcam:
        detector.detect_fire_video(0)
    
    else:
        print("❌ Error: Please specify --image, --video, or --webcam")
        print("\nExamples:")
        print("  python src/detect.py --model models/best.h5 --image test.jpg")
        print("  python src/detect.py --model models/best.h5 --video test.mp4")
        print("  python src/detect.py --model models/best.h5 --webcam")


if __name__ == '__main__':
    main()