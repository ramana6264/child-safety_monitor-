"""
Dangerous Object Detection Module using YOLOv8
Detects harmful objects that children should not handle
"""
import cv2
import numpy as np
from ultralytics import YOLO


class DangerDetector:
    """Detects dangerous objects for child safety using YOLOv8."""
    
    # Objects harmful to children
    DANGEROUS_OBJECTS = {
        'knife', 'scissors', 'wine glass', 'spoon', 'hair drier',
        'fork', 'bottle', 'cup', 'bowl', 'toaster', 'microwave',
        'oven', 'stove', 'remote', 'phone', 'lighter', 'matches'
    }
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize YOLO detector.
        
        Args:
            model_path (str): Path to YOLOv8 model weights
        """
        try:
            self.model = YOLO(model_path)
            print(f"✓ YOLO model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load YOLO model: {e}")
            self.model = None
    
    def detect(self, frame):
        """
        Detect dangerous objects in frame.
        
        Args:
            frame (np.ndarray): BGR image frame
            
        Returns:
            tuple: (annotated_frame, dangerous_detections)
                - annotated_frame: Frame with bounding boxes drawn
                - dangerous_detections: List of dicts with dangerous object info
        """
        dangerous_detections = []
        annotated_frame = frame.copy()
        
        if self.model is None:
            return annotated_frame, dangerous_detections
        
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            
            if len(results) == 0:
                return annotated_frame, dangerous_detections
            
            result = results[0]
            
            # Process detections
            for detection in result.boxes.data:
                x1, y1, x2, y2, conf, cls_id = detection.tolist()
                class_id = int(cls_id)
                confidence = float(conf)
                
                # Get class name from COCO dataset
                class_name = result.names.get(class_id, f"Object_{class_id}")
                
                # Check if it's a dangerous object
                if class_name.lower() in self.DANGEROUS_OBJECTS:
                    # Convert to integers for drawing
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # Draw bounding box in RED for danger
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Draw label with background
                    label = f"⚠ {class_name} ({confidence:.2f})"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + label_size[0], y1), (0, 0, 255), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Store detection info
                    dangerous_detections.append({
                        'object': class_name,
                        'confidence': round(confidence, 3),
                        'bbox': {
                            'x': x1,
                            'y': y1,
                            'w': x2 - x1,
                            'h': y2 - y1
                        }
                    })
        
        except Exception as e:
            print(f"✗ YOLO detection error: {e}")
        
        return annotated_frame, dangerous_detections
