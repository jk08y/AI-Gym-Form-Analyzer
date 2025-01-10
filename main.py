import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import math
import time
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class GymFormAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize movement analyzer
        self.movement_memory = deque(maxlen=30)  # Store last 30 frames
        self.rep_counter = 0
        self.current_stage = "preparing"  # preparing, down, up
        
        # Load pre-trained model for form classification
        self.form_classifier = self._build_form_classifier()
        self.scaler = StandardScaler()
        
        # Exercise-specific angle thresholds
        self.exercise_configs = {
            'squat': {
                'knee_angle_threshold': 130,
                'hip_angle_threshold': 120,
                'ankle_angle_threshold': 80
            },
            'deadlift': {
                'back_angle_threshold': 150,
                'knee_angle_threshold': 140,
                'hip_angle_threshold': 130
            },
            'bench_press': {
                'elbow_angle_threshold': 90,
                'shoulder_angle_threshold': 45,
                'wrist_angle_threshold': 170
            }
        }
    
    def _build_form_classifier(self):
        """Build and return the form classification model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(33*3,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: good, moderate, poor form
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                 np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
    
    def extract_pose_features(self, landmarks):
        """Extract relevant features from pose landmarks"""
        features = []
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features)
    
    def analyze_squat_form(self, landmarks):
        """Analyze squat form and provide feedback"""
        # Extract key points
        hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        
        # Calculate key angles
        knee_angle = self.calculate_angle(
            [hip.x, hip.y],
            [knee.x, knee.y],
            [ankle.x, ankle.y]
        )
        
        # Analyze form
        feedback = []
        if knee_angle < self.exercise_configs['squat']['knee_angle_threshold']:
            if knee.x < ankle.x:
                feedback.append("Knees going too far forward")
            feedback.append("Maintain proper knee alignment")
        
        return feedback
    
    def detect_repetition(self, landmarks):
        """Detect and count exercise repetitions"""
        hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y
        self.movement_memory.append(hip)
        
        if len(self.movement_memory) < 10:
            return
            
        # Detect movement pattern
        if self.current_stage == "preparing":
            if hip > np.mean([p for p in self.movement_memory][-5:]) + 0.1:
                self.current_stage = "down"
        elif self.current_stage == "down":
            if hip < np.mean([p for p in self.movement_memory][-5:]) - 0.1:
                self.current_stage = "up"
                self.rep_counter += 1
                
        return self.rep_counter
    
    def process_frame(self, frame, exercise_type='squat'):
        """Process a single frame and return analyzed results"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Extract features
            features = self.extract_pose_features(results.pose_landmarks)
            
            # Analyze form
            if exercise_type == 'squat':
                feedback = self.analyze_squat_form(results.pose_landmarks)
            else:
                feedback = []
            
            # Count repetitions
            reps = self.detect_repetition(results.pose_landmarks)
            
            # Predict form quality
            form_quality = self.predict_form_quality(features)
            
            return {
                'frame': frame,
                'feedback': feedback,
                'reps': reps,
                'form_quality': form_quality,
                'current_stage': self.current_stage
            }
            
        return {'frame': frame}
    
    def predict_form_quality(self, features):
        """Predict the quality of exercise form"""
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.form_classifier.predict(features_scaled)
        quality_index = np.argmax(prediction[0])
        qualities = ['poor', 'moderate', 'good']
        return qualities[quality_index]
    
    def start_analysis(self, video_source=0):
        """Start real-time video analysis"""
        cap = cv2.VideoCapture(video_source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            results = self.process_frame(frame)
            
            # Display feedback
            if 'feedback' in results:
                y_position = 30
                for feedback in results['feedback']:
                    cv2.putText(
                        results['frame'],
                        feedback,
                        (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    y_position += 30
                
                # Display rep counter
                if 'reps' in results:
                    cv2.putText(
                        results['frame'],
                        f"Reps: {results['reps']}",
                        (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
            
            cv2.imshow('Gym Form Analyzer', results['frame'])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    def save_model(self, filepath):
        """Save the trained form classifier"""
        self.form_classifier.save(filepath)
    
    def load_model(self, filepath):
        """Load a trained form classifier"""
        self.form_classifier = tf.keras.models.load_model(filepath)

if __name__ == "__main__":
    analyzer = GymFormAnalyzer()
    print("Starting Gym Form Analysis...")
    analyzer.start_analysis()
