# AI Gym Form Analyzer

Advanced computer vision system that analyzes workout form using pose estimation and provides real-time feedback to help users improve their exercise technique.

## Features

- Real-time pose estimation and form analysis
- Exercise repetition counting
- Form quality assessment
- Detailed feedback on technique errors
- Support for multiple exercises (squats, deadlifts, bench press)
- Movement path tracking
- Real-time visual feedback

## Technical Stack

- OpenCV for video processing
- MediaPipe for pose estimation
- TensorFlow for form classification
- NumPy for numerical computations
- scikit-learn for data preprocessing

## Requirements

```
python>=3.8
opencv-python>=4.5.0
mediapipe>=0.8.9
tensorflow>=2.8.0
numpy>=1.21.0
scikit-learn>=1.0.2
```

## Installation

```bash
git clone https://github.com/jk08y/gym-form-analyzer.git
cd gym-form-analyzer
pip install -r requirements.txt
```

## Usage

Basic usage:
```python
from gym_form_analyzer import GymFormAnalyzer

# Initialize analyzer
analyzer = GymFormAnalyzer()

# Start real-time analysis
analyzer.start_analysis()
```

Custom video source:
```python
# Analyze video file
analyzer.start_analysis(video_source='path/to/video.mp4')
```

## Project Structure

```
gym-form-analyzer/
├── src/
│   ├── __init__.py
│   ├── gym_form_analyzer.py
│   ├── pose_estimation/
│   │   ├── __init__.py
│   │   └── pose_detector.py
│   ├── movement_analysis/
│   │   ├── __init__.py
│   │   └── movement_analyzer.py
│   └── utils/
│       ├── __init__.py
│       ├── angle_calculator.py
│       └── visualization.py
├── models/
│   └── form_classifier.h5
├── tests/
│   ├── __init__.py
│   └── test_analyzer.py
├── requirements.txt
└── README.md
```

## Model Training

The form classification model can be trained on custom data:

```python
# Train model with your dataset
analyzer.form_classifier.fit(
    X_train,  # Pose landmarks features
    y_train,  # Form quality labels
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Save trained model
analyzer.save_model('models/custom_classifier.h5')
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Contact

GitHub: [@jk08y](https://github.com/jk08y)
