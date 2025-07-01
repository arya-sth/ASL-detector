# ASL Detector with Edge Computing

This project leverages machine learning and edge computing to recognize and translate American Sign Language (ASL) gestures in real-time. By using the Jetson Nano for on-the-spot processing, we eliminate the need for continuous cloud connections, making communication smoother for ASL users. To ensure inclusivity, we developed our own dataset, incorporating diverse skin tones for more accurate recognition.

## Tools & Technologies Used
- **Jetson Nano**: The edge device used for real-time processing.
- **TensorFlow**: For building and optimizing machine learning models.
- **Roboflow**: For dataset management and model training.
- **MediaPipe**: For detecting and tracking ASL hand landmarks.
- **Kaggle**: For accessing and enhancing the ASL dataset.
- **OpenCV**: For image processing and camera configuration.
- **GStreamer**: For configuring real-time video streaming on Jetson Nano.

## Project Phases

### Phase 1: Setting Up MediaPipe
- Installed and configured MediaPipe on Jetson Nano with Python 3.8.
- Resolved compatibility issues by downloading necessary `.whl` files.

### Phase 2: Installing OpenCV and Dependencies
- Uninstalled and reinstalled OpenCV to ensure compatibility with MediaPipe.
- Configured OpenCV to work with GStreamer.

### Phase 3: Camera Configuration
- Configured camera using OpenCV and GStreamer.
- Captured real-time images to create an ASL gesture dataset.

### Phase 4: Model Training with TensorFlow
- Installed TensorFlow and trained the model using a custom dataset of ASL gestures.
- Automated image collection and labeling via Python scripts.

### Phase 5: Refining Data Collection
- Switched to a USB camera for consistent data collection.
- Fine-tuned camera settings to improve image quality.

### Phase 6: Model Inference and Roboflow Integration
- Integrated a pre-trained ASL gesture model using Roboflow.
- Applied bounding box annotations to gestures using the Supervision library for real-time feedback.

## Technical Challenges
- **MediaPipe Compatibility**: Solved compatibility issues between MediaPipe and Jetson Nano L4T 32.7.1.
- **OpenCV Conflicts**: Fixed OpenCV version conflicts that affected integration with MediaPipe.
- **Camera Configuration**: Overcame camera configuration issues, eventually opting for a USB camera for better performance.

## Skills Developed
- **Edge Computing**: Gained experience with deploying models on edge devices like Jetson Nano.
- **Machine Learning**: Improved skills in model training, fine-tuning, and optimization for real-time inference.
- **Computer Vision**: Hands-on experience in video processing and gesture detection with OpenCV and MediaPipe.
- **Troubleshooting**: Addressed challenges related to dependencies, camera setup, and device compatibility.
