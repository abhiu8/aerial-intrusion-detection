# AI-Based Aerial Intrusion Detection System

An AI-powered computer vision system for detecting and tracking **humans
and vehicles in aerial/drone imagery and video**.

The project uses a custom-trained **YOLO object detection model** with a
Streamlit interface, providing an easy-to-use platform for aerial
surveillance and intrusion monitoring.

------------------------------------------------------------------------

## Overview

Aerial surveillance produces large amounts of visual data that can be
difficult to monitor manually. This project applies deep learning-based
object detection and tracking to automatically identify relevant objects
in aerial imagery.

The system supports:

-   🖼️ **Aerial image analysis**
-   🎥 **Drone/aerial video analysis**
-   👤 **Human detection**
-   🚗 **Vehicle detection**
-   🎯 **Object tracking with unique IDs**
-   📊 **Unique object counting**
-   📍 **Approximate machine location capture**
-   📝 **Geo-tagged detection logging**
-   🌐 **Interactive Streamlit web interface**

------------------------------------------------------------------------

## System Architecture

``` text
Aerial Image / Drone Video
          │
          ▼
     Streamlit UI
          │
          ▼
   YOLO Detection Model
          │
     ┌────┴────┐
     │         │
 Image       Video
Detection    Tracking
     │         │
     └────┬────┘
          ▼
 Human / Vehicle Detection
          │
          ▼
 Visualization + Object IDs
          │
          ▼
 Detection Statistics
          │
          ▼
 Geo-Tagged CSV Log
```

------------------------------------------------------------------------

## Technology Stack

  Technology           Purpose
  -------------------- -------------------------------------
  Python               Core application
  YOLO / Ultralytics   Object detection and tracking
  OpenCV               Image and video processing
  Streamlit            Web application interface
  NumPy                Numerical/image processing
  Pandas               Detection logging and data handling
  Requests             Approximate machine location lookup

------------------------------------------------------------------------

## Key Features

### Human & Vehicle Detection

The trained YOLO model identifies humans and vehicles in aerial imagery
and highlights detected objects using bounding boxes.

### Video Object Tracking

For video input, YOLO tracking assigns unique IDs to detected objects,
allowing the system to estimate the number of unique humans and vehicles
appearing throughout the video.

### Detection Logging

The system records the first appearance of tracked objects, including:

-   Object ID
-   Detected class
-   Frame number
-   Machine latitude
-   Machine longitude

The information is exported to:

``` text
detection_log.csv
```

### Interactive Web Interface

The Streamlit application provides a simple interface where users can
upload supported aerial images or videos and view the detection results
directly in the browser.

------------------------------------------------------------------------

## Project Structure

``` text
aerial-intrusion-detection/
│
├── app.py
├── best.pt
├── requirements.txt
├── detection_log.csv
└── README.md
```

> `detection_log.csv` is generated after video analysis and does not
> need to be committed to the repository.

------------------------------------------------------------------------

## Installation

Clone the repository and install the required dependencies:

``` bash
git clone <your-repository-url>
cd aerial-intrusion-detection
pip install -r requirements.txt
```

Run the application:

``` bash
streamlit run app.py
```

The application will open in your browser.

------------------------------------------------------------------------

## Supported Input

### Images

``` text
.jpg
.jpeg
.png
```

### Videos

``` text
.mp4
.avi
.mov
```

------------------------------------------------------------------------

## Model

The project uses a custom-trained YOLO model stored as:

``` text
best.pt
```

The application currently maps the model classes as:

``` text
Class 0 → Human
Class 1 → Vehicle
```

The model is used for both image detection and video tracking.

------------------------------------------------------------------------

## Detection Workflow

1.  User uploads an aerial image or video.
2.  The application loads the trained YOLO model.
3.  The model detects humans and vehicles.
4.  Bounding boxes are generated around detected objects.
5.  For videos, detected objects receive tracking IDs.
6.  Unique objects are counted.
7.  Detection information is recorded.
8.  Video detection data is exported to a CSV file.

------------------------------------------------------------------------

## Location Information

The application attempts to obtain the approximate location of the
machine running the application using IP-based geolocation.

This location is intended as **system/location metadata**, not precise
GPS information from the drone.

------------------------------------------------------------------------

## Performance

Detection performance depends on factors such as:

-   Image/video resolution
-   Object size
-   Camera altitude
-   Lighting conditions
-   Camera movement
-   Object visibility
-   Quality and diversity of the training dataset
-   Available hardware

The system is designed as an applied computer-vision prototype for
aerial surveillance scenarios.

------------------------------------------------------------------------

## Limitations

-   IP-based location is approximate and is not equivalent to drone GPS.
-   Detection accuracy depends on the trained model and input quality.
-   Very small or heavily occluded objects may be missed.
-   Tracking performance can vary with camera movement and crowded
    scenes.
-   The current application is designed around the model's trained
    classes.

------------------------------------------------------------------------

## Future Improvements

Potential extensions include:

-   Real-time drone camera integration
-   GPS-based drone coordinates
-   Interactive surveillance maps
-   Alert generation for detected intrusions
-   Detection history and analytics dashboard
-   Database-backed event storage
-   Multi-camera surveillance
-   Improved tracking algorithms
-   Edge-device deployment
-   Model optimization for faster inference

------------------------------------------------------------------------

## Use Cases

The system can serve as a foundation for:

-   Aerial perimeter monitoring
-   Restricted-area surveillance
-   Infrastructure monitoring
-   Border and boundary observation
-   Industrial site monitoring
-   Search and observation workflows
-   Automated aerial security analysis

------------------------------------------------------------------------

## Disclaimer

This project is intended for **research, educational, and authorized
surveillance applications**. It should be deployed responsibly and in
accordance with applicable privacy, safety, and local regulations.

------------------------------------------------------------------------

## License

Add your preferred open-source license here.

For example:

``` text
MIT License
```

------------------------------------------------------------------------

## Author

**Abhishek Kulal**

AI-Based Aerial Intrusion Detection System

