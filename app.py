import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import pandas as pd
import requests

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="AI-Based Aerial Surveillance",
    layout="wide"
)

st.markdown("##  AI-Based Aerial Intrusion Detection System")
st.markdown(
    "Upload an aerial image or drone video to detect and track humans and vehicles in real time."
)

st.markdown("---")

# ---------------------------------
# MACHINE LOCATION FUNCTION
# ---------------------------------
def get_machine_location():
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        data = response.json()
        loc = data.get("loc", "")
        if loc:
            lat, lon = loc.split(",")
            return float(lat), float(lon)
    except:
        pass
    return None, None


machine_lat, machine_lon = get_machine_location()

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")


with st.spinner("Loading detection model..."):
    model = load_model()

st.success("Model loaded successfully.")

# ---------------------------------
# DISPLAY MACHINE LOCATION
# ---------------------------------
if machine_lat and machine_lon:
    st.info(
        f"💻 System Location → Latitude: {machine_lat} | Longitude: {machine_lon}"
    )
else:
    st.warning("Unable to fetch system location.")

st.markdown("---")

# ---------------------------------
# FILE UPLOADER
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["mp4", "avi", "mov", "jpg", "jpeg", "png"]
)

# ---------------------------------
# MAIN LOGIC
# ---------------------------------
if uploaded_file is not None:

    file_type = uploaded_file.type

    # ==========================================
    # IMAGE PROCESSING
    # ==========================================
    if file_type.startswith("image"):

        with st.spinner("Analyzing image..."):

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            results = model(
                image,
                conf=0.25,
                imgsz=1280,
                verbose=False
            )

            annotated_image = image.copy()

            human_count = 0
            vehicle_count = 0

            if results and results[0].boxes is not None:

                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                for box, cls in zip(boxes, classes):

                    x1, y1, x2, y2 = map(int, box)

                    if int(cls) == 0:
                        human_count += 1
                        color = (0, 255, 0)      # Green for humans
                    elif int(cls) == 1:
                        vehicle_count += 1
                        color = (255, 0, 0)      # Blue for vehicles
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(
                        annotated_image,
                        (x1, y1), (x2, y2),
                        color, 2
                    )

            st.image(annotated_image, channels="BGR")

            col1, col2 = st.columns(2)
            col1.success(f"👤 Humans Detected: {human_count}")
            col2.success(f"🚗 Vehicles Detected: {vehicle_count}")

    # ==========================================
    # VIDEO PROCESSING
    # ==========================================
    elif file_type.startswith("video"):

        with st.spinner("Processing video... This may take a moment."):

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            counter_display = st.empty()

            human_ids = set()
            vehicle_ids = set()
            detection_log = []

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frame = cv2.resize(frame, (960, 540))

                results = model.track(
                    frame,
                    persist=True,
                    conf=0.25,
                    iou=0.5,
                    imgsz=960,
                    verbose=False
                )

                annotated_frame = frame.copy()

                if results and results[0].boxes.id is not None:

                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()

                    for box, obj_id, cls in zip(boxes, ids, classes):

                        x1, y1, x2, y2 = map(int, box)

                        if int(cls) == 0:
                            color = (0, 255, 0)
                            human_ids.add(obj_id)
                            label = "Human"
                        elif int(cls) == 1:
                            color = (255, 0, 0)
                            vehicle_ids.add(obj_id)
                            label = "Vehicle"
                        else:
                            color = (0, 0, 255)
                            label = "Object"

                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1), (x2, y2),
                            color, 2
                        )

                        cv2.putText(
                            annotated_frame,
                            f"{label} | ID {int(obj_id)}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                        # Log first appearance only
                        if not any(log["Object ID"] == int(obj_id) for log in detection_log):
                            detection_log.append({
                                "Object ID": int(obj_id),
                                "Class": label,
                                "Frame": frame_count,
                                "Machine Latitude": machine_lat,
                                "Machine Longitude": machine_lon
                            })

                counter_display.markdown(
                    f"""
                    ### 👤 Unique Humans: {len(human_ids)}
                    ### 🚗 Unique Vehicles: {len(vehicle_ids)}
                    """
                )

                stframe.image(annotated_frame, channels="BGR")

            cap.release()

            if detection_log:
                df = pd.DataFrame(detection_log)
                df.to_csv("detection_log.csv", index=False)
                st.success("Geo-tagged detection log saved as detection_log.csv")

        st.success("Video analysis complete.")

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")
st.markdown("AI-Based Aerial Surveillance System")