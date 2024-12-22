import streamlit as st
import logging
import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
import pytesseract
from PIL import Image
import tempfile

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load YOLO model
net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Add this after loading the network (after net = cv2.dnn.readNet(...))
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load helmet detection model
model = load_model('helmet-nonhelmet_cnn.h5')
print('Model loaded!!!')

# Add constants at the top of file
HELMET_OFFSET_X = 60
HELMET_OFFSET_Y = 350
HELMET_SIZE_INCREASE = 100
RECT_THICKNESS = 7
FONT_SCALE = 2
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
COLORS = [(0, 255, 0), (0, 0, 255)]  # Add COLORS definition

def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi, dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi / 255.0
        return int(model.predict(helmet_roi)[0][0])
    except Exception as e:
        print(f"Error in helmet detection: {e}")
        return None

def read_license_plate(plate_image):
    
    # Convert the image to grayscale for better OCR results
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    gray_plate = cv2.resize(gray_plate, (224, 224))
     
    gray_plate = cv2.equalizeHist(gray_plate)
    license_plate_text = pytesseract.image_to_string(gray_plate,
                                                     config='--psm 6')
    return license_plate_text.strip()

def preprocess_plate_image(plate_image):
    
    if plate_image is None or plate_image.size == 0:
        return None
        
    # Resize for consistent processing
    plate_image = cv2.resize(plate_image, (240, 80))
    
    # Convert to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return processed

def process_detection(img, i, indexes, boxes, classIds, COLORS):
    if i not in indexes:
        return
        
    x, y, w, h = boxes[i]
    color = [int(c) for c in COLORS[classIds[i]]]
    
    try:
        if classIds[i] != 0:  # bikes
            
            process_bike(img, x, y, w, h, color,i)
    
    except cv2.error as e:
        logging.error(f"Error processing detection: {e}")

def process_bike(img, x, y, w, h, color,i):
    # Initialize label
    label = "unknown"
    
    # Calculate helmet detection area
    x_h = max(0, x - HELMET_OFFSET_X)
    y_h = max(0, y - HELMET_OFFSET_Y)
    w_h = min(w + HELMET_SIZE_INCREASE, img.shape[1] - x_h)
    h_h = min(h + HELMET_SIZE_INCREASE, img.shape[0] - y_h)
    
    # Draw vehicle rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, RECT_THICKNESS)
    
    # Process helmet if ROI is valid
    if y_h > 0 and x_h > 0:
        try:
            h_r = img[y_h:y_h+h_h, x_h:x_h+w_h]
            if h_r.size > 0:  # Check if ROI is not empty
                c = helmet_or_nohelmet(h_r)
                label = ['helmet','no-helmet'][c]
                cv2.putText(img, label, (x,y-100), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, GREEN, 2)
                cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h), BLUE, 10)
        except Exception as e:
            logging.error(f"Error processing helmet: {e}")
    
    # Process license plate
    plate_height = h//3
    plate_y_offset = h//2
    
    try:
        plate_image = img[y+plate_y_offset:y+plate_y_offset+plate_height, x:x+w]
        stored_image = img[y:y+h, x:x+w]
        
        if label == 'no-helmet' and stored_image.size > 0:
            st.image(stored_image, caption="Vehicle with no helmet")
            
        processed_plate = preprocess_plate_image(plate_image)
        
        if processed_plate is not None:
            # Configure Tesseract for license plates
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            license_plate_text = pytesseract.image_to_string(processed_plate, config=custom_config)

            # Clean OCR output
            license_plate_text = ''.join(e for e in license_plate_text if e.isalnum())
        else:
            license_plate_text = "No Text Detected"
        
        cv2.putText(img, license_plate_text, (x,y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2)

            
    except Exception as e:
        logging.error(f"Error processing license plate: {e}")

def process_video(input_path):
    frame_placeholder = st.empty()
    video = cv2.VideoCapture(input_path)
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (888, 500))
    
    while True:
        ret, img = video.read()
        if not ret:
            break
            
        img = imutils.resize(img, height=500)
        
        # Process frame
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        boxes = []
        confidences = []
        classIds = []
        
        # Detection loop
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.3:
                    center_x = int(detection[0] * img.shape[1])
                    center_y = int(detection[1] * img.shape[0])
                    w = int(detection[2] * img.shape[1])
                    h = int(detection[3] * img.shape[0])
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIds.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        for i in range(len(boxes)):
            process_detection(img, i, indexes, boxes, classIds, COLORS)
        
        # Write and display frame
        writer.write(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img_rgb, channels="RGB")

st.title("A.I. On the Road")
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    try:
        # Process video with temp file path
        process_video(video_path)
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        # Cleanup temp file
        if os.path.exists(video_path):
            os.unlink(video_path)
