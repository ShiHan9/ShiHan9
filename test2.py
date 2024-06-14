import cv2
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, pipeline

# Load the model and feature extractor
model_name = "dima806/facial_emotions_image_detection"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Initialize the emotion recognition pipeline
emotion_recognition = pipeline('image-classification', model=model_name, feature_extractor=feature_extractor)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Could not open webcam")
    exit()

# Define the position for the text
text_position = (50, 50)
text_color = (0, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2

# Define the rectangle background color
rectangle_bgr = (0, 0, 0)

# Initialize the emotion label
emotion_label = ""

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Clear the previous result by drawing a filled rectangle
    (text_width, text_height), _ = cv2.getTextSize(emotion_label, font, font_scale, thickness)
    frame = cv2.rectangle(frame, text_position, (text_position[0] + text_width, text_position[1] - text_height), rectangle_bgr, -1)

    # Convert the captured frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform emotion recognition on the current frame
    results = emotion_recognition(pil_image)

    # Sort the results by confidence score in descending order
    results.sort(key=lambda x: x['score'], reverse=True)

    # Get the highest confidence result
    if results:
        highest_confidence_result = results[0]
        emotion_label = f"{highest_confidence_result['label']} ({highest_confidence_result['score']:.2f})"
        print(f"Detected emotion: {emotion_label}")

        # Display the resulting frame with detected emotion
        cv2.putText(frame, emotion_label, text_position, font, font_scale, text_color, thickness)

    cv2.imshow('Emotion Recognition', frame)

    # Press 'q' to exit the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
