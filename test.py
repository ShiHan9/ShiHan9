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

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the captured frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform emotion recognition on the current frame
    results = emotion_recognition(pil_image)

    # Display the results
    for result in results:
        label = result['label']
        confidence = result['score']
        print(f"Detected emotion: {label} with confidence {confidence:.2f}")

        # Display the resulting frame with detected emotion
        cv2.putText(frame, f"{label} ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Recognition', frame)

    # Press 'q' to exit the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
