import cv2
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openvino.inference_engine import IECore
import subprocess

# Load pre-trained model and tokenizer
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Convert ONNX model to OpenVINO IR format (run this once)
# You may need to adjust the conversion command based on your ONNX model
onnx_model_path = "C:\\Users\\yokon\\OneDrive\\Desktop\\INTEL\\emotion_recognition.onnx"
output_dir = "C:\\Users\\yokon\\OneDrive\\Desktop\\INTEL"
conversion_command = f"mo_onnx.py --input_model {onnx_model_path} --output_dir {output_dir} --data_type FP32"
subprocess.run(conversion_command, shell=True)

# Initialize OpenVINO Inference Engine
ie = IECore()

# Load OpenVINO IR model
model_xml = f"C:\\Users\\yokon\\OneDrive\\Desktop\\INTEL\\model.xml"
model_bin = f"C:\\Users\\yokon\\OneDrive\\Desktop\\INTEL\\model.bin"
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name='CPU', num_requests=1)

# Access webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess frame
        # Example: resize frame to match model input size
        resized_frame = cv2.resize(frame, (224, 224))

        # Perform inference using OpenVINO
        # Convert frame to format expected by the model (e.g., BGR to RGB, normalize pixel values)
        # Example: input_data = np.transpose(resized_frame, (2, 0, 1))[None, :, :, :] / 255.0

        # Run inference
        # Example: output = exec_net.infer(inputs={input_blob: input_data})

        # Post-process output to get predicted emotion
        # Example: predicted_emotion = np.argmax(output['output_blob'])

        # Display the resulting frame with predicted emotion
        cv2.putText(frame, f"Emotion: {predicted_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
