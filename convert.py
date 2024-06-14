from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch

# Load the model
model_name = "dima806/facial_emotions_image_detection"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Extract the height and width from the feature_extractor's size attribute
height = feature_extractor.size['height']
width = feature_extractor.size['width']

# Create a dummy input tensor
dummy_input = torch.randn(1, 3, height, width)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "model.onnx", input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
