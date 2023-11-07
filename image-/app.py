from flask import Flask, render_template, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# Load a pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class labels for the ImageNet dataset (for example purposes)
class_labels = ["class_0", "class_1", "class_2", "class_3"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    image = request.files['image']
    
    try:
        img = Image.open(image)
        img = transform(img)
        img = img.unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img)
        
        _, predicted = outputs.max(1)
        label = class_labels[predicted.item()]
        
        return jsonify({'label': label})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
