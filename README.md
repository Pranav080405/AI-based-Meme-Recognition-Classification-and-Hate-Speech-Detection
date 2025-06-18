# AI-based-Meme-Recognition-Classification-and-Hate-Speech-Detection
This project analyzes memes by extracting text, detecting sentiment, classifying images, flagging harmful content, and assigning meme categories using BERT, ResNet-50, and EasyOCR. It provides a full report and visualizes component-wise model accuracy.
here's the code:

import torch
from PIL import Image
import easyocr
from transformers import BertTokenizer, BertForSequenceClassification
from torchvision import models, transforms
import requests
from io import BytesIO
from google.colab import files
import numpy as np
import cv2
import json

# Initialize EasyOCR for text extraction
reader = easyocr.Reader(['en'])

# Load pre-trained BERT model for sentiment analysis
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentiment_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)  # 4 labels: positive, negative, neutral, sarcastic

# Load pre-trained ResNet-50 for image classification
image_model = models.resnet50(pretrained=True)
image_model.eval()

# Load ImageNet labels
response = requests.get("https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json")
imagenet_classes = response.json()
imagenet_labels = {int(k): v[1] for k, v in imagenet_classes.items()}

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Sentiment labels
sentiment_labels = {0: 'positive', 1: 'negative', 2: 'neutral', 3: 'sarcastic'}

# Meme categories
meme_categories = ['Pop culture', 'Anime', 'Games', 'Sports', 'Political', 'Humour', 'Motivation']

# Harmful content keywords
harmful_keywords = ['hate', 'kill', 'attack', 'racist', 'sexist', 'offensive']

def extract_text_from_image(image):
    """Extract text from an image using EasyOCR."""
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    result = reader.readtext(image_np)
    extracted_text = " ".join([text[1] for text in result])
    return extracted_text

def analyze_sentiment(text):
    """Analyze sentiment of the extracted text using BERT."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze(0)
    sentiment_id = torch.argmax(probs).item()
    sentiment_score = round(probs[sentiment_id].item() * 9) + 1  # Scale to 1-10
    return sentiment_labels[sentiment_id], sentiment_score

def classify_image(image):
    """Classify the image using ResNet-50."""
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = image_model(image_tensor)
    _, predicted_idx = torch.max(outputs, 1)
    return imagenet_labels.get(predicted_idx.item(), "Unknown")

def detect_harmful_content(text):
    """Detect harmful or offensive content in the text."""
    harmful = any(keyword in text.lower() for keyword in harmful_keywords)
    return harmful

def categorize_meme():
    """Randomly assign a meme category (Placeholder: Enhance with better classification later)."""
    return np.random.choice(meme_categories)

def analyze_meme(image):
    """Analyze a meme image and text."""
    extracted_text = extract_text_from_image(image)
    sentiment, sentiment_score = analyze_sentiment(extracted_text)
    image_class = classify_image(image)
    harmful = detect_harmful_content(extracted_text)
    meme_category = categorize_meme()
    
    print("\n📌 Meme Analysis Report:")
    print("-----------------------------------")
    print(f"📝 Extracted Text: \n\"{extracted_text}\"")
    print(f"😊 Sentiment: {sentiment} (Score: {sentiment_score}/10)")
    print(f"🖼️ Image Classification: {image_class}")
    print(f"🚨 Harmful Content: {'Yes' if harmful else 'No'}")
    print(f"🏷️ Meme Category: {meme_category}")
    print("-----------------------------------\n")
    
    return {
        "extracted_text": extracted_text,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "image_class": image_class,
        "harmful_content": harmful,
        "meme_category": meme_category
    }

# Upload the file
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
print(f"Uploaded file: {image_path}\n")

# Load and analyze the image
image = load_image_from_path(image_path)
analysis_result = analyze_meme(image)
print(json.dumps(analysis_result, indent=4))






import matplotlib.pyplot as plt
import numpy as np

# Sample accuracy scores
labels = ["Sentiment", "Image Classification", "Harmful Content", "Meme Category"]
accuracy_scores = [85, 78, 92, 80]  # Replace with actual accuracy values

# Plot the graph
plt.figure(figsize=(8, 5))
plt.bar(labels, accuracy_scores, color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Meme Analysis Model Accuracy")
plt.show()
