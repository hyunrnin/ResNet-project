import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
import numpy as np

# --- 1. 기본 설정 ---
FILE_PATH = 'resnet_trained_weights.pth' # train에서 학습한 모델 가져오기
IMAGE_PATH = 'C:/2025-2 학부연구-ResNet 구현/test_image/sample.jpg' # 분류할 이미지 경로를 지정하세요!

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# --- 2. 클래스 이름 로드 ---
try:
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f]
    num_classes = len(class_names)
    print(f"클래스 이름 로드 완료: {class_names}")
except FileNotFoundError:
    print("Error: 'class_names.txt' 파일을 찾을 수 없습니다. train.py를 실행하여 파일을 생성하세요.")
    sys.exit(1)


# --- 3. 모델 로드 및 가중치 적용 ---
model = models.resnet18(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes) 

try:
    model.load_state_dict(torch.load(FILE_PATH, map_location=device))
    model = model.to(device)
    model.eval() 
    print(f"모델 가중치 '{FILE_PATH}' 로드 및 평가 모드 설정 완료.")
except FileNotFoundError:
    print(f"Error: 모델 가중치 파일 '{FILE_PATH}'을 찾을 수 없습니다. train.py를 먼저 실행하여 모델을 학습하고 저장하세요.")
    sys.exit(1)


# --- 4. 이미지 전처리 정의  ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


# --- 5. 이미지 분류 함수 ---
def classify_image(image_path, model, transform, class_names, device):
    """지정된 경로의 이미지를 분류하고 결과를 출력합니다."""
    
    if not os.path.exists(image_path):
        return f"Error: 이미지 파일 '{image_path}'을 찾을 수 없습니다."

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return f"Error: 이미지 로드 또는 변환 중 오류 발생: {e}"

    input_tensor = transform(image).unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        output = model(input_tensor)
        
    # 확률 변환 및 예측
    probabilities = torch.softmax(output, dim=1)[0]
    predicted_index = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index].item()
    
    # 결과 출력
    result = "\n--- 이미지 분류 결과 ---"
    result += f"\n입력 이미지 경로: {image_path}"
    result += f"\n예측 클래스: {predicted_class}"
    result += f"\n확신도: {confidence * 100:.2f}%"
    result += "\n------------------------"
    
    # 모든 클래스의 확률 (Confusion Matrix 대신 상세 출력)
    result += "\n[모든 클래스에 대한 확률 분포]"
    
    # 확률을 내림차순으로 정렬
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    
    for idx, prob in zip(sorted_indices.cpu().numpy(), sorted_probs.cpu().numpy()):
        class_name = class_names[idx]
        result += f"\n- {class_name}: {prob * 100:.2f}%"
        
    return result

# --- 6. 실행 ---
classification_result = classify_image(IMAGE_PATH, model, inference_transform, class_names, device)
print(classification_result)