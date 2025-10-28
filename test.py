import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
import json
import numpy as np
import pandas as pd

# =========================
# 1. 기본 설정
# =========================
FILE_PATH = r'C:\2025-2 학부연구\model.pth'
IMAGE_PATH = r'C:\2025-2 학부연구\real\02_11_output'
CLASS_NAMES_FILE_PATH = r'C:\2025-2 학부연구\class_names.txt'
OUTPUT_JSON_PATH = r'C:\2025-2 학부연구\classification_results.json'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# =========================
# 2. 클래스 이름 로드
# =========================
try:
    with open(CLASS_NAMES_FILE_PATH, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f]
    num_classes = len(class_names)
    print(f"클래스 이름 로드 완료: {class_names}")
except FileNotFoundError:
    print(f"Error: '{CLASS_NAMES_FILE_PATH}' 파일을 찾을 수 없습니다.")
    sys.exit(1)

# =========================
# 3. 모델 로드 및 가중치 적용
# =========================
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

try:
    model.load_state_dict(torch.load(FILE_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"모델 가중치 '{FILE_PATH}' 로드 완료 및 평가 모드 설정 완료.")
except FileNotFoundError:
    print(f"Error: 모델 가중치 파일 '{FILE_PATH}'을 찾을 수 없습니다.")
    sys.exit(1)

# =========================
# 4. 이미지 전처리
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# =========================
# 5. 이미지 분류 함수
# =========================
def classify_image(image_path, model, transform, class_names, device):
    if not os.path.exists(image_path):
        print(f"Error: 이미지 파일 '{image_path}'을 찾을 수 없습니다.")
        return None

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error: 이미지 로드 오류: {e}")
        return None

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)[0]
    predicted_index = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index].item()

    prob_dict = {class_names[i]: prob.item() for i, prob in enumerate(probabilities)}

    return {
        "image_path": image_path,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict
    }

# =========================
# 6. 이미지 폴더 처리
# =========================
classification_results_data = []

if os.path.isdir(IMAGE_PATH):
    print(f"'{IMAGE_PATH}' 폴더 내의 이미지들을 분류합니다.")
    image_files = [
        os.path.join(IMAGE_PATH, f)
        for f in os.listdir(IMAGE_PATH)
        if os.path.isfile(os.path.join(IMAGE_PATH, f))
        and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    ]

    if not image_files:
        print("지원되는 이미지 파일을 찾을 수 없습니다.")
    else:
        for image_file in image_files:
            result = classify_image(image_file, model, inference_transform, class_names, device)
            if result is not None:
                classification_results_data.append(result)

else:
    print(f"단일 이미지 '{IMAGE_PATH}'을 분류합니다.")
    result = classify_image(IMAGE_PATH, model, inference_transform, class_names, device)
    if result is not None:
        classification_results_data.append(result)

print(f"\n분류 완료: 총 {len(classification_results_data)}개의 이미지가 처리되었습니다.")

# =========================
# 7. JSON 저장
# =========================
try:
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(classification_results_data, f, indent=4, ensure_ascii=False)
    print(f"분류 결과가 '{OUTPUT_JSON_PATH}'에 저장되었습니다.")
except Exception as e:
    print(f"Error: JSON 저장 중 오류 발생: {e}")

# =========================
# 8. 통계 분석
# =========================
try:
    classification_results_df = pd.read_json(OUTPUT_JSON_PATH)
    print(f"\n'{OUTPUT_JSON_PATH}' 로부터 데이터 로드 완료.")
    print("\n처음 5개 항목:")
    print(classification_results_df.head())
except FileNotFoundError:
    print(f"Error: '{OUTPUT_JSON_PATH}' 파일을 찾을 수 없습니다.")
    sys.exit(1)
except Exception as e:
    print(f"Error: JSON 파일 로드 중 오류 발생: {e}")
    sys.exit(1)

# 클래스별 예측 수
class_counts = classification_results_df['predicted_class'].value_counts()
print("\n--- 분류 통계 ---")
print("클래스별 예측 개수:")
print(class_counts)

# 평균 확신도
average_confidence = classification_results_df['confidence'].mean()
print(f"\n평균 확신도: {average_confidence:.2f}")
print("------------------")