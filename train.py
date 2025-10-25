# train.py (Confusion Matrix 출력 기능 추가)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
import os

# Confusion Matrix를 위한 라이브러리 추가
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 기본 설정 및 장치 확인 ---
batch_size = 256 # gpu 따라서 설정하기 (32, 64, 128, 256, 512 이하) 
# 전용 gpu 사용률 -> 9.4 ~ 9.6 GB(배치 256 사용 시)
# batch_size = 512로 설정할 경우, 공유 GPU 메모리 사용

num_epoch = 100 # 100 epoch 실행
learning_rate = 0.001 
FILE_PATH = 'resnet_trained_weights.pth' 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# --- 데이터 로딩 및 ImageNet 정규화 ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) 
])

# 경로 설정
train_path = "C:/2025-2 학부연구-ResNet 구현/ResNet-simulation-sample" 
test_path = "C:/2025-2 학부연구-ResNet 구현/ResNet-real-sample"   

if not (os.path.exists(train_path) and os.path.exists(test_path)):
    print("Error: train_path 또는 test_path 경로를 확인해주세요.")

# load Dataset (train -> simulation, test -> real)
train_data = dset.ImageFolder(root=train_path, transform=transform) # ImageFolder 객체 생성
test_data = dset.ImageFolder(root=test_path, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

num_classes = len(train_data.classes) # .classes 속성을 사용하여 클래스 이름 리스트 반환
# False, True, Unknown으로 3개 
print(f"데이터셋 클래스 수: {num_classes}")

class_names = train_data.classes
print(f"클래스 이름: {class_names}")


# --- 3. 모델 로드 및 수정 ---
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes) 
model = model.to(device)


# --- 4. 손실 함수, 옵티마이저 설정  ---
loss_func = nn.CrossEntropyLoss() # 교차 엔트로피 오차
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam (학습률 lr = 0.001)


# --- 5. 검증 함수 정의 ---
def validate_model(model, test_loader, device, class_names, epoch_num=None):
    """테스트 데이터셋에 대한 정확도와 Confusion Matrix를 계산하고 출력합니다."""
    model.eval() 
    correct = 0
    total = 0
    
    # Confusion Matrix 계산을 위한 리스트
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for image, label in test_loader:
            x = image.to(device)
            y_ = label.to(device)

            output = model.forward(x)
            _, predicted_index = torch.max(output, 1)

            total += label.size(0)
            correct += (predicted_index == y_).sum().item()
            
            # 리스트에 정답과 예측 값을 추가
            all_labels.extend(y_.cpu().numpy())
            all_preds.extend(predicted_index.cpu().numpy())

    accuracy = 100 * correct / total
    model.train() 

    # Confusion Matrix 계산
    cm = confusion_matrix(all_labels, all_preds)
    
    # Confusion Matrix 출력
    plt.figure(figsize=(8, 6))
    title = f"Confusion Matrix (Epoch {epoch_num if epoch_num is not None else 'Final'})"
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.show()

    return accuracy

# --- 6. 학습 루프 및 모델 저장 ---
start = time.time() 
print("\n 학습 시작 \n")

best_accuracy = 0.0 

# 클래스 이름을 저장하여 test.py에서 사용할 수 있도록 함
with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

for i in tqdm(range(num_epoch), desc="Total Training"):
    # ... 학습 루프 (생략) ...
    model.train()
    total_loss = 0.0
    
    epoch_iterator = tqdm(train_loader, desc=f"Epoch {i+1}/{num_epoch}", leave=False)
    
    for _,[image,label] in enumerate(epoch_iterator):
        x = image.to(device)
        y_= label.to(device)

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        total_loss += current_loss
        
        epoch_iterator.set_postfix({"Loss": f"{current_loss:.4f}"})

    avg_loss = total_loss / len(train_loader)
    
    # 매 에폭마다 테스트 성능 출력 (Confusion Matrix는 10 에폭마다 출력하도록 조정)
    if (i + 1) % 10 == 0 or (i == num_epoch - 1):
        test_accuracy = validate_model(model, test_loader, device, class_names, epoch_num=i + 1)
        tqdm.write(f"Epoch {i+1}/{num_epoch}, Train Loss: {avg_loss:.6f}, Test Accuracy: {test_accuracy:.2f}% (CM Plotted)")
    else:
        test_accuracy = validate_model(model, test_loader, device, class_names)
        tqdm.write(f"Epoch {i+1}/{num_epoch}, Train Loss: {avg_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%")

    
    # 베스트 모델 저장
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), FILE_PATH)
        tqdm.write(f"-> Model saved with improved accuracy: {best_accuracy:.2f}%")


print("\n 학습 완료 \n\n")
print("--------------------------------------------------")
print("\n 최종 결과 \n")
print("Best Accurary of Test Data: {:.2f}%".format(best_accuracy))
print("Total elapsed time: {:.2f} seconds".format(time.time() - start))