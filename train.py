import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from tqdm.auto import tqdm
import time
import os
import subprocess
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ============================
# 3. Confusion Matrix + 검증 함수
# ============================
def validate_model(model, test_loader, device, class_names, epoch_num=None):
    model.eval()
    correct = 0
    total = 0
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
            all_labels.extend(y_.cpu().numpy())
            all_preds.extend(predicted_index.cpu().numpy())

    accuracy = 100 * correct / total
    model.train()

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    title = f"Confusion Matrix (Epoch {epoch_num if epoch_num is not None else 'Final'})"
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    # plt.show()
    plt.savefig(f"confusion_{epoch_num}.png")

    return accuracy

# ============================
# 8. 학습 루프
# ============================
if __name__ == "__main__":


    # ============================
    # 1. 기본 설정
    # ============================
    batch_size = 1024
    # batch_size = 512   
    num_epoch = 5
    learning_rate = 0.001

    # ============================
    # 2. GPU 상태 출력
    # ============================
    try:
        print("GPU 상태 확인:")
        subprocess.run(["nvidia-smi"])
    except FileNotFoundError:
        print("nvidia-smi 명령어를 찾을 수 없습니다. GPU가 없거나 드라이버가 설치되지 않았습니다.")


    # ============================
    # 4. 데이터 로드 및 전처리
    # ============================
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    with open('split_train.txt', 'r', encoding='utf-8') as f:            
        train_data = ConcatDataset([dset.ImageFolder(root=line.strip(), transform=transform, allow_empty=True) for line in f])
    with open('split_test.txt', 'r', encoding='utf-8') as f:            
        test_data = ConcatDataset([dset.ImageFolder(root=line.strip(), transform=transform, allow_empty=True) for line in f])

# ----------

    # ============================
    # 5. 클래스 불균형 보정 (Weighted Sampler)
    # ============================
    # targets = [label for label in train_data.imgs]  # 모든 이미지의 클래스 라벨
    targets = sum([dset.targets for dset in train_data.datasets], [])
    class_counts = Counter(targets)                    # 각 클래스별 개수

    print("class_counts : ", class_counts)

    num_samples = len(train_data)                      # 전체 샘플 수
    print("num_samples : ", num_samples)

    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples, replacement=True)

# ----------

    # Weighted Sampler 적용한 DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=6, drop_last=True)
    if len(train_loader) == 0:
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=6, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


    num_classes = len(class_counts) # 클래스의 개수
    # class_names = class_counts # 클래스의 원소를 받아 이름을 리턴 
    class_names = train_data.datasets[0].classes
    print(f"데이터셋 클래스 수: {num_classes}")
    print(f"클래스 이름: {class_names}")
    print(f"클래스별 샘플 개수: {class_counts}")

    # ============================
    # 6. 모델 정의
    # ============================
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # ============================
    # 7. 손실 함수 및 옵티마이저
    # ============================
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    start = time.time()
    print("\n학습 시작\n")
    best_accuracy = 0.0

    with open('class_names.txt', 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")

    for i in tqdm(range(num_epoch), desc="Total Training"):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0

        epoch_iterator = tqdm(train_loader, desc=f"Epoch {i+1}/{num_epoch}", leave=False)

        for _, (image, label) in enumerate(epoch_iterator):
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output, y_)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            epoch_iterator.set_postfix({"Loss": f"{current_loss:.4f}"})

        # print("len(train_loader) :     ", len(train_loader))

        avg_loss = total_loss / len(train_loader)
        epoch_elapsed_time = time.time() - epoch_start_time

        print("--" * 10)

        test_accuracy = validate_model(model, test_loader, device, class_names, epoch_num=i + 1)
        print(f"Epoch {i+1}/{num_epoch}, Train Loss: {avg_loss:.6f}, Test Acc: {test_accuracy:.2f}% (Time: {epoch_elapsed_time:.2f}s, CM Plotted)")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), f"model.pth")
            print(f"-> Model saved with improved accuracy: {best_accuracy:.2f}%")

    print("\n학습 완료\n")
    print("--------------------------------------------------")
    print("\n최종 결과\n")
    print("Best Accuracy of Test Data: {:.2f}%".format(best_accuracy))
    print("Total elapsed time: {:.2f} seconds".format(time.time() - start))