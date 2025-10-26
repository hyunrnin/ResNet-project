# readme

## 1. 동영상 현황 및 통계

**real**

- 02_10
    - 총 영상 19개
    - 1개 영상 당 ROI 2개
- 02_11
    - 총 영상 18개
    - 1개 영상 당 ROI 1개
- 05_06
    - 영상 50개
    - 1개 영상 당 ROI 2개

**simulation**

- 02_10
    - 총 영상 2개
    - 1개 영상 당 ROI 2개
- 02_11
    - 총 영상 2개
    - 1개 영상 당 ROI 1개
- 05_06
    - 총 영상 1개
    - 1개 영상 당 ROI 2개

---

## 2. 레이블링 현황 및 통계

```
# 10/25 기준 labeling 완료된 영상
real/
├── 02_10/
│   ├── cctv2-Video-20250620_162948_0
│   ├── cctv2-Video-20250620_162948_1
│   ├── cctv2-Video-20250623_104746_0
│   ├── cctv2-Video-20250623_104746_1
│   ├── cctv2-Video-20250623_123029_0
│   ├── cctv2-Video-20250623_123029_1
│   └──cctv2-Video-20250623_123923_0
│
├── 02_11/
│   ├── cctv2-Video-20250623_110717/
│   ├── cctv2-Video-20250623_110717_01/
│   ├── cctv2-Video-20250623_113734/
│   ├── cctv2-Video-20250623_113734_01/
│   ├── cctv2-Video-20250623_120717/
│   ├── cctv2-Video-20250623_122126/
│   └── cctv2-Video-20250623_133815/
│
└── 05_06/
    

simulation/
├── 02_10/
│   ├── cctv2-Video-20250620_151919
│   └──cctv2-Video-20250623_161117
│
├── 02_11/
│   ├── cctv2-Video-20250620_151128
│   └── cctv2-Video-20250623_161610
│
└── 05_06/
    └── VRN10072_192.168.1.207_1835-Cam06_20250620_143300_20250620_144100_ID_0000
```

- **labeling 진행 상황 (10/25 기준)**
    - **simulation**
        
        
        |  | 02_10 | 02_11 | 05_06 |
        | --- | --- | --- | --- |
        | 분류 O | 2 | 2 | 1 |
        | 분류 X | - | - | - |
        | 전체  | 2 | 2 | 1 |
    - **real**
        
        
        |  | 02_10 | 02_11 | 05_06 |
        | --- | --- | --- | --- |
        | 분류 O | 4 | 7 | - |
        | 분류 X | 15 | 11 | 50 |
        | 전체 | 19 | 18 | 50 |
    
- **각 라벨의 세그먼트 수량**
    
    

---

## 3. 이미지 현황 및 통계

동영상은 JSON으로 구간별 라벨이 지정되어 있습니다.

라벨은 true, false, unknown 으로 구분되며 클래스 비율은 아래 표와 같습니다.

원하는 분포로 보완하기 위해 true 및 unknown 클래스에 가중치 샘플링을 적용하였습니다.

### crop해서 만든 이미지 세트 현황

**simulation**

- **02_10**
    
    cctv2-Video-20250620_151128
    
    False : 14357
    
    True : 0
    
    Unknown : 0
    
    cctv2-Video-20250620_151919
    
    False : 17475
    
    True : 275
    
    Unknown : 106
    
- **02_11**
    
    cctv2-Video-20250623_161117
    
    False : 17470
    
    True : 335
    
    Unknown : 95
    
    cctv2-Video-20250623_161610
    
    False : 14357
    
    True : 0
    
    Unknown : 0
    
- **05_06**
    
    VRN10072_192.168.1.207_1835-Cam06_20250620_143300_20250620_144100_ID_0000
    
    False : 14095
    
    True : 13780
    
    Unknown : 315
    

**real**

- **02_10**
    
    cctv2-Video-20250620_162948
    
    False : 61458
    
    True : 0
    
    Unknown : 0
    
    cctv2-Video-20250623_104746
    
    False : 61435
    
    True : 
    
    Unknown : 
    
    cctv2-Video-20250623_123029
    
    False : 32906
    
    True : 25021
    
    Unknown : 1626
    
    cctv2-Video-20250623_123923
    
    False : 28882
    
    True : 19974
    
    Unknown : 77
    
- **02_11**
    
    cctv2-Video-20250623_110717
    
    False : 61434
    
    True : 0
    
    Unknown : 0
    
    cctv2-Video-20250623_110717_01
    
    False : 47142
    
    True : 0
    
    Unknown : 0
    
    cctv2-Video-20250623_113734
    
    False : 16208
    
    True : 43313
    
    Unknown : 1934
    
    cctv2-Video-20250623_113734_01
    
    False : 1639
    
    True : 0
    
    Unknown : 0
    
    cctv2-Video-20250623_120717
    
    False : 11863
    
    True : 32812
    
    Unknown : 2722
    
    cctv2-Video-20250623_122126
    
    False : 1834
    
    True : 21077
    
    Unknown : 649
    
    cctv2-Video-20250623_133815
    
    False : 0
    
    True : 17012
    
    Unknown : 3909
    

**05_06**

라벨링 진행 중

---

## 4. 훈련/검증 스플릿 기준 및 통계

### **colab 환경**

**train data** 

**simulation 영상에서 랜덤 추출된 이미지 573장**

| 라벨 | 개수 | 비율 |
| --- | --- | --- |
| true | 199 | 34.7% |
| false | 328 | 57.2% |
| unknown | 46 | 8% |

validation 데이터

- 코드
    
    ```python
    import cv2
    import json
    import os
    import glob
    import numpy as np
    import random
    
    JSON_FOLDER = r""
    OUTPUT_BASE = r""
    
    TARGET_PER_VIDEO = 100
    MOTION_THRESH = 5.0
    COOLDOWN = 2
    HASH_DIST_THRESH = 5
    
    LABEL_RATIO = {
        "false": 0.25,
        "true": 0.60,
        "unknown": 0.15
    }
    
    def avg_hash(img_bgr, size=8):
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        mean = g.mean()
        bits = (g > mean).astype(np.uint8).flatten()
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val
    
    def hamming_distance(a, b):
        return (a ^ b).bit_count()
    
    json_files = glob.glob(os.path.join(JSON_FOLDER, "**", "*.json"), recursive=True)
    if not json_files:
        raise FileNotFoundError(f"JSON 파일이 없음: {JSON_FOLDER}")
    
    print(f"[+] {len(json_files)}개의 JSON 파일을 찾았습니다.\n")
    
    total_label_count = {"false": 0, "true": 0, "unknown": 0}
    total_saved = 0
    total_error = 0
    
    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    
        video_path = data["video_path"]
        roi = data["roi"]
        segments = data["segments"]
        label_map = {int(k): v.lower() for k, v in data["label_map"].items()}
    
        x, y, w, h = roi
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(OUTPUT_BASE, video_name)
        os.makedirs(output_dir, exist_ok=True)
        for lbl_name in label_map.values():
            os.makedirs(os.path.join(output_dir, lbl_name), exist_ok=True)
    
        if not os.path.exists(video_path):
            video_filename = os.path.basename(video_path)
            subdir = os.path.basename(os.path.dirname(video_path))
            corrected_path = os.path.join(JSON_FOLDER, subdir, video_filename)
            if os.path.exists(corrected_path):
                print(f"경로 교정됨: {video_path} → {corrected_path}")
                video_path = corrected_path
            else:
                print(f"비디오를 찾을 수 없음: {corrected_path}")
                continue
    
        print(f"\n처리 시작: {video_name}")
        print(f"    Video: {video_path}")
        print(f"    ROI: {roi}")
        print(f"    Segments: {len(segments)}개")
    
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"비디오 열기 실패: {video_path}")
            continue
    
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"영상 정보: {total_frames} frames, {frame_w}x{frame_h}")
    
        rng = np.random.default_rng(42)
        label_frames = {"false": [], "true": [], "unknown": []}
    
        for seg in segments:
            s = int(seg.get("start_frame", 0))
            e = int(seg.get("end_frame", 0))
            lbl_id = int(seg.get("label", 0))
            label_name = label_map.get(lbl_id, "unknown").lower()
            label_frames[label_name].extend(list(range(s, e + 1)))
    
        selected_frames = []
        for lbl, ratio in LABEL_RATIO.items():
            n_select = int(TARGET_PER_VIDEO * ratio)
            frames = label_frames[lbl]
            if len(frames) > n_select:
                chosen = rng.choice(frames, size=n_select, replace=False)
            else:
                chosen = frames
            for fidx in chosen:
                selected_frames.append((int(fidx), lbl))
    
        total_selected = len(selected_frames)
        if total_selected < TARGET_PER_VIDEO:
            deficit = TARGET_PER_VIDEO - total_selected
            all_rest = []
            for lbl, frames in label_frames.items():
                all_rest.extend([(int(f), lbl) for f in frames if (int(f), lbl) not in selected_frames])
            if len(all_rest) > 0:
                extra = rng.choice(all_rest, size=min(deficit, len(all_rest)), replace=False)
                selected_frames.extend(extra)
    
        selected_frames = [(int(f), lbl) for f, lbl in selected_frames]
        selected_frames = sorted(selected_frames, key=lambda x: x[0])
    
        saved_count = 0
        error_count = 0
        prev_roi_gray = None
        cooldown = 0
        recent_hashes = []
        label_count = {"false": 0, "true": 0, "unknown": 0}
    
        frame_idx = 0
        selected_dict = {int(f): lbl for f, lbl in selected_frames}
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx not in selected_dict:
                frame_idx += 1
                continue
    
            label_name = selected_dict[frame_idx]
            x_clipped = max(0, min(x, frame_w - 1))
            y_clipped = max(0, min(y, frame_h - 1))
            w_clipped = min(w, frame_w - x_clipped)
            h_clipped = min(h, frame_h - y_clipped)
            if w_clipped <= 0 or h_clipped <= 0:
                frame_idx += 1
                continue
    
            roi_frame = frame[y_clipped:y_clipped+h_clipped, x_clipped:x_clipped+w_clipped]
            roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    
            save_this = True
            if prev_roi_gray is not None:
                mad = np.mean(cv2.absdiff(roi_gray, prev_roi_gray))
                if mad < MOTION_THRESH:
                    save_this = False
            if cooldown > 0:
                save_this = False
    
            if save_this:
                ah = avg_hash(roi_frame)
                is_dup = any(hamming_distance(ah, h) <= HASH_DIST_THRESH for h in recent_hashes)
                if is_dup:
                    save_this = False
                else:
                    recent_hashes.append(ah)
                    if len(recent_hashes) > 5000:
                        recent_hashes = recent_hashes[-2000:]
    
            if save_this:
                save_dir = os.path.join(output_dir, label_name)
                filename = f"{video_name}_frame_{frame_idx:05d}.jpg"
                save_path = os.path.join(save_dir, filename)
                success, encoded = cv2.imencode(".jpg", roi_frame)
                if success:
                    encoded.tofile(save_path)
                    saved_count += 1
                    label_count[label_name] += 1
                    cooldown = COOLDOWN
                else:
                    error_count += 1
                if saved_count >= TARGET_PER_VIDEO:
                    break
    
            prev_roi_gray = roi_gray
            if not save_this:
                cooldown = max(0, cooldown - 1)
            frame_idx += 1
    
        cap.release()
    
        if saved_count < TARGET_PER_VIDEO:
            remaining = TARGET_PER_VIDEO - saved_count
            print(f"부족한 {remaining}장 랜덤 추가 저장 중...")
            all_pool = [(int(f), lbl) for f, lbl in selected_frames]
            random.shuffle(all_pool)
            for fidx, label_name in all_pool:
                if saved_count >= TARGET_PER_VIDEO:
                    break
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    continue
                roi_frame = frame[y:y+h, x:x+w]
                save_dir = os.path.join(output_dir, label_name)
                filename = f"{video_name}_extra_{fidx:05d}.jpg"
                save_path = os.path.join(save_dir, filename)
                success, encoded = cv2.imencode(".jpg", roi_frame)
                if success:
                    encoded.tofile(save_path)
                    saved_count += 1
                    label_count[label_name] += 1
    
        print(f"저장된 이미지 총합: {saved_count}장 (목표 100장)")
        for lbl, cnt in label_count.items():
            print(f"    {lbl:8s}: {cnt:4d}장")
    
        for lbl in label_count:
            total_label_count[lbl] += label_count[lbl]
        total_saved += saved_count
        total_error += error_count
    
    print("\n전체 처리 완료")
    print(f"총 저장 이미지 수: {total_saved}장")
    for lbl, cnt in total_label_count.items():
        print(f"    {lbl:8s}: {cnt:4d}장")
    ```
    

| 라벨 | 개수 | 비율 |
| --- | --- | --- |
| true | 151 | 41.8% |
| false | 120 | 33.2% |
| unknown | 90 | 24.9% |

아래의 비율에 맞추어 영상마다 label별 이미지를 추출

| False | 0.25 |
| --- | --- |
| True | 0.6 |
| Unknown | 0.15 |
- 학습 결과
    
    [files/resnet_trained_weights.pth](resnet_trained_weights.pth)
    
    [class_names.txt](class_names.txt)
    

![files/train-validation.png](image.png)

test 데이터

real 영상의 02_10, 02_11중 labeling을 마친 11개의 cctv 영상에서 라벨과 상관 없이 500장을 무작위 추출한 이미지, train에 사용된 test 데이터가 들어가지 않도록 한 후 추출하였다.

05_06은 모델 학습 당시 라벨링을 마친 영상이 없었기에 test set으로 넣을 수 없었다.

- 코드
    
    ```python
    import os
    import shutil
    import random
    import glob
    
    # 설정
    SOURCE_DIR = 'real/output'
    TARGET_DIR = 'real/output-random'
    NUM_FILES_TO_COPY = 500
    
    IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"대상 폴더 '{TARGET_DIR}'가 준비되었습니다.")
    
    # 2. 모든 하위 폴더의 이미지 파일 목록 수집
    all_files = []
    
    # real/output/Video1/False, real/output/Video1/True 등 모든 라벨 폴더를 탐색합니다.
    for ext in IMAGE_EXTENSIONS:
        # glob을 사용하여 SOURCE_DIR의 모든 하위 폴더 내의 이미지 파일을 탐색
        # 'real/output/**/{*.jpg, *.png}' 패턴으로 모든 하위 파일을 검색하도록 설정.
        search_path = os.path.join(SOURCE_DIR, '**', ext)
        
        # glob.glob은 OS 경로 구분자를 자동으로 처리합니다.
        all_files.extend(glob.glob(search_path, recursive=True))
    
    if not all_files:
        print(f"경로 '{SOURCE_DIR}'에서 지정된 확장자의 파일을 찾을 수 없습니다. 경로 또는 확장자를 확인하십시오.")
    else:
        print(f"총 {len(all_files)}개의 이미지 파일을 찾았습니다.")
    
    # NUM_FILES_TO_COPY값에 맞는 이미지 개수 추출
    copy_count = min(NUM_FILES_TO_COPY, len(all_files))
        
    # random.sample을 사용하여 랜덤으로 파일을 선택.
    selected_files = random.sample(all_files, copy_count)
    print(f"{copy_count}개의 파일을 무작위로 선택했습니다.")
    
    # 선택된 파일을 대상 폴더에 복사
    print("\n--- 파일 복사 시작 ---")
    for i, file_path in enumerate(selected_files):
        try:
            # 파일 이름만 추출하여 대상 폴더에 복사합니다.
            # 이렇게 하면 라벨링/영상 폴더 구조가 아닌 단일 폴더에 모든 파일이 복사됩니다.
            file_name = os.path.basename(file_path)
            target_path = os.path.join(TARGET_DIR, file_name)
            
            # 파일 이름이 중복되는 경우, 충돌을 방지하기 위해 파일 이름에 인덱스를 추가합니다.
            # (예: 'image.jpg' -> 'image_1.jpg')
            base, ext = os.path.splitext(file_name)
            counter = 1
            original_target_path = target_path
            while os.path.exists(target_path):
                target_path = os.path.join(TARGET_DIR, f"{base}_{counter}{ext}")
                counter += 1
                
            shutil.copy2(file_path, target_path)
            
        except Exception as e:
            print(f"파일 복사 중 오류 발생 ({file_path}): {e}")
    
    print("----------------------")
    print(f"✅ 총 {copy_count}개의 파일이 '{TARGET_DIR}'에 복사되었습니다.")
    ```
    

![files/test.png](image%201.png)

```bash
# validation, test에 사용된 영상들 
real/
├── 02_10/
│   ├── cctv2-Video-20250620_162948
│   ├── cctv2-Video-20250623_104746
│   ├── cctv2-Video-20250623_123029
│   └──cctv2-Video-20250623_123923_0
│
├── 02_11/
│   ├── cctv2-Video-20250623_110717
│   ├── cctv2-Video-20250623_110717_01
│   ├── cctv2-Video-20250623_113734
│   ├── cctv2-Video-20250623_113734_01
│   ├── cctv2-Video-20250623_120717
│   ├── cctv2-Video-20250623_122126
│   └── cctv2-Video-20250623_133815
│
└── 05_06/
  
```

### **local(진행 중)**

**모든 simulation 이미지를 학습에 사용** 

1 에포크 당 대략 12분 소요, 4시간 동안 20 에포크 진행