import cv2
import json
import os
import glob
import numpy as np

JSON_FOLDER = r""
OUTPUT_BASE = r""

MOTION_THRESH = 5.0
HASH_DIST_THRESH = 5
USE_HASH_FILTER = False

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
    raise FileNotFoundError(f"JSON 파일이 없습니다: {JSON_FOLDER}")

print(f"[+] {len(json_files)}개의 JSON 파일을 찾았습니다.\n")

total_label_count = {"false": 0, "true": 0, "unknown": 0}
total_saved = 0

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

    for lbl_name in {"false", "true", "unknown"}:
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

    print(f"\n[처리 시작] {video_name}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 열기 실패: {video_path}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    총 {total_frames}프레임")

    prev_roi_gray = None
    prev_label = None
    saved_count = 0
    label_count = {"false": 0, "true": 0, "unknown": 0}
    recent_hashes = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_label = None
        for seg in segments:
            if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
                lbl_id = int(seg.get("label", 0))
                current_label = label_map.get(lbl_id, "unknown")
                break
        if current_label is None:
            frame_idx += 1
            continue

        roi_frame = frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        save_this = False
        if current_label != prev_label:
            save_this = True
        elif prev_roi_gray is not None:
            diff = np.mean(cv2.absdiff(roi_gray, prev_roi_gray))
            if diff >= MOTION_THRESH:
                save_this = True

        if save_this and USE_HASH_FILTER:
            ah = avg_hash(roi_frame)
            if any(hamming_distance(ah, h) <= HASH_DIST_THRESH for h in recent_hashes):
                save_this = False
            else:
                recent_hashes.append(ah)
                if len(recent_hashes) > 5000:
                    recent_hashes = recent_hashes[-2000:]

        if save_this:
            save_dir = os.path.join(output_dir, current_label)
            filename = f"{video_name}_frame_{frame_idx:05d}.jpg"
            save_path = os.path.join(save_dir, filename)
            success, encoded = cv2.imencode(".jpg", roi_frame)
            if success:
                encoded.tofile(save_path)
                saved_count += 1
                label_count[current_label] += 1

        prev_label = current_label
        prev_roi_gray = roi_gray
        frame_idx += 1

    cap.release()

    print(f"\n[{video_name}] 저장 결과")
    print(f"    총 저장된 이미지 수: {saved_count}")
    for lbl, cnt in label_count.items():
        print(f"    - {lbl:8s}: {cnt:4d}장")

    for lbl in label_count:
        total_label_count[lbl] += label_count[lbl]
    total_saved += saved_count

print("\n============================")
print("전체 처리 완료 요약")
print(f"총 저장된 이미지 수: {total_saved}장")
for lbl, cnt in total_label_count.items():
    print(f"    - {lbl:8s}: {cnt:4d}장")
print("============================\n")
