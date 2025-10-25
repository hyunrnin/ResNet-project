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