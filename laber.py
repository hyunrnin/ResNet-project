import cv2
import json
import os
import glob
import bisect
import numpy as np

# --- 스크립트 설정 ---

# 1. CCTV 고유번호(디렉토리명)와 해당 ROI 좌표 목록을 매핑
#    형식: 'CCTV_ID': [ [x1, y1, w1, h1], [x2, y2, w2, h2], ... ]
CCTV_ROI_CONFIG = {
    '02_10': [
        [780, 0, 128, 128],   # 02_10의 첫 번째 ROI
        [1000, 0, 128, 128]  # 02_10의 두 번째 ROI
    ],
    '02_11': [
        [1250, 330, 128, 128]  # 02_11의 첫 번째 ROI
    ],
    '05_06': [
        [1200, 970, 128, 128],  # 05_06의 첫 번째 ROI
        [1500, 915, 128, 128]  # 05_06의 두 번째 ROI
    ]
    # 새로운 CCTV를 추가하려면 이 딕셔너리에 항목을 추가
}

# 2. 레이블 단축키(숫자)와 실제 의미를 매핑
LABEL_MAP = {
    0: 'False',    # 키 '0'
    1: 'True',     # 키 '1'
    2: 'Unknown'   # 키 '2'
}

# 3. 데이터를 탐색할 기본 경로와 하위 폴더 목록
BASE_DIR = 'D:/smart/'  # 데이터 디렉토리
DATA_TYPES = ['real', 'simulation']  # 탐색할 하위 폴더
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.webm']  # 탐색할 비디오 파일 확장자

# --- 헬퍼 함수 ---


def draw_text(img, text, pos, scale, color, thickness=1, bg_color=(0, 0, 0), bg_thickness_factor=2.5):
    """
    OpenCV의 putText 함수를 감싸, 텍스트에 검은색 테두리를 추가하여 가독성을 높임입니다.

    Args:
        img: 텍스트를 그릴 이미지 (numpy array)
        text: 표시할 문자열
        pos: (x, y) 텍스트 시작 좌표
        scale: 폰트 크기 스케일
        color: 텍스트 색상 (B, G, R)
        thickness: 텍스트 굵기
        bg_color: 테두리(배경) 색상
        bg_thickness_factor: 테두리를 텍스트보다 얼마나 더 굵게 할지 결정하는 배율
    """
    bg_thickness = int(thickness * bg_thickness_factor)

    # 1. 테두리(배경) 먼저 그리기
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, bg_color, bg_thickness, cv2.LINE_AA)
    # 2. 원본 텍스트 그리기
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# --- 핵심 레이블링 함수 ---

def interactive_label_video(video_path, roi, out_json):
    """
    비디오 파일의 특정 ROI에 대해 상호작용형 레이블링 GUI를 실행합니다.

    이 함수는 'On-demand seeking'(실시간 프레임 탐색)을 사용하여
    비디오 전체를 메모리에 로드하지 않고도 대용량 파일을 효율적으로 처리합니다.
    또한 ROI 주변의 확대된 영역(Context view)을 함께 표시하여 레이블링을 돕습니다.

    Args:
        video_path (str): 레이블링할 비디오 파일의 전체 경로
        roi (list): [x, y, w, h] 형식의 원본 ROI 좌표
        out_json (str): 레이블링 결과가 저장될 JSON 파일 경로
    """
    x, y, w, h = roi

    # 비디오 캡처 객체 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 파일을 열 수 없습니다: {video_path}")

    # 비디오 메타데이터 읽기
    actual_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 파일 헤더의 총 프레임 수
    try:
        frame_w_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 원본 비디오 너비
        frame_h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 원본 비디오 높이
    except:
        print(f"[!] 비디오 메타데이터(크기)를 읽을 수 없습니다: {video_path}")
        cap.release()
        return

    if actual_total == 0:
        print(f"[!] 비디오에 프레임이 0개입니다 (건너뛰기): {video_path}")
        cap.release()
        return

    print(f"[+] 비디오 열기 완료. 총 프레임(메타데이터): {actual_total}. (크기: {frame_w_full}x{frame_h_full})")

    # 1. ROI 주변 4배 영역(너비 2배, 높이 2배) 계산
    ctx_w, ctx_h = w * 2, h * 2
    ctx_x = (x + w // 2) - (ctx_w // 2)  # ROI 중심을 기준으로 컨텍스트 영역 계산
    ctx_y = (y + h // 2) - (ctx_h // 2)

    # 2. 컨텍스트 영역이 비디오 프레임 경계를 벗어나지 않도록 좌표 클리핑
    clip_ctx_x1 = max(0, ctx_x)
    clip_ctx_y1 = max(0, ctx_y)
    clip_ctx_x2 = min(frame_w_full, ctx_x + ctx_w)
    clip_ctx_y2 = min(frame_h_full, ctx_y + ctx_h)

    # 3. 최종적으로 '잘린 컨텍스트 영역' 내부에서 원본 ROI의 '상대 좌표' 계산
    roi_rel_x1 = x - clip_ctx_x1
    roi_rel_y1 = y - clip_ctx_y1
    roi_rel_x2 = x + w - clip_ctx_x1
    roi_rel_y2 = y + h - clip_ctx_y1

    # 4. 레이블링 루프를 위한 변수 초기화
    # events: {프레임 인덱스: 레이블 ID} 딕셔너리.
    events = {0: 0}  # 0번 프레임은 기본값 'False'(0)로 시작
    idx = 0         # 현재 사용자가 보고 있는 프레임 인덱스

    current_frame_data = None     # 현재 메모리에 로드된 '원본' 프레임
    current_frame_idx_read = -1   # current_frame_data에 저장된 프레임의 실제 인덱스

    # max_reachable_frame: 성공적으로 읽은 '최대' 프레임 인덱스.
    # (비디오 메타데이터의 총 프레임 수는 부정확할 수 있으므로, 실제 읽은 값을 추적)
    max_reachable_frame = -1

    cv2.namedWindow('Label', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Label', 600, 600)  # GUI 창 기본 크기 설정

    while True:
        disp_frame = None  # GUI 창에 표시될 최종 이미지 (크롭된 컨텍스트 영역)

        try:
            # --- 프레임 로딩 로직 ---
            # 사용자가 요청한 'idx' 프레임을 로드합니다.

            # A. (최적화) 사용자가 다음 프레임(.)을 눌렀고,
            #    현재 로드된 프레임이 바로 이전 프레임인 경우 -> seek 대신 read()
            if idx == current_frame_idx_read + 1:
                ret, frame = cap.read()
                if ret:
                    current_frame_data = frame
                    current_frame_idx_read = idx
                    # 성공적으로 읽었으므로, 최대 도달 가능 프레임 갱신
                    max_reachable_frame = max(max_reachable_frame, idx)
                else:
                    # 비디오의 실제 끝에 도달함
                    print(f"[!] 비디오의 실제 끝에 도달했습니다 (프레임 {idx}).")
                    idx = current_frame_idx_read  # 인덱스를 마지막으로 성공한 위치로 되돌림

            # B. (탐색) 사용자가 프레임을 점프했거나(,, m, /) 이전(,)으로 갔을 경우
            #    -> cap.set()을 사용하여 해당 프레임으로 탐색(seek)
            elif idx != current_frame_idx_read:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    current_frame_data = frame
                    current_frame_idx_read = idx
                    # 성공적으로 읽었으므로, 최대 도달 가능 프레임 갱신
                    max_reachable_frame = max(max_reachable_frame, idx)
                else:
                    # 탐색 실패 (예: 비디오의 실제 끝을 초과하여 탐색 시도)
                    print(f"[!] 프레임 {idx}로 탐색 실패. 마지막 성공 프레임({current_frame_idx_read}) 유지.")
                    idx = current_frame_idx_read  # 인덱스 되돌리기

            # C. (캐시) idx == current_frame_idx_read 인 경우
            #    (즉, 프레임 이동이 없었거나, 이동 실패로 인덱스가 되돌아온 경우)
            #    -> 아무것도 안 함 (current_frame_data 재사용)

            # --- 프레임 크롭 및 표시 ---
            if current_frame_data is not None:
                # '원본' 프레임에서 '컨텍스트 영역'만큼 잘라내어 표시할 프레임 생성
                disp_frame = current_frame_data[clip_ctx_y1:clip_ctx_y2, clip_ctx_x1:clip_ctx_x2].copy()
            else:
                # 비디오가 아예 비어있는 극히 드문 경우
                if max_reachable_frame == -1:
                    print(f"[!] 비디오에서 프레임을 전혀 읽을 수 없습니다.")
                    break  # while 루프 탈출
                raise RuntimeError("current_frame_data가 None입니다.")

        except Exception as e:
            print(f"[!] 프레임 {idx} 로드/크롭 중 오류 발생: {e}")
            # 오류 발생 시, 검은색 에러 화면 생성
            disp_h = clip_ctx_y2 - clip_ctx_y1
            disp_w = clip_ctx_x2 - clip_ctx_x1
            disp_frame = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
            draw_text(disp_frame, "Read Error", (10, disp_h // 2), 0.7, (0, 0, 255), 2)

        # 6. 표시할 프레임(disp_frame)에 원본 ROI 위치를 노란색 사각형으로 그리기
        cv2.rectangle(disp_frame, (roi_rel_x1, roi_rel_y1), (roi_rel_x2, roi_rel_y2),
                      (0, 255, 255), 1)  # (B, G, R) = 노란색, 1px 굵기

        # 7. 화면에 정보 텍스트 표시 (프레임 번호, 레이블, 단축키)
        disp_h, disp_w = disp_frame.shape[:2]

        # 현재 프레임(idx)에 적용되는 레이블이 무엇인지 찾기
        sorted_keys = sorted(events.keys())  # 레이블이 변경된 지점들
        insert_pos = bisect.bisect_right(sorted_keys, idx)  # idx가 어느 구간에 속하는지
        active_key = sorted_keys[insert_pos - 1]  # 현재 구간의 시작 프레임
        active_label = events[active_key]        # 현재 구간의 레이블 ID
        active_label_str = LABEL_MAP.get(active_label, 'N/A')

        # 상단 정보 (프레임 번호, 현재 레이블)
        draw_text(disp_frame, f"Frame {idx}/{actual_total-1} (Meta)", (10, 20), 0.6, (255, 255, 255), 1)
        draw_text(disp_frame, f"Label: {active_label_str}", (10, 45), 0.7, (0, 255, 0), 2)

        # 하단 정보 (단축키 도움말)
        y_pos = disp_h - 95  # 화면 하단에서 95px 위에서부터 텍스트 시작
        draw_text(disp_frame, ". or / : Fwd (1/100)", (10, y_pos), 0.5, (255, 255, 255), 1)
        y_pos += 20
        draw_text(disp_frame, ", or m : Back (1/100)", (10, y_pos), 0.5, (255, 255, 255), 1)
        y_pos += 20
        draw_text(disp_frame, "0,1,2 : Label (False,True,Unk)", (10, y_pos), 0.5, (255, 255, 255), 1)
        y_pos += 20
        draw_text(disp_frame, "d : Delete Label", (10, y_pos), 0.5, (255, 255, 255), 1)
        y_pos += 20
        draw_text(disp_frame, "Esc : Save & Exit", (10, y_pos), 0.5, (255, 255, 255), 1)

        # 최종 이미지를 'Label' 창에 표시
        cv2.imshow('Label', disp_frame)

        # 8. 사용자 키보드 입력 대기 및 처리
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # 'Esc' 키: 저장 및 종료
            break

        # --- 프레임 이동 단축키 ---
        elif key == ord('.'):  # 다음 프레임 (1)
            idx = min(idx + 1, actual_total - 1)
        elif key == ord(','):  # 이전 프레임 (-1)
            idx = max(idx - 1, 0)
        elif key == ord('/'):  # 앞으로 점프 (100)
            idx = min(idx + 100, actual_total - 1)
        elif key == ord('m'):  # 뒤로 점프 (-100)
            idx = max(idx - 100, 0)

        # --- 레이블링 단축키 ---
        elif key in (ord('0'), ord('1'), ord('2')):
            label = int(chr(key))
            events[idx] = label  # 현재 프레임에 레이블 변경 이벤트 기록
            label_str = LABEL_MAP.get(label, 'N/A')
            print(f"[+] 프레임 {idx}에 레이블 [{label_str}] 저장")
            # 레이블 입력 후, 자동으로 다음 프레임으로 이동 (빠른 작업을 위함)
            idx = min(idx + 1, actual_total - 1)

        # --- 레이블 삭제 단축키 ---
        elif key == ord('d'):
            if idx in events and idx != 0:  # 0번 프레임의 기본 레이블은 삭제 불가
                del events[idx]
                print(f"[-] 프레임 {idx}에 지정된 레이블 삭제")
            elif idx == 0:
                print(f"[!] 0번 프레임의 기본 레이블은 삭제할 수 없습니다.")
            else:
                print(f"[!] 프레임 {idx}에 삭제할 레이블이 없습니다.")

    # 9. 종료 처리: 비디오 객체와 GUI 창 닫기
    cap.release()
    cv2.destroyAllWindows()

    # 만약 비디오를 단 한 프레임도 읽지 못했다면, 레이블 파일을 저장하지 않음
    if max_reachable_frame == -1:
        print(f"[!] 비디오를 전혀 읽을 수 없어 레이블 파일을 저장하지 않습니다: {video_path}")
        return

    # 10. 레이블 이벤트(events)를 연속된 세그먼트(segments)로 변환

    # 마지막 프레임 기준: 메타데이터(actual_total)가 아닌, '실제 도달 가능했던 최대 프레임'
    final_end_frame = max_reachable_frame
    print(f"[i] 실제 읽은 마지막 프레임({final_end_frame}) 기준으로 세그먼트 저장 중...")

    frames = sorted(events.keys())  # 레이블이 변경된 프레임 인덱스 목록
    raw, merged = [], []
    for i, start in enumerate(frames):
        lbl = events[start]

        # 세그먼트의 끝 프레임: (다음 이벤트 프레임 - 1)
        # 만약 마지막 이벤트라면, (실제 도달 가능했던 최대 프레임)
        end = frames[i+1] - 1 if i < len(frames) - 1 else final_end_frame

        if end < start:
            # (예: 마지막 이벤트가 500, final_end_frame이 499인 극단적 경우)
            continue

        raw.append({'start_frame': start, 'end_frame': end, 'label': lbl})

    # 동일한 레이블을 가진 연속된 세그먼트 병합
    for seg in raw:
        if merged and seg['label'] == merged[-1]['label'] and seg['start_frame'] <= merged[-1]['end_frame'] + 1:
            # 이전 세그먼트와 이어지므로, 이전 세그먼트의 'end_frame'만 확장
            merged[-1]['end_frame'] = max(merged[-1]['end_frame'], seg['end_frame'])
        else:
            # 새로운 레이블이거나, 프레임이 끊어진 경우 -> 새 세그먼트로 추가
            merged.append(seg.copy())

    # 최종 데이터를 딕셔너리로 정리
    out_data = {
        'video_path': video_path,  # 원본 비디오 경로
        'roi': roi,               # 작업한 ROI 좌표
        'label_map': LABEL_MAP,   # 레이블 정의
        'segments': merged        # 최종 세그먼트 목록
    }

    # JSON 파일로 저장
    with open(out_json, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"[+] {len(merged)}개 세그먼트를 {out_json}에 저장했습니다.")


# --- 메인 실행 로직 ---

if __name__ == '__main__':
    print("--- 인터랙티브 비디오 ROI 레이블링 스크립트 ---")

    # 1. 설정된 데이터 타입('real', 'simulation') 폴더 순회
    for data_type in DATA_TYPES:
        print(f"\n[{data_type}] 데이터 타입 처리 중...")

        # 2. 설정된 CCTV_ROI_CONFIG의 모든 CCTV ID 순회
        for cctv_id, rois in CCTV_ROI_CONFIG.items():
            # 실제 디렉토리 경로 조합
            cctv_dir = os.path.join(BASE_DIR, data_type, cctv_id)

            # 해당 CCTV ID의 디렉토리가 존재하지 않으면 건너뛰기
            if not os.path.isdir(cctv_dir):
                continue

            print(f"  > CCTV ID 스캔: {cctv_id} (설정된 ROI {len(rois)}개)")

            # 3. 해당 디렉토리 내의 모든 비디오 파일 탐색
            video_files = []
            for ext in VIDEO_EXTENSIONS:
                video_files.extend(glob.glob(os.path.join(cctv_dir, ext)))

            if not video_files:
                print(f"    - {cctv_dir} 에서 비디오 파일을 찾지 못했습니다.")
                continue

            # 4. 탐색된 각 비디오 파일 순회
            for video_path in video_files:
                video_name = os.path.basename(video_path)
                base_name = os.path.splitext(video_name)[0]

                # 5. 해당 CCTV에 설정된 모든 ROI 순회
                for roi_index, roi in enumerate(rois):
                    # 출력될 JSON 파일명 결정 (예: video_name_0.json, video_name_1.json)
                    json_name = f"{base_name}_{roi_index}.json"
                    json_path = os.path.join(cctv_dir, json_name)

                    # 6. 만약 결과 JSON 파일이 이미 존재한다면, 작업을 건너뛰기
                    if os.path.exists(json_path):
                        print(f"  [=] 건너뛰기 (이미 레이블링됨): {json_path}")
                        continue

                    # 7. 레이블링 함수 호출
                    print(f"\n  [>] 레이블링 작업 시작:")
                    print(f"    - 비디오: {video_path}")
                    print(f"    - ROI #{roi_index}: {roi}")
                    print(f"    - 출력파일: {json_path}")
                    try:
                        interactive_label_video(video_path, roi, json_path)
                    except Exception as e:
                        print(f"  [!] 레이블링 중 예상치 못한 오류 발생: {e}")
                        # 오류 발생 시, 열려있을 수 있는 OpenCV 창을 강제로 닫음
                        cv2.destroyAllWindows()
                        continue  # 다음 작업으로 넘어감

    print("\n--- 모든 작업이 완료되었습니다 ---")
