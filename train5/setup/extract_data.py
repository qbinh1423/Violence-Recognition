import os
import cv2
import numpy as np
import pandas as pd
import random
from ultralytics import YOLO
from tqdm import tqdm

INPUT_ROOT = "../../dataset/Real Life Violence Situations Dataset" 
OUTPUT_ROOT = "npy_file_yolov8m_RLVSD"

CONF_THRESHOLD = 0.25
WINDOW_SIZE = 30
STRIDE = 30
MAX_PEOPLE = 5
TRAIN_RATIO = 0.8 # 80% Train, 20% Test

print("Đang tải model YOLOv8-Pose...")
model = YOLO('yolov8m-pose.pt')

def interpolate_data(data):
    T, V, C = data.shape
    data = data.reshape(T, -1) 
    df = pd.DataFrame(data)
    df = df.replace(0, np.nan)
    df = df.interpolate(method='linear', limit=3, limit_direction='both')
    df = df.fillna(0)
    return df.values.reshape(T, V, C)

def process_full_video(video_path, save_path_base, window_size, stride):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Track toàn bộ video
    results = model.track(source=video_path, persist=True, verbose=False, tracker="botsort.yaml", stream=True)
    
    all_tracks = {} 
    frame_idx = 0
    
    for r in results:
        if r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            kpts = r.keypoints.data.cpu().numpy()
            
            for i, track_id in enumerate(ids):
                if track_id not in all_tracks:
                    all_tracks[track_id] = np.zeros((total_frames_video, 17, 3))
                
                if frame_idx < total_frames_video:
                    current_kp = kpts[i].copy()
                    current_kp[:, 0] /= width
                    current_kp[:, 1] /= height
                    all_tracks[track_id][frame_idx] = current_kp
        
        frame_idx += 1
        if frame_idx >= total_frames_video: break
    
    cap.release()
    
    if len(all_tracks) == 0: return

    # Lọc Top-K
    track_lengths = []
    for tid, data in all_tracks.items():
        valid_frames = np.sum(np.mean(data[:, :, 2], axis=1) > 0.1)
        track_lengths.append((tid, valid_frames, data))
    
    track_lengths.sort(key=lambda x: x[1], reverse=True)
    top_tracks = track_lengths[:MAX_PEOPLE]
    
    processed_tracks = []
    for tid, _, data in top_tracks:
        xy = data[:, :, :2]
        conf = data[:, :, 2:]
        xy_interp = interpolate_data(xy)
        data_processed = np.concatenate([xy_interp, conf], axis=2)
        processed_tracks.append(data_processed)
        
    while len(processed_tracks) < MAX_PEOPLE:
        processed_tracks.append(np.zeros((total_frames_video, 17, 3)))
        
    final_data = np.stack(processed_tracks, axis=0)
    final_data = final_data.transpose(1, 2, 3, 0)
    
    # Cắt clip
    clip_idx = 0
    if total_frames_video < window_size:
        pad_len = window_size - total_frames_video
        padding = np.repeat(final_data[[-1]], pad_len, axis=0)
        final_data = np.concatenate([final_data, padding], axis=0)
        total_frames_video = window_size

    for i in range(0, total_frames_video - window_size + 1, stride):
        clip_data = final_data[i : i + window_size]
        clip_out = clip_data.transpose(2, 0, 1, 3) 
        
        save_name = f"{save_path_base}_clip{clip_idx}.npy"
        np.save(save_name, clip_out)
        clip_idx += 1

def main():
    categories = ['Violence', 'NonViolence']
    
    for category in categories:
        input_dir = os.path.join(INPUT_ROOT, category)
        
        if not os.path.exists(input_dir):
            print(f"Không tìm thấy thư mục: {input_dir}")
            continue
            
        # Lấy danh sách video
        all_videos = [f for f in os.listdir(input_dir) if f.lower().endswith(('.avi', '.mp4', '.mkv', '.mov'))]
        
        # Trộn và chia Train/Test
        random.shuffle(all_videos)
        split_idx = int(len(all_videos) * TRAIN_RATIO)
        
        train_videos = all_videos[:split_idx]
        test_videos = all_videos[split_idx:]
        
        print(f"\nĐang xử lý lớp: {category}")
        print(f"   Tổng: {len(all_videos)} | Train: {len(train_videos)} | Test: {len(test_videos)}")
        
        datasets = {
            'Train': train_videos,
            'Test': test_videos
        }
        
        for phase, videos in datasets.items():
            output_dir = os.path.join(OUTPUT_ROOT, phase, category)
            os.makedirs(output_dir, exist_ok=True)
            
            for video_file in tqdm(videos, desc=f"{phase}/{category}"):
                video_path = os.path.join(input_dir, video_file)
                file_name = os.path.splitext(video_file)[0]
                save_base = os.path.join(output_dir, file_name)
                
                # Resume logic
                if os.path.exists(f"{save_base}_clip0.npy"):
                    continue
                
                try:
                    process_full_video(video_path, save_base, WINDOW_SIZE, STRIDE)
                except Exception as e:
                    print(f"Lỗi {video_file}: {e}")

if __name__ == "__main__":
    main()