import gradio as gr
import cv2
import numpy as np
import torch
import os
import sys
import time
from ultralytics import YOLO
from collections import deque, defaultdict
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, './train5/', 'model'))
from model import ViolenceModel

MODEL_PATH = "./train5/model/result_yolov8m/best_model_violence.pth"
YOLO_PATH = './train5/setup/yolov8m-pose.pt'

WINDOW_SIZE = 30
MAX_PEOPLE = 5

# Ngưỡng báo động
MODEL_CONF_THRESHOLD = 0.70
TRIGGER_THRESH_LOW = 0.50

# Ổn định trạng thái
STABILITY_FRAMES = 12
VIOLENCE_COOLDOWN_FRAMES = 45

# Tối ưu
INFERENCE_INTERVAL = 3
SENSITIVITY_INTERVAL = 6
SMOOTH_ALPHA = 0.6
MAX_VIDEO_WIDTH = 1024

COLOR_SAFE = (0, 255, 0)
COLOR_VIOLENCE = (0, 0, 255)
COLOR_WARNING = (0, 255, 255)

print("Đang khởi tạo hệ thống...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViolenceModel(num_classes=2, in_channels=3, t_frames=WINDOW_SIZE, num_person=MAX_PEOPLE).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Đã load GCN Model.")
else:
    print(f"LỖI: Không tìm thấy {MODEL_PATH}")

pose_model = YOLO(YOLO_PATH) 
print("Đã load YOLOv8 Pose.")

def normalize_keypoints(data):
    data_norm = data.copy()
    root_x = (data[0, :, 11, :] + data[0, :, 12, :]) / 2
    root_y = (data[1, :, 11, :] + data[1, :, 12, :]) / 2
    root_x = np.expand_dims(root_x, axis=1) 
    root_y = np.expand_dims(root_y, axis=1)
    data_norm[0, :, :, :] = data[0, :, :, :] - root_x 
    data_norm[1, :, :, :] = data[1, :, :, :] - root_y
    return data_norm

def get_velocity(track_history, tid, width, height):
    if len(track_history[tid]) < 5: return 1.0
    curr_kpt = track_history[tid][-1] 
    prev_kpt = track_history[tid][-5]
    
    curr_cx = (curr_kpt[0, 11] + curr_kpt[0, 12]) / 2
    curr_cy = (curr_kpt[1, 11] + curr_kpt[1, 12]) / 2
    prev_cx = (prev_kpt[0, 11] + prev_kpt[0, 12]) / 2
    prev_cy = (prev_kpt[1, 11] + prev_kpt[1, 12]) / 2
    
    dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
    return dist

def check_interaction_type(boxes_map, active_ids, keypoints_map, width, height):
    if len(active_ids) < 2: return None
    
    attack_indices = [9, 10, 15, 16] 
    body_indices = [0, 5, 6, 11, 12]

    for i in range(len(active_ids)):
        for j in range(i + 1, len(active_ids)):
            id1, id2 = active_ids[i], active_ids[j]
            
            # Check IoU
            if id1 in boxes_map and id2 in boxes_map:
                box1, box2 = boxes_map[id1], boxes_map[id2]
                xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
                xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                if interArea > 0: return 'iou'

            # Check Distance
            if id1 in keypoints_map and id2 in keypoints_map:
                kp1, kp2 = keypoints_map[id1], keypoints_map[id2]
                diag_len = np.sqrt(width**2 + height**2) + 1e-6
                min_dist = 1.0

                pts1 = kp1[attack_indices, :2]; pts2 = kp2[body_indices, :2]
                for p1 in pts1:
                    if p1[0] == 0: continue
                    for p2 in pts2:
                        if p2[0] == 0: continue
                        d = np.linalg.norm(p1 - p2) / diag_len
                        if d < min_dist: min_dist = d
                
                pts1 = kp2[attack_indices, :2]; pts2 = kp1[body_indices, :2]
                for p1 in pts1:
                    if p1[0] == 0: continue
                    for p2 in pts2:
                        if p2[0] == 0: continue
                        d = np.linalg.norm(p1 - p2) / diag_len
                        if d < min_dist: min_dist = d
                
                # Ngưỡng khoảng cách
                if min_dist < 0.20: return 'dist'

    return None

def process_video(input_video_path):
    if input_video_path is None: return None, "No video."

    cap = cv2.VideoCapture(input_video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    scale_factor = 1.0
    if orig_w > MAX_VIDEO_WIDTH:
        scale_factor = MAX_VIDEO_WIDTH / orig_w
        new_w = MAX_VIDEO_WIDTH
        new_h = int(orig_h * scale_factor)
    else:
        new_w, new_h = orig_w, orig_h

    out_path = os.path.join(tempfile.gettempdir(), "result_dynamic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(out_path, fourcc, fps, (new_w, new_h))

    track_history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
    last_positions = {}
    
    consecutive_viol_frames = 0
    cooldown_counter = 0
    max_conf_global = 0.0
    total_frames = 0
    
    cached_prob = 0.0
    cached_culprits = []
    status_message = "SAFE"
    
    prev_time = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_display = 0.9 * fps_display + 0.1 * fps
        
        total_frames += 1
        if scale_factor != 1.0: frame = cv2.resize(frame, (new_w, new_h))

        # Tracking
        results = pose_model.track(frame, persist=True, verbose=False, tracker="botsort.yaml")
        
        boxes_map = {}
        keypoints_map = {}
        current_ids = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.data.cpu().numpy()

            current_ids = track_ids

            for box, tid, kp in zip(boxes, track_ids, keypoints):
                boxes_map[tid] = box
                keypoints_map[tid] = kp 
                
                raw_x = kp[:, 0] / new_w
                raw_y = kp[:, 1] / new_h
                conf = kp[:, 2]
                curr_kpt = np.stack([raw_x, raw_y, conf], axis=0)
                
                if tid in last_positions:
                    last_kpt = last_positions[tid]
                    curr_kpt[0] = SMOOTH_ALPHA * curr_kpt[0] + (1 - SMOOTH_ALPHA) * last_kpt[0]
                    curr_kpt[1] = SMOOTH_ALPHA * curr_kpt[1] + (1 - SMOOTH_ALPHA) * last_kpt[1]
                
                last_positions[tid] = curr_kpt.copy()
                track_history[tid].append(curr_kpt)

        # Inference
        active_ids = [tid for tid in current_ids if len(track_history[tid]) == WINDOW_SIZE]
        selected_ids = active_ids[:MAX_PEOPLE]
        
        should_infer = (total_frames % INFERENCE_INTERVAL == 0) and (len(selected_ids) > 0)
        
        if should_infer:
            input_clip = np.zeros((3, WINDOW_SIZE, 17, MAX_PEOPLE))
            max_velocity = 0.0
            
            for i, tid in enumerate(selected_ids):
                input_clip[:, :, :, i] = np.array(track_history[tid]).transpose(1, 0, 2)
                vel = get_velocity(track_history, tid, new_w, new_h)
                if vel > max_velocity: max_velocity = vel
            
            input_clip = normalize_keypoints(input_clip)
            input_tensor = torch.FloatTensor(input_clip).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out_gcn = model(input_tensor)
                
                # Temperature Scaling
                temperature = 0.70 
                raw_prob = torch.softmax(out_gcn / temperature, dim=1)[0, 1].item()
                
                contact_type = check_interaction_type(boxes_map, selected_ids, keypoints_map, new_w, new_h)
                
                # Ngưỡng vận tốc
                ACTIVE_MOTION_THRESHOLD = 0.02
                if contact_type == 'iou': 
                    # Có chồng lấn (Overlap)
                    if max_velocity > ACTIVE_MOTION_THRESHOLD:
                        # Chồng lấn + Di chuyển nhanh -> Đánh nhau thật
                        final_prob = min(1.0, raw_prob * 1.05) 
                        status_message = "Fighting (High Vel)"
                        required_movement = 0.02
                    else:
                        # Chồng lấn nhưng Đứng Yên -> Chen chúc
                        final_prob = raw_prob * 0.4 
                        status_message = "Crowding (Static)"
                        required_movement = 0.02

                elif contact_type == 'dist': # Gần nhau
                    final_prob = raw_prob * 1.0 
                    status_message = "Close Range"
                    required_movement = 0.002 

                else: # Xa nhau
                    if raw_prob > 0.95: 
                        final_prob = raw_prob
                        status_message = "High Conf (Far)"
                    else:
                        final_prob = raw_prob * 0.7 
                        status_message = "No Contact"
                    required_movement = 0.002
                
                # Velocity check
                if max_velocity < required_movement:
                    final_prob = final_prob * 0.1
                    status_message += " (Static)"
                
                cached_prob = final_prob
                if final_prob > max_conf_global: max_conf_global = final_prob

                # Trigger
                trigger = TRIGGER_THRESH_LOW if cooldown_counter > 0 else MODEL_CONF_THRESHOLD
                
                if final_prob > trigger:
                    consecutive_viol_frames += 1
                    cooldown_counter = VIOLENCE_COOLDOWN_FRAMES
                else:
                    consecutive_viol_frames = max(0, consecutive_viol_frames - 1)
                    if cooldown_counter > 0: cooldown_counter -= 1

                # Culprit Logic
                is_active = (consecutive_viol_frames >= STABILITY_FRAMES) or (cooldown_counter > 10)
                if is_active and (total_frames % SENSITIVITY_INTERVAL == 0):
                    temp_culprits = []
                    for i, tid in enumerate(selected_ids):
                        masked = input_clip.copy(); masked[:, :, :, i] = 0 
                        t_masked = torch.FloatTensor(masked).unsqueeze(0).to(device)
                        p_masked = torch.softmax(model(t_masked), dim=1)[0, 1].item()
                        if (raw_prob - p_masked) > 0.20:
                            temp_culprits.append(tid)
                    cached_culprits = temp_culprits

        # Drawing
        is_viol = (consecutive_viol_frames >= STABILITY_FRAMES) or (cooldown_counter > 0)
        
        for tid, box in boxes_map.items():
            x1, y1, x2, y2 = map(int, box)
            
            # Tính vận tốc riêng
            indiv_vel = get_velocity(track_history, tid, new_w, new_h) if len(track_history[tid]) >= 5 else 0.0
            
            color = COLOR_SAFE
            label = ""
            thick = 2

            if len(track_history[tid]) < WINDOW_SIZE:
                color = (150, 150, 150)
                thick = 1
            
            elif is_viol:
                # Logic vẽ: Chỉ tô đỏ nếu là tác nhân gây bạo lực HOẶC di chuyển nhanh
                is_active_participant = (indiv_vel > 0.01) 
                
                if (tid in cached_culprits) and is_active_participant:
                    color = COLOR_VIOLENCE
                    label = f"VIOLENCE"
                    thick = 4
                else:
                    color = COLOR_WARNING
                    label = f"WARNING"
                    thick = 2
            else:
                color = COLOR_SAFE

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            if label: 
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        font_scale = max(0.5, new_w / 1500)
        thickness = max(1, int(font_scale * 2))
        
        fps_text = f"FPS: {int(fps_display)}"
        (text_w, text_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        cv2.putText(frame, fps_text, (new_w - text_w - 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        out.write(frame)

    cap.release()
    out.release()
    
    verdict = "Violence" if max_conf_global > MODEL_CONF_THRESHOLD else "Non Violence"
    report = f"Result: {verdict}\nProbability: {max_conf_global * 100:.1f}%"
    return out_path, report

# UI
custom_css = """
#video_out { height: 500px; }

label span {
    font-size: 24px !important;
    font-weight: bold !important;
}

textarea {
    font-size: 24px !important;
    line-height: 1.5 !important;
    font-family: 'Arial', sans-serif;
}

button {
    font-size: 20px !important;
}
"""
with gr.Blocks(css=custom_css, title="Violence Detection") as demo:
    gr.Markdown("""
        <h1 style='text-align: center; font-size: 48px; margin-bottom: 20px;'>
            IDENTIFYING SCHOOL VIOLENCE BEHAVIOR
        </h1>
    """)
    
    with gr.Row():
        inp = gr.Video(label="Input Source", height=300)
        out_vid = gr.Video(label="Processing Result", elem_id="video_out", interactive=False)
    
    btn = gr.Button("ANALYSIS", variant="primary")
    out_txt = gr.Textbox(label="Details:", lines=2, max_lines=10)
    
    btn.click(fn=process_video, inputs=inp, outputs=[out_vid, out_txt])

if __name__ == "__main__":
    demo.launch(share=True)