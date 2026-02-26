import numpy as np
import cv2
import os

EDGE_LINKS = [
    (0, 1), (0, 2), (1, 3), (2, 4), 
    (5, 6), (5, 7), (7, 9), 
    (6, 8), (8, 10), (11, 12), 
    (5, 11), (6, 12), (11, 13), 
    (13, 15), (12, 14), (14, 16)
]

COLORS_PALETTE = [
    (0, 255, 0),    # Người 1: Xanh lá
    (255, 0, 0),    # Người 2: Xanh dương
    (0, 255, 255),  # Người 3: Vàng
    (255, 0, 255),  # Người 4: Tím
    (0, 165, 255),  # Người 5: Cam
    (255, 255, 255)
]
COLOR_JOINT = (0, 0, 255)
CANVAS_SIZE = (640, 640)    

def save_npy_to_video(npy_path, output_video_path, fps=30):
    if not os.path.exists(npy_path):
        print(f"Lỗi: Không tìm thấy file {npy_path}")
        return

    try:
        data = np.load(npy_path)
    except Exception as e:
        print(f"Lỗi đọc file npy: {e}")
        return

    C, T, V, M = data.shape
    print(f"File: {npy_path} | Shape: {data.shape}")
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        out = cv2.VideoWriter(output_video_path, fourcc, fps, CANVAS_SIZE)
    except:
        print("Cảnh báo: Codec avc1 lỗi, chuyển sang mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, CANVAS_SIZE)
    
    for t in range(T):
        canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    
        # Ghi số frame
        cv2.putText(canvas, f"Frame: {t}/{T}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Duyệt qua từng người
        for m in range(M):
            conf_scores = data[2, t, :, m]
            if np.sum(conf_scores) == 0: continue

            # Chọn màu xương theo ID người
            bone_color = COLORS_PALETTE[m % len(COLORS_PALETTE)]

            xs = data[0, t, :, m] * CANVAS_SIZE[0]
            ys = data[1, t, :, m] * CANVAS_SIZE[1]

            # Vẽ xương trước
            for (idx1, idx2) in EDGE_LINKS:
                if conf_scores[idx1] > 0.3 and conf_scores[idx2] > 0.3:
                    p1 = (int(xs[idx1]), int(ys[idx1]))
                    p2 = (int(xs[idx2]), int(ys[idx2]))
                    cv2.line(canvas, p1, p2, bone_color, 2)
            
            # Vẽ khớp sau
            for i in range(V):
                if conf_scores[i] > 0.3:
                    cv2.circle(canvas, (int(xs[i]), int(ys[i])), 4, COLOR_JOINT, -1)
            
            # Ghi ID người lên đầu
            avg_conf = np.mean(conf_scores)
            cv2.putText(canvas, f"ID:{m} ({avg_conf:.2f})", (int(xs[0]), int(ys[0])-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bone_color, 1)

        out.write(canvas)

    out.release()
    print(f"-> Đã xuất video: {output_video_path}")

if __name__ == "__main__":
    input_npy = "./npy_file_yolov8m_combined/NonViolence/Train/02_014_clip7.npy"
    output_vid = "npy_02_014_clip7.mp4"
    save_npy_to_video(input_npy, output_vid)