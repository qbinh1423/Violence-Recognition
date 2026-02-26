import numpy as np
import os

# Đường dẫn đến 1 file npy bất kỳ bạn vừa tạo
NPY_PATH = "./npy_file_yolov8m_combined/NonViolence/Train/02_014_clip0.npy" 

def inspect_npy(path):
    if not os.path.exists(path):
        print("File không tồn tại!")
        return

    data = np.load(path)
    print(f"--- THÔNG TIN FILE: {os.path.basename(path)} ---")
    print(f"1. Shape (Kích thước): {data.shape}")
    print("   -> Mong đợi: (Frames, Persons, 17, 3)")
    print("   -> Ví dụ: (100, 2, 17, 3)")
    
    print(f"\n2. Kiểm tra dữ liệu:")
    # Lấy frame đầu tiên, người đầu tiên
    first_frame_person = data[0, 0, :, :] 
    print(f"   Dữ liệu mẫu (Frame 0, Person 0):\n{first_frame_person[:3]}") # In 3 khớp đầu
    
    # Kiểm tra chuẩn hóa (phải nằm trong khoảng 0-1)
    max_val = np.max(data[:, :, :, :2]) # Chỉ check x,y
    min_val = np.min(data[:, :, :, :2])
    
    print(f"\n3. Kiểm tra chuẩn hóa (Normalization):")
    print(f"   Max value (x,y): {max_val} (Phải <= 1.0)")
    print(f"   Min value (x,y): {min_val} (Phải >= 0.0)")
    
    if max_val > 1.0:
        print("   [CẢNH BÁO] Dữ liệu chưa được chuẩn hóa! (Lớn hơn 1)")
    elif max_val == 0.0:
        print("   [CẢNH BÁO] Dữ liệu toàn số 0 (Mô hình không nhìn thấy ai)!")
    else:
        print("   [OK] Dữ liệu có vẻ đã được chuẩn hóa đúng.")

# Chạy thử
inspect_npy(NPY_PATH)