import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import json
from model import ViolenceModel

SAVE_RESULT = "./result_yolov8m_2"
DATA_PATH = "../setup/npy_file_yolov8m_combined"
CHECKPOINT_PATH = os.path.join(SAVE_RESULT, "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(SAVE_RESULT, "best_model_violence.pth")
LOSS_CHART_NAME = os.path.join(SAVE_RESULT, "chart_loss.png")
ACC_CHART_NAME = os.path.join(SAVE_RESULT, "chart_accuracy.png")

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
FRAME_LENGTH = 30
MAX_PEOPLE = 5
CLASSES = ['NonViolence', 'Violence']

class ViolenceDataset(Dataset):
    def __init__(self, data_list, labels, phase='train'):
        self.data_list = data_list
        self.labels = labels
        self.phase = phase
        
    def smooth_data(self, data):
        # data shape: (3, T, V, M)
        # làm mượt chuyển động dọc theo trục thời gian (axis=1)
        try:
            # Chỉ làm mượt toạ độ x, y (channel 0, 1), giữ nguyên confidence (channel 2)
            data[0, :, :, :] = savgol_filter(data[0, :, :, :], 5, 2, axis=0) 
            data[1, :, :, :] = savgol_filter(data[1, :, :, :], 5, 2, axis=0)
        except Exception:
            pass
        return data
        
    def augment(self, data):
        # 1. Noise
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, data.shape)
            data = data + noise
            
        # 2. Scale
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            data[:, :, :, :] *= scale
        
        # 3. Time Shift
        if np.random.rand() < 0.3:
            C, T, V, M = data.shape
            shift = np.random.randint(1, 4) # Dịch từ 1 đến 3 frames
            if np.random.rand() < 0.5:
                # Dịch trái (mất frame đầu, lặp frame cuối)
                data[:, :-shift, :, :] = data[:, shift:, :, :]
                for i in range(shift):
                    data[:, -1-i, :, :] = data[:, -1, :, :] # Pad frame cuối
            else:
                # Dịch phải
                data[:, shift:, :, :] = data[:, :-shift, :, :]
                for i in range(shift):
                    data[:, i, :, :] = data[:, 0, :, :] # Pad frame đầu
            
        return data

    def __len__(self):
        return len(self.data_list)
    
    def normalize_data(self, data):
        data_norm = data.copy() # data shape gốc: (3, T, V, M)
        root_x = (data[0, :, 11, :] + data[0, :, 12, :]) / 2
        root_y = (data[1, :, 11, :] + data[1, :, 12, :]) / 2
        
        root_x = np.expand_dims(root_x, axis=1)
        root_y = np.expand_dims(root_y, axis=1) 
        
        data_norm[0, :, :, :] = data[0, :, :, :] - root_x # Trừ tọa độ root để đưa về gốc (0,0) -> Model học chuyển động tương đối
        data_norm[1, :, :, :] = data[1, :, :, :] - root_y

        return data_norm

    def __getitem__(self, idx):
        try:
            data = np.load(self.data_list[idx])
            if data.shape != (3, FRAME_LENGTH, 17, MAX_PEOPLE): 
                return torch.zeros((3, FRAME_LENGTH, 17, MAX_PEOPLE)), torch.tensor(self.labels[idx], dtype=torch.long)
            data = self.normalize_data(data)
            data = self.smooth_data(data)
            if self.phase == 'Train':
                data = self.augment(data)
            return torch.FloatTensor(data), torch.tensor(self.labels[idx], dtype=torch.long)
            
        except Exception as e:
            return torch.zeros((3, FRAME_LENGTH, 17, MAX_PEOPLE)), torch.tensor(self.labels[idx], dtype=torch.long)

def load_data_by_structure(root_dir, phase):
    file_paths = []
    labels = []
    
    class_folders = {
        'NonViolence': 0, 
        'Violence': 1
    }
    
    print(f"--- Đang quét dữ liệu tập {phase.upper()} ---")
    
    for folder_name, label_id in class_folders.items():
        target_path = os.path.join(root_dir, folder_name, phase)
        
        if not os.path.exists(target_path):
            if folder_name == 'NonViolence':
                alt_path = os.path.join(root_dir, 'Normal', phase)
                if os.path.exists(alt_path):
                    target_path = alt_path
            
        if not os.path.exists(target_path):
            print(f"Cảnh báo: Không tìm thấy thư mục {target_path}")
            continue
            
        # Lấy file npy
        files = [f for f in os.listdir(target_path) if f.endswith('.npy')]
        
        for f in files:
            file_paths.append(os.path.join(target_path, f))
            labels.append(label_id)
            
        print(f"   + Lớp '{folder_name}' ({phase}): tìm thấy {len(files)} file.")
        
    return file_paths, labels

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs) # Forward
        loss = criterion(outputs, labels) # Compute Loss
        loss.backward() # Backprop
        optimizer.step() # Update weights
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), 100 * correct / total

def save_plots(history):
    sns.set_theme(style="whitegrid")
    
    # LOSS
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(LOSS_CHART_NAME)
    plt.close()

    # ACCURACY
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Train Acc', color='green')
    plt.plot(history['val_acc'], label='Val Acc', color='red')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(ACC_CHART_NAME)
    plt.close()

def save_report(y_true, y_pred):
    # Confusion Matrix Image
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(SAVE_RESULT, 'confusion_matrix.png'))
    plt.close()
    
    # Text Report
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT:")
    print(report)
    print("="*50)
    
    # Save JSON
    report_dict = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    with open(os.path.join(SAVE_RESULT, 'classification_report.json'), 'w') as f:
        json.dump(report_dict, f, indent=4)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")
    
    if not os.path.exists(SAVE_RESULT):
        os.makedirs(SAVE_RESULT)

    train_files, train_labels = load_data_by_structure(DATA_PATH, "Train")
    test_files, test_labels = load_data_by_structure(DATA_PATH, "Test")
    
    if len(train_files) == 0:
        print(f"LỖI: Không tìm thấy dữ liệu tại {DATA_PATH}. Hãy kiểm tra lại tên thư mục.")
        return

    # Dataset & DataLoader
    train_ds = ViolenceDataset(train_files, train_labels, phase='Train')
    test_ds = ViolenceDataset(test_files, test_labels, phase='Test')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\nTổng mẫu Train: {len(train_ds)}")
    print(f"Tổng mẫu Test: {len(test_ds)}")
    print("\nKhởi tạo mô hình GCN-BiLSTM.")
    model = ViolenceModel(num_classes=2, in_channels=3, t_frames=FRAME_LENGTH, num_person=MAX_PEOPLE).to(device)
    
    # Loss & Optimizer (Giữ nguyên cấu hình tốt nhất)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    patience = 6       # Nếu 8 epochs liên tiếp Val Loss không giảm thì dừng
    counter = 0        # Bộ đếm
    best_val_loss = float('inf')

    # Loop
    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Phát hiện Checkpoint. Đang tải lại...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        best_acc = checkpoint['best_acc']
        print(f"-> Tiếp tục từ Epoch {start_epoch + 1}")

    try:
        for epoch in range(start_epoch, EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}:")
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"-> Đã lưu Best Model! (Val Loss giảm xuống {val_loss:.4f})")
                counter = 0 
            else:
                counter += 1
                print(f"-> Cảnh báo: Val Loss không giảm! ({counter}/{patience})")
                if counter >= patience:
                    print("!!! DỪNG SỚM (Early Stopping) ĐỂ TRÁNH OVERFITTING !!!")
                    break

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'best_acc': best_acc
            }, CHECKPOINT_PATH)
            
            save_plots(history)

    except KeyboardInterrupt:
        print("\nDừng thủ công! Dữ liệu đã lưu trong checkpoint.")

    print("\nHUẤN LUYỆN HOÀN TẤT!")
    
    # Final Eval
    if os.path.exists(BEST_MODEL_PATH):
        print("Đang đánh giá trên Best Model...")
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Final Eval"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        save_report(all_labels, all_preds)

if __name__ == "__main__":
    main()