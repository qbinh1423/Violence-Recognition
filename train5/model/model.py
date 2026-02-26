import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Graph():
    def __init__(self, strategy='spatial'):
        self.get_edge()
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge)
        self.get_adjacency(strategy)

    def get_edge(self):
        self.num_node = 17
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            (0, 1), (0, 2), (1, 3), (2, 4),      # Mắt - Tai
            (5, 6), (5, 7), (7, 9),              # Vai - Khuỷu - Cổ tay (Trái)
            (6, 8), (8, 10),                     # Vai - Khuỷu - Cổ tay (Phải)
            (11, 12),                            # Hông trái - Hông phải
            (5, 11), (6, 12),                    # Vai - Hông
            (11, 13), (13, 15),                  # Hông - Gối - Chân (Trái)
            (12, 14), (14, 16)                   # Hông - Gối - Chân (Phải)
        ]
        self.edge = self_link + neighbor_link
        self.center = 0 # Chọn ngực làm tâm

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        return A # Trả về ma trận kề nhị phân (0 hoặc 1)

    def get_adjacency(self, strategy): # Normalize Adjacency Matrix
        valid_hop = range(0, 2)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        
        normalize_adjacency = self.normalize_digraph(adjacency)
        self.A = torch.tensor(normalize_adjacency, dtype=torch.float32).unsqueeze(0)

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(SpatialGraphConv, self).__init__()
        self.kernel_size = kernel_size
       
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=1) # Conv2d này đóng vai trò nhân trọng số W
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        x = self.conv(x) 
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A)) 
        
        x = self.bn(x)
        x = self.relu(x)
        return x

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, 1)

    def forward(self, rnn_output):
        energy = self.attn(rnn_output)
        weights = F.softmax(energy, dim=1)  # Chuyển thành xác suất (Weights) bằng Softmax: (Batch, Time, 1)
        context = torch.sum(rnn_output * weights, dim=1)
        
        return context, weights

# GCN-BiLSTM
class ViolenceModel(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, t_frames=30, v_nodes=17, num_person=5):
        super(ViolenceModel, self).__init__()
        self.graph = Graph()
       
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False) # Lấy ma trận gốc từ Graph
        self.register_buffer('A', A)
        
        # Tạo một ma trận tham số có thể học (Learnable Parameter)
        # Khởi tạo giá trị rất nhỏ (1e-6) để ban đầu nó giống ma trận gốc
        self.PA = nn.Parameter(torch.zeros_like(A)) 
        nn.init.constant_(self.PA, 1e-6)
        # -------------------------------

        self.gcn1 = SpatialGraphConv(in_channels, 32)
        self.gcn2 = SpatialGraphConv(32, 64)
        self.gcn3 = SpatialGraphConv(64, 128)
        
        self.pool = nn.AdaptiveAvgPool2d((t_frames, 1)) 
        
        # Temporal LSTM
        self.lstm_input_size = 128 
        self.hidden_size = 128
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=1, 
                            batch_first=True, 
                            dropout=0.0,
                            bidirectional=True)
        
        self.attention = Attention(self.hidden_size * 2)
        
        self.fc1 = nn.Linear(self.hidden_size * 2, 64)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
            N, C, T, V, M = x.size()
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
            
            # Kết hợp Graph tĩnh và động 
            # A_final = A_fixed + A_learnable
            A_adaptive = self.A + self.PA 
            
            # GCN
            x = self.gcn1(x, A_adaptive)
            x = self.gcn2(x, A_adaptive)
            x = self.gcn3(x, A_adaptive) 
            
            # Pooling
            x = x.mean(dim=3) 
            x = x.view(N, M, -1, T) # Shape tự động tính
            x, _ = torch.max(x, dim=1) 
            x = x.permute(0, 2, 1) 
            
            # LSTM
            self.lstm.flatten_parameters()
            x, (hn, cn) = self.lstm(x) 
            x, attn_weights = self.attention(x) 
            
            # Classifier
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x