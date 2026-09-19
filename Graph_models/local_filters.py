import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from Graph_models.st_waveformer import RevIN
except ImportError:
    from st_waveformer import RevIN


class SpatialDilatedTCN(nn.Module):
    """
    Nhánh Cục bộ nâng cấp: Spatial-aware Dilated TCN.
    Tối ưu hóa toàn diện theo 4 nguyên lý khoa học:
    1. Trích xuất đặc trưng thời gian tuần hoàn liên tục (Continuous Temporal Features):
       - Tích hợp sin/cos(Time-of-day) và sin/cos(Day-of-week) triệt tiêu gián đoạn biên.
    2. Bộ lọc tích chập giãn nở đa tỷ lệ (Multi-Scale Dilated TCN d=1, 2, 4) trích xuất vi động lực ngắn hạn.
    3. Trao đổi không gian Topology-Masked Adaptive Adjacency:
       - Ma trận kề thích ứng học được được ràng buộc chặt chẽ bởi Topology vật lý cục bộ (Top-k sparse),
         ngăn chặn triệt để hiện tượng rò rỉ liên kết toàn cục (Global Bleed / Over-smoothing).
    4. Cơ chế Residual Last-Value Skip Connection:
       - Dự đoán độ lệch Delta y so với bước thời gian trước x_{t-1}, giúp mô hình hội tụ nhanh và ổn định.
       - Tầng dự báo Delta là Linear thuần túy (không bọc ReLU) để hỗ trợ cả biến động tăng và giảm.
    5. Tích hợp RevIN (Reversible Instance Normalization):
       - Chuẩn hóa chống trôi dạt phân phối (covariate shift) và giải chuẩn hóa đồng bộ.
    """
    def __init__(self, num_nodes: int, hidden_dim=64, emb_dim=10, dropout=0.1, adj_mx=None):
        super(SpatialDilatedTCN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 1. RevIN Normalization layer
        self.revin = RevIN(num_features=num_nodes, affine=True)

        # 2. Input Projection: 1 traffic channel + 4 temporal channels (tod_sin, tod_cos, dow_sin, dow_cos) = 5 channels
        self.in_proj = nn.Conv1d(5, hidden_dim, kernel_size=1)

        # 3. Multi-scale Dilated Temporal Convolutions (d=1, 2, 4)
        self.conv_d1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.conv_d4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)

        # 4. Spatial Graph Interaction (Topology-Masked Awareness)
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        
        if adj_mx is not None:
            self.register_buffer('static_adj', torch.as_tensor(adj_mx, dtype=torch.float32))
        else:
            self.static_adj = None

        self.spatial_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 5. Output Delta Projection Head (Linear thuần túy ở lớp cuối - KHÔNG bọc ReLU)
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        # Zero-Init cho tầng cuối: tại Epoch 0, delta = 0 giúp mô hình bắt đầu ngay
        # từ baseline tự thân hoàn hảo (Persistence) mà không bị nhiễu ngẫu nhiên
        nn.init.zeros_(self.out_head[-1].weight)
        nn.init.zeros_(self.out_head[-1].bias)

    def forward(self, x, return_delta: bool = False):
        """return_delta=True trả thêm mean(|delta_norm|) - chỉ báo sống/chết của nhánh.

        Nếu đại lượng này ở mức ~1e-8 thì delta ~ 0 và đầu ra chính là persistence
        (lỗi B1): mạng chưa học được gì, early stopping đang giữ lại trạng thái
        khởi tạo zero-init.
        """
        # x: [B, T, N, C] hoặc [B, T, N]
        if x.dim() == 4:
            x_val = x[:, :, :, 0] # [B, T, N]
            has_time = (x.size(-1) >= 3)
        else:
            x_val = x
            has_time = False

        B, T, N = x_val.shape

        # B1. Chuẩn hóa chống trôi dạt RevIN cho kênh lưu lượng
        x_norm = self.revin(x_val, mode='norm') # [B, T, N]

        # B2. Chuẩn bị đặc trưng thời gian (TOD/DOW)
        if has_time:
            tod = x[:, :, :, 1] # [B, T, N]
            dow = x[:, :, :, 2] # [B, T, N]
            tod_sin = torch.sin(2.0 * math.pi * tod)
            tod_cos = torch.cos(2.0 * math.pi * tod)
            dow_sin = torch.sin(2.0 * math.pi * dow)
            dow_cos = torch.cos(2.0 * math.pi * dow)
            x_feat = torch.stack([x_norm, tod_sin, tod_cos, dow_sin, dow_cos], dim=-1) # [B, T, N, 5]
        else:
            zeros = torch.zeros(B, T, N, 4, device=x.device, dtype=x.dtype)
            x_feat = torch.cat([x_norm.unsqueeze(-1), zeros], dim=-1) # [B, T, N, 5]

        # B3. Chiếu đầu vào và Multi-scale Dilated Convolutions
        # Chuyển đổi thành [B * N, 5, T] để trích xuất vi biến động độc lập cho từng luồng
        x_in = x_feat.permute(0, 2, 3, 1).contiguous().view(B * N, 5, T)
        h_in = F.gelu(self.in_proj(x_in)) # [B * N, H, T]

        c1 = F.gelu(self.conv_d1(h_in))   # [B * N, H, T]
        c2 = F.gelu(self.conv_d2(h_in))   # [B * N, H, T]
        c4 = F.gelu(self.conv_d4(h_in))   # [B * N, H, T]

        c_cat = torch.cat([c1, c2, c4], dim=1) # [B * N, 3*H, T]
        c_last = c_cat[:, :, -1]               # [B * N, 3*H]
        c_feat = c_last.view(B, N, 3 * self.hidden_dim) # [B, N, 3*H]

        # B4. Spatial Graph Interaction (Topology-Masked Adaptive Adjacency)
        A_adapt = F.softmax(F.relu(torch.mm(self.node_emb1, self.node_emb2.T)), dim=-1) # [N, N]
        if self.static_adj is not None:
            # Topology Masking: Ràng buộc thích ứng trong phạm vi lân cận vật lý Top-k
            topo_mask = (self.static_adj > 0).float()
            A_adapt_masked = A_adapt * topo_mask
            A_adapt_norm = A_adapt_masked / (A_adapt_masked.sum(dim=-1, keepdim=True) + 1e-8)
            A = 0.5 * A_adapt_norm + 0.5 * self.static_adj
        else:
            A = A_adapt

        # Lan truyền không gian cục bộ: [B, N, 3*H] -> [B, N, H]
        h_local = self.spatial_proj(c_feat) # [B, N, H]
        h_spatial = torch.matmul(A, h_local) # [B, N, H]
        h_combined = h_local + h_spatial     # Residual connection

        # B5. Dự đoán độ lệch Delta y (Linear head)
        delta_norm = self.out_head(h_combined).squeeze(-1) # [B, N]

        # B6. Residual Last-Value Skip Connection: y_{t} = x_{t-1} + Delta y
        last_val = x_norm[:, -1, :] # [B, N]
        out_norm = last_val + delta_norm # [B, N]

        # B7. Giải chuẩn hóa RevIN khôi phục về thang đo MinMax [0, 1]
        out = self.revin(out_norm, mode='denorm')
        if return_delta:
            return out, delta_norm.abs().mean().detach()
        return out


def load_local_branch(ds_key: str, seq_len: int, run_id: int, num_flows: int, adj_flow=None, logs_dir='logs', device='cpu'):
    """
    Nạp Nhánh Cục bộ (Local Branch - LocalSpatialTCN / SpatialDilatedTCN) đã huấn luyện hoàn chỉnh,
    kết hợp ma trận kề vật lý cấp độ luồng adj_flow.
    """
    import os
    candidate_paths = [
        os.path.join(logs_dir, f"local_spatial_tcn_data_{ds_key}_seq_{seq_len}", f"run_{run_id}", 'best_model.pth'),
        os.path.join(logs_dir, f"localspatialtcn_data_{ds_key}_seq_{seq_len}", f"run_{run_id}", 'best_model.pth'),
    ]
    for ckpt in candidate_paths:
        if os.path.exists(ckpt):
            model = SpatialDilatedTCN(num_nodes=num_flows, hidden_dim=64, adj_mx=adj_flow)
            with open(ckpt, 'rb') as f:
                model.load_state_dict(torch.load(f, map_location=device))
            model.to(device)
            model.eval()
            print(f"      [Local] Đã nạp checkpoint LocalSpatialTCN độc lập: {ckpt}", flush=True)
            return model, "LocalSpatialTCN"

    raise FileNotFoundError(
        f"Không tìm thấy checkpoint đã huấn luyện cho nhánh Local (LocalSpatialTCN) tại:\n"
        f" - {candidate_paths[0]}\n"
        f" - {candidate_paths[1]}\n"
        f"Vui lòng chạy huấn luyện LocalSpatialTCN trước khi tạo cache!"
    )

