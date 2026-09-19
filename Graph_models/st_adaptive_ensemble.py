import torch
import torch.nn as nn
from features.temporal_features import extract_context_features_torch


class STAdaptiveEnsemble(nn.Module):
    """
    KIẾN TRÚC MÔ HÌNH HỌC KẾT HỢP THÍCH ỨNG: ST-ADAPTIVE-ENSEMBLE
    Kết hợp 3 nhánh chuyên biệt với cơ chế Scale Alignment và Per-Flow Meta-Gating:
    
    1. Nhánh Vĩ mô (Global): ST-WaveFormer (Transformer + Dynamic GCN + RevIN)
    2. Nhánh Vi mô (Local): SpatialDilatedTCN (Spatial-aware TCN có ma trận kề A + RevIN)
    3. Nhánh Đột biến (ML): Champion GBDT (bắt micro-bursts trên SDN trong không gian MinMax [0, 1])
    4. Cổng điều phối động (Contextual Meta-Gating): [B, N, 3] cho từng luồng OD
    """
    def __init__(self, global_branch, local_branch, ml_branch, gate):
        super(STAdaptiveEnsemble, self).__init__()
        self.global_branch = global_branch
        self.local_branch = local_branch
        self.ml_branch = ml_branch
        self.gate = gate

    def forward(self, x, context_features=None):
        """
        x: [B, T, N, 3] hoặc [B, T, N]
        context_features: [B, N, 4] (nếu None sẽ tự động trích xuất từ x)
        
        Trả về:
            y_hat: [B, N] dự báo cuối cùng trong không gian MinMax [0, 1]
            weights: [B, N, 3] ma trận trọng số cổng mềm theo từng luồng
        """
        if context_features is None:
            context_features = extract_context_features_torch(x)

        # B1. Lấy dự đoán từ 3 nhánh (Tất cả đã chuẩn hóa về cùng không gian MinMax [0, 1])
        y_global = self.global_branch(x)              # [B, N]
        y_local  = self.local_branch(x)               # [B, N]
        if self.ml_branch is not None:
            y_ml = self.ml_branch(x).detach()         # [B, N] (chặn gradient sang cây)
        else:
            y_ml = torch.zeros_like(y_global)

        # B2. Tính trọng số thích ứng độc lập cho từng luồng OD
        weights = self.gate(context_features)         # [B, N, 3]

        # B3. Phép cộng lồi có trọng số (Convex Combination per flow)
        y_hat = (weights[:, :, 0] * y_global +
                 weights[:, :, 1] * y_local +
                 weights[:, :, 2] * y_ml)             # [B, N]

        return y_hat, weights

    def predict_components(self, x, context_features=None):
        """
        Trả về chi tiết dự báo của từng nhánh riêng lẻ và trọng số cổng:
        Phục vụ cho việc trực quan hóa động học cổng và phân tích thực nghiệm.
        """
        if context_features is None:
            context_features = extract_context_features_torch(x)

        y_global = self.global_branch(x)
        y_local = self.local_branch(x)
        y_ml = self.ml_branch(x).detach() if self.ml_branch is not None else torch.zeros_like(y_global)
        weights = self.gate(context_features)

        y_hat = (weights[:, :, 0] * y_global +
                 weights[:, :, 1] * y_local +
                 weights[:, :, 2] * y_ml)

        return {
            'y_hat': y_hat,
            'y_global': y_global,
            'y_local': y_local,
            'y_ml': y_ml,
            'weights': weights
        }
