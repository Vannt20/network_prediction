import torch
import torch.nn as nn
import torch.nn.functional as F


class PerFlowContextualMetaGating(nn.Module):
    """
    Cổng nơ-ron điều phối động độc lập cho từng luồng OD (Per-Flow Contextual Meta-Gating):
    - Kích thước đầu ra: [B, N, 3] tương ứng với 3 trọng số [w_global, w_local, w_ml].
    - Đảm bảo tính lồi: sum(weights, dim=-1) = 1.0 cho từng luồng.
    - Đầu vào ngữ cảnh: [local_volatility, spike_flag, tod, dow] (context_dim=4).
    - Khởi tạo: Empirical Loss Prior hoặc Neutral (bias = 0.0), loại bỏ bias cứng +1.5.
    """
    def __init__(self, context_dim=4, hidden_dim=32, dropout=0.05, init_prior=None):
        super(PerFlowContextualMetaGating, self).__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3) # 3 logits: Global, Local, ML
        )

        # Khởi tạo trọng số
        self.init_weights(init_prior)

    def init_weights(self, init_prior=None):
        # Khởi tạo lớp tuyến tính đầu
        nn.init.xavier_uniform_(self.gate_net[0].weight)
        nn.init.zeros_(self.gate_net[0].bias)

        # Khởi tạo lớp tuyến tính cuối
        nn.init.xavier_uniform_(self.gate_net[3].weight)
        if init_prior is not None:
            # init_prior: tensor [3] = -ln(val_mse_k + eps)
            with torch.no_grad():
                if isinstance(init_prior, torch.Tensor):
                    self.gate_net[3].bias.copy_(init_prior)
                else:
                    self.gate_net[3].bias.copy_(torch.tensor(init_prior, dtype=torch.float32))
        else:
            # Khởi tạo trung tính hoàn toàn khách quan (bias = 0.0)
            nn.init.zeros_(self.gate_net[3].bias)

    def forward(self, context_features):
        """
        context_features: [B, N, context_dim]
        Trả về:
            weights: [B, N, 3] với w_global + w_local + w_ml = 1.0 per flow
        """
        logits = self.gate_net(context_features)  # [B, N, 3]
        weights = F.softmax(logits, dim=-1)       # [B, N, 3]
        return weights
