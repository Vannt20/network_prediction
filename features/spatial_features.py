import re
import numpy as np
import scipy.sparse as sp


def parse_od_columns(columns):
    """
    Phân tích danh sách tên cột để trích xuất cặp (src, dst) cho từng luồng OD.
    Định dạng mong đợi: 'OD_{src}-{dst}' hoặc tương đương.
    """
    od_pairs = []
    pattern = re.compile(r"OD_([A-Za-z0-9]+)[-_]([A-Za-z0-9]+)", re.IGNORECASE)
    
    for idx, col in enumerate(columns):
        m = pattern.search(col)
        if m:
            od_pairs.append((m.group(1), m.group(2)))
        else:
            # Fallback nếu tên cột không theo mẫu OD_x-y
            od_pairs.append((str(idx), str(idx)))
            
    return od_pairs


def build_od_topology_matrices(columns, as_sparse=False):
    """
    Xây dựng 2 ma trận ánh xạ nguồn/đích M_in và M_out kích thước [N, N].
    - M_in[i, j] = 1 nếu luồng j != i có cùng node đích với luồng i (cùng đổ về 1 đích).
    - M_out[i, j] = 1 nếu luồng j != i có cùng node nguồn với luồng i (cùng xuất phát từ 1 nguồn).
    
    Khi nhân với vector lưu lượng X[t-1] kích thước [T, N]:
    - neighbor_in = X @ M_in.T  ([T, N])
    - neighbor_out = X @ M_out.T ([T, N])
    Phép tính hoàn tất trong vài mili-giây mà không cần duyệt vòng lặp.
    """
    od_pairs = parse_od_columns(columns)
    N = len(od_pairs)
    
    M_in = np.zeros((N, N), dtype=np.float32)
    M_out = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        src_i, dst_i = od_pairs[i]
        for j in range(N):
            if i == j:
                continue
            src_j, dst_j = od_pairs[j]
            if dst_i == dst_j:
                M_in[i, j] = 1.0
            if src_i == src_j:
                M_out[i, j] = 1.0
                
    if as_sparse:
        return sp.csr_matrix(M_in), sp.csr_matrix(M_out)
    return M_in, M_out


def compute_spatial_neighbors(traffic_arr, M_in, M_out):
    """
    Tính nhanh neighbor_sum_in và neighbor_sum_out cho toàn bộ chuỗi thời gian.
    - traffic_arr: mảng numpy [T, N] (ví dụ giá trị tại t-1)
    - M_in, M_out: ma trận [N, N]
    
    Trả về:
    - neighbor_in: [T, N]
    - neighbor_out: [T, N]
    """
    if sp.issparse(M_in):
        neighbor_in = traffic_arr @ M_in.T.toarray()
        neighbor_out = traffic_arr @ M_out.T.toarray()
    else:
        neighbor_in = np.matmul(traffic_arr, M_in.T)
        neighbor_out = np.matmul(traffic_arr, M_out.T)
        
    return neighbor_in, neighbor_out


def build_physical_flow_adjacency(dataset_name: str, columns: list, top_k: int = 16, data_dir=None) -> np.ndarray:
    """
    Xây dựng ma trận kề vật lý cấp độ luồng (Flow-Level Physical Adjacency Matrix)
    kích thước [N, N] từ ma trận kề topo nút mạng vật lý data/{dataset_name}_adj.npy [V, V].

    Công thức liên kết topo mạng:
    1. A_endpoints = (S @ S.T) + (D @ D.T):
       - S @ S.T: các luồng có chung node nguồn (cùng cạnh tranh băng thông ingress và hàng đợi xuất phát).
       - D @ D.T: các luồng có chung node đích (cùng hội tụ về cổng egress và cạnh tranh giải tỏa lưu lượng).
    2. A_topology = (S @ A_node @ S.T) + (D @ A_node @ D.T):
       - Phản ánh mức độ lân cận vật lý (1-hop physical neighbor) giữa các router nguồn và router đích.
    3. Thêm self-loop bảo toàn thông tin tự thân của luồng.
    4. Giới hạn Top-k lân cận mạnh nhất (Top-k Sparsification, mặc định k=16) để loại bỏ hiện tượng
       over-smoothing và ép mô hình chỉ học tương tác cục bộ vi mô.
    5. Chuẩn hóa theo hàng (Row Normalization) để tạo ma trận ngẫu nhiên (stochastic matrix):
       Tổng mỗi hàng = 1.0, đóng vai trò như bộ lọc trung bình có trọng số theo topo mạng cục bộ.
    """
    import os
    if data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), 'data')

    ds_key = dataset_name.lower()
    adj_file = os.path.join(data_dir, f"{ds_key}_adj.npy")

    od_pairs = parse_od_columns(columns)
    N = len(od_pairs)

    nodes = set()
    for s, d in od_pairs:
        nodes.add(s)
        nodes.add(d)

    try:
        node_ints = [int(n) for n in nodes]
        is_int_nodes = True
        V = max(node_ints)
    except ValueError:
        is_int_nodes = False
        sorted_nodes = sorted(list(nodes))
        node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
        V = len(sorted_nodes)

    S = np.zeros((N, V), dtype=np.float32)
    D = np.zeros((N, V), dtype=np.float32)

    for i, (s, d) in enumerate(od_pairs):
        if is_int_nodes:
            s_idx = int(s) - 1
            d_idx = int(d) - 1
        else:
            s_idx = node_to_idx[s]
            d_idx = node_to_idx[d]
        if 0 <= s_idx < V:
            S[i, s_idx] = 1.0
        if 0 <= d_idx < V:
            D[i, d_idx] = 1.0

    if os.path.exists(adj_file):
        try:
            A_node = np.load(adj_file).astype(np.float32)
            if A_node.shape != (V, V):
                A_node = np.eye(V, dtype=np.float32)
        except Exception:
            A_node = np.eye(V, dtype=np.float32)
    else:
        A_node = np.eye(V, dtype=np.float32)

    # Lan truyền topo: Cặp nguồn, cặp đích và lân cận vật lý 1-hop
    A_flow = (S @ S.T) + (D @ D.T) + (S @ A_node @ S.T) + (D @ A_node @ D.T)

    # Thêm self-loop bảo toàn thông tin tự thân của luồng
    np.fill_diagonal(A_flow, np.diag(A_flow) + 1.0)

    # Top-k Sparsification: Chỉ giữ lại k láng giềng liên kết vật lý mạnh nhất
    if top_k is not None and 0 < top_k < N:
        A_sparse = np.zeros_like(A_flow)
        for i in range(N):
            row = A_flow[i].copy()
            top_k_indices = np.argpartition(row, -top_k)[-top_k:]
            # Đảm bảo self-loop i luôn có mặt
            if i not in top_k_indices:
                min_idx = top_k_indices[np.argmin(row[top_k_indices])]
                top_k_indices[np.where(top_k_indices == min_idx)[0][0]] = i
            A_sparse[i, top_k_indices] = row[top_k_indices]
        A_flow = A_sparse

    # Chuẩn hóa theo hàng để tổng hàng = 1.0
    row_sum = A_flow.sum(axis=-1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    A_flow_norm = (A_flow / row_sum).astype(np.float32)

    return A_flow_norm

