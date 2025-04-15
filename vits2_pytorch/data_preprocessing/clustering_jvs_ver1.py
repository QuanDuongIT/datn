import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def clustering_speaker(speaker_similarity):
    df = pd.read_csv(speaker_similarity, header=None)

    # Tạo ID từ cột đầu tiên (ID nằm trong cột đầu tiên)
    ids = df.iloc[:, 0]

    # Bỏ cột đầu tiên (cột ID) để chỉ lấy các giá trị ma trận tương đồng
    df_without_first_column = df.iloc[:, 1:]

    # Kiểm tra và xử lý giá trị thiếu (nếu có)
    if df_without_first_column.isnull().values.any():
        print("Dữ liệu có giá trị thiếu. Đang thay thế bằng 0...")
        df_without_first_column = df_without_first_column.fillna(0)

    # Chuyển ma trận tương đồng sang ma trận khoảng cách
    similarity_matrix = df_without_first_column.values
    max_sim = np.max(similarity_matrix)
    distance_matrix = max_sim - similarity_matrix

    # Số nhóm mong muốn (có thể thay đổi)
    n_clusters = 5

    # Áp dụng Agglomerative Clustering để phân nhóm
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',  # Dùng ma trận khoảng cách đã tính
        linkage='average'      # Phương pháp liên kết
    )

    # Tính nhãn nhóm cho từng người nói
    labels = clustering.fit_predict(distance_matrix)

    # Gộp kết quả vào DataFrame với ID
    df_result = pd.DataFrame({'Speaker': ids, 'Group': labels})

    # Đảm bảo mỗi ID chỉ có một nhóm duy nhất và không bị chia tách
    grouped = df_result.groupby('Group')['Speaker'].apply(list)

    # In ra các ID trong từng nhóm dưới dạng mảng
    for i, speakers in grouped.items():
        print(f"Nhóm {i + 1} ({len(speakers)} ID): {speakers}")


