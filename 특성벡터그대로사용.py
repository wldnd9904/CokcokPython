import os
import json
import numpy as np
from scipy.fft import fft
from sklearn.cluster import KMeans

data_folder = "renameddata"  # JSON 파일이 있는 폴더 경로
file_labels = {"A": 0, "B": 1, "C": 2, "D": 3}  # 각 파일의 라벨

# FFT 결과를 저장할 리스트 초기화
fft_results = []

# JSON 파일 순회
for filename in os.listdir(data_folder):
    if filename.endswith(".json"):
        label = file_labels[filename[0]]  # 파일명의 첫 글자를 라벨로 사용
        with open(os.path.join(data_folder, filename), "r") as file:
            json_data = json.load(file)

            sin_pitch_values = []  # Sin 변환된 pitch 데이터를 저장할 리스트
            sin_roll_values = []  # Sin 변환된 roll 데이터를 저장할 리스트
            sin_yaw_values = []  # Sin 변환된 yaw 데이터를 저장할 리스트
            user_acceleration_x = []
            user_acceleration_y = []
            user_acceleration_z = []

            # 데이터 추출 및 FFT 계산
            for entry in json_data:
                attitude = entry["attitude"]
                pitch = float(attitude["pitch"])
                roll = float(attitude["roll"])
                yaw = float(attitude["yaw"])

                # 각도 데이터를 사인 함수로 변환하여 저장
                sin_pitch_values.append(np.sin(pitch))
                sin_roll_values.append(np.sin(roll))
                sin_yaw_values.append(np.sin(yaw))

                user_acceleration = entry["userAcceleration"]
                user_acceleration_x.append(float(user_acceleration["x"]))
                user_acceleration_y.append(float(user_acceleration["y"]))
                user_acceleration_z.append(float(user_acceleration["z"]))

            # 각 차원에 대한 FFT 계산
            fft_pitch = fft(sin_pitch_values)
            fft_roll = fft(sin_roll_values)
            fft_yaw = fft(sin_yaw_values)
            fft_acc_x = fft(user_acceleration_x)
            fft_acc_y = fft(user_acceleration_y)
            fft_acc_z = fft(user_acceleration_z)

            # FFT 결과를 하나의 특성 벡터로 결합
            combined_fft_result = np.concatenate([fft_pitch, fft_roll, fft_yaw, fft_acc_x, fft_acc_y, fft_acc_z])

            # 결합된 FFT 결과를 리스트에 추가
            fft_results.append((combined_fft_result, label))

# FFT 결과를 NumPy 배열로 변환
fft_results = np.array(fft_results)

# 특성 벡터와 라벨 추출
X = fft_results[:, 0]
y = fft_results[:, 1]

# K-Means 클러스터링을 사용하여 데이터 군집화
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# 군집화 결과
clusters = kmeans.labels_

# 군집화 결과 출력
for cluster_label in np.unique(clusters):
    print(f"Cluster {cluster_label}:")
    cluster_indices = np.where(clusters == cluster_label)[0]
    cluster_labels = y[cluster_indices]
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
    for unique_label, count in zip(unique_labels, label_counts):
        print(f"Label {unique_label}: {count} instances")
    print("=" * 50)
