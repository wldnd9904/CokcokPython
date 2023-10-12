import pywt
import numpy as np
import matplotlib.pyplot as plt
import json

# JSON 파일 경로
file_path = 'motiondata/A10_01.json'

# JSON 파일 읽기
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터를 저장할 리스트 생성
timestamps = []
sin_pitch_values = []
sin_roll_values = []
sin_yaw_values = []
user_acceleration_x = []
user_acceleration_y = []
user_acceleration_z = []

# 데이터 추출
for entry in data:
    timestamps.append(entry["timestamp"])
    attitude = entry["attitude"]
    pitch = float(attitude["pitch"])
    roll = float(attitude["roll"])
    yaw = float(attitude["yaw"])

    sin_pitch_values.append(np.sin(pitch))
    sin_roll_values.append(np.sin(roll))
    sin_yaw_values.append(np.sin(yaw))

    user_acceleration = entry["userAcceleration"]
    user_acceleration_x.append(float(user_acceleration["x"]))
    user_acceleration_y.append(float(user_acceleration["y"]))
    user_acceleration_z.append(float(user_acceleration["z"]))

# 웨이블릿 변환 수행 함수
def perform_wavelet_transform(data, wavelet, level, energy_threshold):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    detail = coeffs[1]
    # 에너지가 임계값 미만인 세그먼트의 에너지를 0으로 설정
    selected_coefficients = [coeff if np.sum(np.square(coeff)) > energy_threshold else np.zeros_like(coeff) for coeff in detail]
    reconstructed_data = pywt.waverec([None, selected_coefficients], wavelet)
    return reconstructed_data


# 에너지 임계값 설정 (에너지가 1 미만인 세그먼트를 제거)
energy_threshold = 1

# X 축 데이터 처리
reconstructed_x = perform_wavelet_transform(user_acceleration_x, 'db4', 2, energy_threshold)

# Y 축 데이터 처리
reconstructed_y = perform_wavelet_transform(user_acceleration_y, 'db4', 2, energy_threshold)

# Z 축 데이터 처리
reconstructed_z = perform_wavelet_transform(user_acceleration_z, 'db4', 2, energy_threshold)
# reconstructed_x, reconstructed_y, reconstructed_z의 길이를 timestamps와 맞춤
min_length = min(len(reconstructed_x), len(timestamps))
reconstructed_x = reconstructed_x[:min_length]
reconstructed_y = reconstructed_y[:min_length]
reconstructed_z = reconstructed_z[:min_length]

# 그래프로 시각화
plt.figure(figsize=(12, 6))
plt.plot(timestamps, user_acceleration_x[:min_length], label='Original X')
plt.plot(timestamps, reconstructed_x, label='Selected X', linestyle='--')
plt.plot(timestamps, user_acceleration_y[:min_length], label='Original Y')
plt.plot(timestamps, reconstructed_y, label='Selected Y', linestyle='--')
plt.plot(timestamps, user_acceleration_z[:min_length], label='Original Z')
plt.plot(timestamps, reconstructed_z, label='Selected Z', linestyle='--')

plt.xlabel('Timestamp')
plt.ylabel('Acceleration')
plt.title('Selected Segments Based on Energy Threshold (X, Y, Z)')
plt.legend()
plt.grid(True)
plt.show()
