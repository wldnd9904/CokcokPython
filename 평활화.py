import json
import numpy as np
import matplotlib.pyplot as plt

# JSON 파일 경로
file_path = 'motiondata/A10_01.json'
# JSON 파일 읽기
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터를 저장할 리스트 생성 (data와 data2에 맞게 변수 이름 변경)
timestamps = []
sin_pitch_values = []  # Sin 변환된 pitch 데이터를 저장할 리스트
sin_roll_values = []  # Sin 변환된 roll 데이터를 저장할 리스트
sin_yaw_values = []  # Sin 변환된 yaw 데이터를 저장할 리스트
user_acceleration_x = []
user_acceleration_y = []
user_acceleration_z = []


# 데이터 추출 (data와 data2에 맞게 변수 이름 변경)

for entry in data:
        timestamps.append(entry["timestamp"])
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

# 데이터 평활화 (예: 이동평균)
window_size = 10 # 이동평균 윈도우 크기
smoothed_x = np.convolve(user_acceleration_x, np.ones(window_size) / window_size, mode='valid')
smoothed_y = np.convolve(user_acceleration_y, np.ones(window_size) / window_size, mode='valid')
smoothed_z = np.convolve(user_acceleration_z, np.ones(window_size) / window_size, mode='valid')

# 그래프로 시각화
plt.figure(figsize=(12, 6))
plt.plot(timestamps[window_size - 1:], smoothed_x, label='Smoothed X')
plt.plot(timestamps[window_size - 1:], smoothed_y, label='Smoothed Y')
plt.plot(timestamps[window_size - 1:], smoothed_z, label='Smoothed Z')

# 원래 데이터도 함께 그래프에 표시
plt.plot(timestamps, user_acceleration_x, linestyle='--', label='Original X', alpha=0.5)
plt.plot(timestamps, user_acceleration_y, linestyle='--', label='Original Y', alpha=0.5)
plt.plot(timestamps, user_acceleration_z, linestyle='--', label='Original Z', alpha=0.5)

plt.xlabel('Timestamp')
plt.ylabel('Acceleration')
plt.title('Smoothed User Acceleration Data')
plt.legend()
plt.grid(True)
plt.show()
