import os
import json
import numpy as np
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft


# FFT를 사용하여 고주파 제거
def filter_high_frequency(data, threshold):
    filtered_data = []
    for datum in data:
        feature = [
            float(datum['userAcceleration']['x']),
            float(datum['userAcceleration']['y']),
            float(datum['userAcceleration']['z'])
        ]
        # FFT를 수행하여 주파수 영역으로 변환
        fft_data = fft(feature)
        # 고주파를 제거 (threshold 이하의 주파수는 0으로 설정)
        fft_data[np.abs(fft_data) < threshold] = 0
        # 역 FFT를 수행하여 시간 영역으로 변환
        filtered_feature = (ifft(fft_data))
        filtered_entry = {
            'timestamp': datum['timestamp'],
            'userAcceleration': {
                'x': filtered_feature[0],
                'y': filtered_feature[1],
                'z': filtered_feature[2],
            },
            'attitude': datum['attitude']
        }
        filtered_data.append(filtered_entry)
    return filtered_data

# 함수를 사용하여 데이터 자르기
def cut_data(data):
    max_value = 0
    index = 0
    for i, feature in enumerate(data):
        z_values = float(feature['userAcceleration']['z'])  # z 가속도 값의 절대값
        if max_value < z_values:
            max_value = z_values
            index = i

    start_index = max(0, index - 25)
    end_index = start_index + 50

    data = data[start_index:end_index]

    return data

# 폴더 경로 설정
input_folder = 'renameddata'
output_folder = 'preprocessed'

# 'preprocessed' 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# 'renameddata' 폴더 내의 모든 .json 파일에 대해 처리
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        input_path = os.path.join(input_folder, filename)
        output_path_json = os.path.join(output_folder, filename)
        output_path_png = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

        # JSON 파일 읽기
        with open(input_path, 'r') as json_file:
            data = json.load(json_file)

        # 데이터 처리
        data = cut_data(data)
        threshold = 0.5
        filtered_data = filter_high_frequency(data, threshold)

        # JSON 파일로 저장
        with open(output_path_json, 'w') as output_json_file:
            json.dump(data, output_json_file)

        # 그래프로 시각화 및 저장
        # 데이터를 저장할 리스트 생성 (data와 data2에 맞게 변수 이름 변경)
        timestamps = []
        sin_pitch_values = []  # Sin 변환된 pitch 데이터를 저장할 리스트
        sin_roll_values = []  # Sin 변환된 roll 데이터를 저장할 리스트
        sin_yaw_values = []  # Sin 변환된 yaw 데이터를 저장할 리스트
        user_acceleration_x = []
        user_acceleration_y = []
        user_acceleration_z = []

        # 데이터 추출 (data와 data2에 맞게 변수 이름 변경)

        for entry in filtered_data:
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

        # 첫 번째 그래프 창 생성
        plt.figure(1, figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(timestamps, sin_pitch_values, label='Sin(Pitch)')
        plt.plot(timestamps, sin_roll_values, label='Sin(Roll)')
        plt.plot(timestamps, sin_yaw_values, label='Sin(Yaw)')
        plt.title('Sin Transformed Attitude Data (data)')
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, user_acceleration_x, label='X')
        plt.plot(timestamps, user_acceleration_y, label='Y')
        plt.plot(timestamps, user_acceleration_z, label='Z')
        plt.title('User Acceleration Data (data)')
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.legend()
        plt.savefig(output_path_png)
        plt.close()

        print(f"Processed and saved: {filename}")
