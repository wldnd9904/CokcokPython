import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 입력 및 출력 디렉토리 경로 설정
input_dir = 'renameddata'
output_dir = 'swingdatanonsin'

# 입력 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 입력 디렉토리 내의 모든 JSON 파일 처리
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        # JSON 파일 경로 설정
        file_path = os.path.join(input_dir, filename)

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

        # 각도 및 각속도 데이터 추출
        for entry in data:
            timestamps.append(entry["timestamp"])
            attitude = entry["attitude"]
            pitch = float(attitude["pitch"])
            roll = float(attitude["roll"])
            yaw = float(attitude["yaw"])
            sin_pitch_values.append(pitch)
            sin_roll_values.append(roll)
            sin_yaw_values.append(yaw)

            user_acceleration = entry["userAcceleration"]
            user_acceleration_x.append(float(user_acceleration["x"]))
            user_acceleration_y.append(float(user_acceleration["y"]))
            user_acceleration_z.append(float(user_acceleration["z"]))

        # X 축 가속도 데이터 중 최고점 찾기
        max_x_acceleration = max(user_acceleration_x)
        max_x_index = user_acceleration_x.index(max_x_acceleration)

        # 최고점을 중심으로 앞뒤로 25개의 데이터 추출
        start_index = max_x_index - 25 if max_x_index - 25 >= 0 else 0
        end_index = max_x_index + 25 if max_x_index + 25 < len(user_acceleration_x) else len(user_acceleration_x) - 1

        # 추출된 데이터 저장
        output_data = {
            "timestamps": timestamps[start_index:end_index + 1],
            "x_swing_data": user_acceleration_x[start_index:end_index + 1],
            "y_swing_data": user_acceleration_y[start_index:end_index + 1],
            "z_swing_data": user_acceleration_z[start_index:end_index + 1],
            "sin_pitch_values": sin_pitch_values[start_index:end_index + 1],
            "sin_roll_values": sin_roll_values[start_index:end_index + 1],
            "sin_yaw_values": sin_yaw_values[start_index:end_index + 1]
        }

        # 새로운 JSON 파일로 저장
        json_filename = os.path.splitext(filename)[0] + '_data.json'
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as json_output_file:
            json.dump(output_data, json_output_file)

        # 그래프 생성 및 이미지 파일로 저장
        plt.figure(figsize=(12, 6))
        plt.subplot(211)  # 2x1 그리드 중 첫 번째 영역
        plt.plot(output_data["timestamps"], output_data["sin_pitch_values"], label='Sin(Pitch)')
        plt.plot(output_data["timestamps"], output_data["sin_roll_values"], label='Sin(Roll)')
        plt.plot(output_data["timestamps"], output_data["sin_yaw_values"], label='Sin(Yaw)')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Combined Sin(Pitch), Sin(Roll), and Sin(Yaw) Data')
        plt.legend()
        plt.grid(True)

        plt.subplot(212)  # 2x1 그리드 중 두 번째 영역
        plt.plot(output_data["timestamps"], output_data["x_swing_data"], label='X-Axis Acceleration Data')
        plt.plot(output_data["timestamps"], output_data["y_swing_data"], label='Y-Axis Acceleration Data')
        plt.plot(output_data["timestamps"], output_data["z_swing_data"], label='Z-Axis Acceleration Data')
        plt.xlabel('Time')
        plt.ylabel('Acceleration')
        plt.title('Combined X, Y, and Z-Axis Swing Data (Using Absolute Values)')
        plt.legend()
        plt.grid(True)

        image_filename = os.path.splitext(filename)[0] + '_combined.png'
        image_path = os.path.join(output_dir, image_filename)
        plt.tight_layout()  # 서브플롯 레이아웃 조정
        plt.savefig(image_path)
        plt.close()

        print(f"Processed: {json_filename}")

print("Data extraction and saving completed.")
