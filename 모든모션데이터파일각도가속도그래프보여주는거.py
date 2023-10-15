import matplotlib.pyplot as plt
import numpy as np
import json
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# renameddata 폴더 경로 설정
folder_path = 'renameddata_wolgye'

# renameddata 폴더 내의 모든 파일에 대해 작업 수행
for file_name in os.listdir(folder_path):
    # 파일이 JSON 파일인지 확인
    if file_name.endswith('.json'):
        # JSON 파일 경로 설정
        file_path = os.path.join(folder_path, file_name)

        # JSON 파일 읽기
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        timestamps = []
        sin_pitch_values = []  # Sin 변환된 pitch 데이터를 저장할 리스트
        sin_roll_values = []  # Sin 변환된 roll 데이터를 저장할 리스트
        sin_yaw_values = []  # Sin 변환된 yaw 데이터를 저장할 리스트
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

            # 각도 데이터를 사인 함수로 변환하여 저장
            sin_pitch_values.append(np.sin(pitch))
            sin_roll_values.append(np.sin(roll))
            sin_yaw_values.append(np.sin(yaw))

            user_acceleration = entry["userAcceleration"]
            user_acceleration_x.append(float(user_acceleration["x"]))
            user_acceleration_y.append(float(user_acceleration["y"]))
            user_acceleration_z.append(float(user_acceleration["z"]))

        # 그래프 생성 및 저장
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(timestamps, sin_pitch_values, label='Sin(Pitch)')
        plt.plot(timestamps, sin_roll_values, label='Sin(Roll)')
        plt.plot(timestamps, sin_yaw_values, label='Sin(Yaw)')
        plt.title('Sin Transformed Attitude Data')
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(timestamps, user_acceleration_x, label='X')
        plt.plot(timestamps, user_acceleration_y, label='Y')
        plt.plot(timestamps, user_acceleration_z, label='Z')
        plt.title('User Acceleration Data')
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.legend()

        # 그래프 파일로 저장 (파일 이름은 JSON 파일 이름과 동일하게)
        graph_file_name = os.path.splitext(file_name)[0] + '.png'
        graph_file_path = os.path.join(folder_path, graph_file_name)
        plt.tight_layout()
        plt.savefig(graph_file_path)
        plt.close()

# 그래프 창 띄우기
plt.tight_layout()
plt.show()