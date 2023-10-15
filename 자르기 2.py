import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 입력 및 출력 디렉토리 경로 설정
input_dir = 'renameddata'
output_dir = 'swingdata'

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

        # 스윙 세그먼트 추출 함수
        def extract_swing_segments(acceleration, threshold=1):
            is_swinging = False
            swing_segments = []
            current_segment = []

            for i, accel_value in enumerate(acceleration):
                if abs(accel_value) >= threshold:  # 절댓값 적용
                    if not is_swinging:
                        is_swinging = True
                        current_segment = [i]
                elif is_swinging:
                    is_swinging = False
                    current_segment.append(i)
                    swing_segments.append(current_segment)

            return swing_segments

        # 스윙 세그먼트 추출
        swing_threshold = 1  # 스윙 감지 임계값 설정

        # X 축 스윙 세그먼트 추출
        x_swing_segments = extract_swing_segments(user_acceleration_x, swing_threshold)

        # Y 축 스윙 세그먼트 추출
        y_swing_segments = extract_swing_segments(user_acceleration_y, swing_threshold)

        # Z 축 스윙 세그먼트 추출
        z_swing_segments = extract_swing_segments(user_acceleration_z, swing_threshold)

        # 가장 빠른 시작점과 가장 늦은 종료점을 찾아 동일한 시간 범위로 맞추기
        # X, Y, Z 축 스윙 세그먼트의 시작 시간 중 가장 작은 값
        x_min_start_time = min(segment[0] for segment in x_swing_segments) if x_swing_segments else 0
        y_min_start_time = min(segment[0] for segment in y_swing_segments) if y_swing_segments else 0
        z_min_start_time = min(segment[0] for segment in z_swing_segments) if z_swing_segments else 0
        min_start_time = min(x_min_start_time, y_min_start_time, z_min_start_time)

        # X, Y, Z 축 스윙 세그먼트의 종료 시간 중 가장 큰 값
        x_max_end_time = max(segment[-1] for segment in x_swing_segments) if x_swing_segments else 0
        y_max_end_time = max(segment[-1] for segment in y_swing_segments) if y_swing_segments else 0
        z_max_end_time = max(segment[-1] for segment in z_swing_segments) if z_swing_segments else 0
        max_end_time = max(x_max_end_time, y_max_end_time, z_max_end_time)

        # 시작 및 종료 시간을 설정하되, 범위를 벗어나지 않도록 조건을 추가
        start_time = max(min_start_time - 5, 0)
        end_time = min(max_end_time + 5, len(timestamps) - 1)  # 배열 범위를 벗어나지 않도록 설정

        # 스윙 세그먼트를 동일한 시간 범위로 맞춤
        x_swing_data = user_acceleration_x[start_time:end_time + 1]
        y_swing_data = user_acceleration_y[start_time:end_time + 1]
        z_swing_data = user_acceleration_z[start_time:end_time + 1]

        # 그래프로 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps[start_time:end_time + 1], x_swing_data, label='X-Axis Acceleration Data')
        plt.plot(timestamps[start_time:end_time + 1], y_swing_data, label='Y-Axis Acceleration Data')
        plt.plot(timestamps[start_time:end_time + 1], z_swing_data, label='Z-Axis Acceleration Data')
        plt.xlabel('Time')
        plt.ylabel('Acceleration')
        plt.title('Simultaneously Displayed X, Y, and Z-Axis Swing Data (Using Absolute Values)')
        plt.legend()
        plt.grid(True)

        # 그래프 저장
        graph_filename = os.path.splitext(filename)[0] + '.png'
        graph_path = os.path.join(output_dir, graph_filename)
        plt.savefig(graph_path)
        plt.close()

        # JSON 파일 저장
        output_data = {
            "timestamps": timestamps[start_time:end_time + 1],
            "x_swing_data": x_swing_data,
            "y_swing_data": y_swing_data,
            "z_swing_data": z_swing_data
        }

        json_filename = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as json_output_file:
            json.dump(output_data, json_output_file)
