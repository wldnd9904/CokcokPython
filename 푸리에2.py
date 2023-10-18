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

    return fft_data

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
input_folder = 'preprocessed'

# 'renameddata' 폴더 내의 모든 .json 파일에 대해 처리
for filename in os.listdir(input_folder):
    if filename.endswith(".json") and filename.startswith("A"):
        input_path = os.path.join(input_folder, filename)
        # JSON 파일 읽기
        with open(input_path, 'r') as json_file:
            data = json.load(json_file)

        # 데이터 처리
        data = cut_data(data)
        threshold = 0.5
        filtered_data = filter_high_frequency(data, threshold)
        print(filename.title())
        print(filtered_data)