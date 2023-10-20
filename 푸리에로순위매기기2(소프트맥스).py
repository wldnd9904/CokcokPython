import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from unicodedata import normalize as nml


# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(data_dir, isnotforTrainning = True):
    data = []
    labels = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            file_name = nml("NFC", file_name)
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'r') as json_file:
                input_data = json.load(json_file)
                #print(file_name)
                # 푸리에 변환 수행
                # 데이터 추출
                x_acceleration = [float(entry["userAcceleration"]["x"]) for entry in input_data]
                y_acceleration = [float(entry["userAcceleration"]["y"]) for entry in input_data]
                z_acceleration = [float(entry["userAcceleration"]["z"]) for entry in input_data]
                yaw_values1 = [float(entry["attitude"]["yaw"]) for entry in input_data]
                pitch_values1 = [float(entry["attitude"]["pitch"]) for entry in input_data]
                roll_values1 = [float(entry["attitude"]["roll"]) for entry in input_data]

                # 푸리에 변환 수행
                sampling_rate = 50  # 샘플링 주파수 (예: 1초에 1개의 데이터 포인트)
                x_fft = np.fft.fft(x_acceleration)
                x_freq = np.fft.fftfreq(len(x_acceleration), 1 / sampling_rate)
                x_amp = np.abs(x_fft)
                y_fft = np.fft.fft(y_acceleration)
                y_freq = np.fft.fftfreq(len(y_acceleration), 1 / sampling_rate)
                y_amp = np.abs(y_fft)
                z_fft = np.fft.fft(z_acceleration)
                z_freq = np.fft.fftfreq(len(z_acceleration), 1 / sampling_rate)
                z_amp = np.abs(z_fft)
                yaw_fft = np.fft.fft(yaw_values1)
                yaw_freq = np.fft.fftfreq(len(yaw_values1), 1 / sampling_rate)
                yaw_amp = np.abs(yaw_fft)
                pitch_fft = np.fft.fft(pitch_values1)
                pitch_freq = np.fft.fftfreq(len(pitch_values1), 1 / sampling_rate)
                pitch_amp = np.abs(pitch_fft)
                roll_fft = np.fft.fft(roll_values1)
                roll_freq = np.fft.fftfreq(len(roll_values1), 1 / sampling_rate)
                roll_amp = np.abs(roll_fft)

                # 양의 부분만 선택
                x_freq_positive = x_freq[x_freq > 0]
                x_amp_positive = x_amp[x_freq > 0]
                y_freq_positive = y_freq[y_freq > 0]
                y_amp_positive = y_amp[y_freq > 0]
                z_freq_positive = z_freq[z_freq > 0]
                z_amp_positive = z_amp[z_freq > 0]
                yaw_freq_positive = yaw_freq[yaw_freq > 0]
                yaw_amp_positive = yaw_amp[yaw_freq > 0]
                pitch_freq_positive = pitch_freq[pitch_freq > 0]
                pitch_amp_positive = pitch_amp[pitch_freq > 0]
                roll_freq_positive = roll_freq[roll_freq > 0]
                roll_amp_positive = roll_amp[roll_freq > 0]
                # 푸리에 변환된 데이터와 다른 데이터 결합
                combined_data = np.column_stack((x_freq_positive,x_amp_positive,y_freq_positive,y_amp_positive,z_freq_positive,z_amp_positive,yaw_freq_positive,yaw_amp_positive,pitch_freq_positive,pitch_amp_positive,roll_freq_positive,roll_amp_positive))

                # 데이터 정규화
                combined_data = StandardScaler().fit_transform(combined_data)

                if file_name.startswith('A'):
                    label = 2  # 파일 이름이 'A'로 시작하면 1로 설정
                elif file_name.startswith('C'):
                    label = 1
                else:
                    label = 0  # 그렇지 않으면 0으로 설정

                data.append(combined_data)
                labels.append([label,file_name])
    return np.array(data), np.array(labels)


# 데이터 로드
train_data, train_labels = load_and_preprocess_data('preprocessed')

# MLP 모델 생성
model = keras.Sequential([
    layers.Flatten(input_shape=(train_data.shape[1], train_data.shape[2])),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 학습 데이터 및 테스트 데이터 분리
#train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2)
# 데이터 로드
test_data, test_labels = load_and_preprocess_data('preprocessed_wolgye')

train_labels = np.array([int(item[0]) for item in train_labels])  # 숫자를 추출하여 배열에 저장
test_names = [item[1] for item in test_labels] # 이름 추출하여 배열에 저장
# 모델 학습
model.fit(train_data, train_labels, epochs=40, batch_size=32, validation_split=0.2)

# 모델을 사용하여 예측 확률을 얻습니다
predicted_probabilities = model.predict(test_data)
print(predicted_probabilities)
# predicted_probabilities는 각 데이터 포인트에 대한 예측 확률을 포함하는 배열입니다.
# 예를 들어, predicted_probabilities[0]은 첫 번째 데이터 포인트의 예측 확률을 나타냅니다.

# 이제 이 확률을 양성 클래스 (참)로 예측되는 확률로 해석할 수 있습니다.
# 예를 들어, 각 데이터 포인트가 양성 클래스로 예측될 확률을 출력해보겠습니다.
prob_dict = {}
for i, prob in enumerate(predicted_probabilities):
    #print(f"{test_names[i]} {i + 1}: 양성 클래스로 예측될 확률 = {prob[0]:.4f}")
    key = test_names[i][0:3]
    if key not in prob_dict:
        prob_dict[key] = []
    prob_dict[key].append(prob)


# 각 키에 대한 평균을 계산하고 rank 배열에 저장
rank = []
for key in prob_dict:
    numbers = prob_dict[key]
    psum = 0
    for pred in numbers:
        psum += pred[1] + 2*pred[2]
    average = psum / len(numbers)
    rank.append((key, average))

# rank 배열을 평균값을 기준으로 내림차순으로 정렬
rank = sorted(rank, key=lambda x: x[1], reverse=True)

# 정렬된 rank 배열 출력
for key, average in rank:
    print(f"{key}: {average:.6f}")

# 모델 평가
#test_l