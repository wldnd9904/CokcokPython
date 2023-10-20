import numpy as np
import matplotlib.pyplot as plt
import os
import json
# JSON 파일 경로
file_path = 'preprocessed/A남 10년 1(우리아빠).json'
# JSON 파일 읽기
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 추출
x_acceleration1 = [float(entry["userAcceleration"]["x"]) for entry in data]
y_acceleration1 = [float(entry["userAcceleration"]["y"]) for entry in data]
z_acceleration1 = [float(entry["userAcceleration"]["z"]) for entry in data]
yaw_values1 = [float(entry["attitude"]["yaw"]) for entry in data]
pitch_values1 = [float(entry["attitude"]["pitch"]) for entry in data]
roll_values1 = [float(entry["attitude"]["roll"]) for entry in data]

# 푸리에 변환 수행
sampling_rate = 50  # 샘플링 주파수 (예: 1초에 1개의 데이터 포인트)
x_fft = np.fft.fft(x_acceleration1)
x_freq = np.fft.fftfreq(len(x_acceleration1), 1 / sampling_rate)
x_amp = np.abs(x_fft)
y_fft = np.fft.fft(y_acceleration1)
y_freq = np.fft.fftfreq(len(y_acceleration1), 1 / sampling_rate)
y_amp = np.abs(y_fft)
z_fft = np.fft.fft(z_acceleration1)
z_freq = np.fft.fftfreq(len(z_acceleration1), 1 / sampling_rate)
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
print(yaw_freq_positive)
print(yaw_amp_positive)

# 결과 출력 (두 데이터 세트를 겹쳐진 플롯으로 표시)
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(x_freq_positive, x_amp_positive, label='X')
plt.plot(y_freq_positive, y_amp_positive, label='Y')
plt.plot(z_freq_positive, z_amp_positive, label='Z')
plt.xlabel('frequency (Hz)')
plt.title('Acceleration frequency domain')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(yaw_freq_positive, yaw_amp_positive, label='Yaw')
plt.plot(pitch_freq_positive, pitch_amp_positive, label='Pitch')
plt.plot(roll_freq_positive, roll_amp_positive, label='Roll')
plt.xlabel('frequency (Hz)')
plt.title('Yaw, Pitch, Roll frequency domain')
plt.legend()

plt.tight_layout()
plt.show()