import numpy as np
from dtw import *

# 두 시계열 데이터 예시
time_series1 = np.array([1, 2, 3, 4, 5])
time_series2 = np.array([2, 3, 4, 5, 6])

# DTW 경로를 계산
alignment = dtw(time_series1, time_series2, keep_internals=True)

# DTW 경로 및 거리를 출력
alignment.plot(type="twoway")
print("DTW 거리:", alignment.distance)