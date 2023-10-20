import pandas as pd
import math
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from unicodedata import normalize as nml
from fastdtw import fastdtw


def getData(forder):
    return_data = []
    namelist = []

    # expert 폴더 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, forder)

    # 폴더 내의 모든 파일에 대해 작업 수행
    for file_name in os.listdir(folder_path):

        # 파일이 JSON 파일인지 확인
        if file_name.endswith('.json'):

            # JSON 파일 경로 설정
            file_path = os.path.join(folder_path, file_name)

            # JSON 파일 읽기
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # 빈 리스트 생성
            new_data = []

            # JSON 데이터를 원하는 형태로 변환
            for entry in data:
                new_entry = {
                    "timestamp": entry["timestamp"],
                    "x": float(entry["userAcceleration"]["x"]),
                    "y": float(entry["userAcceleration"]["y"]),
                    "z": float(entry["userAcceleration"]["z"]),
                    "p": np.sin(float(entry["attitude"]["pitch"])),
                    "r": np.sin(float(entry["attitude"]["roll"])),
                    #"w": np.sin(float(entry["attitude"]["yaw"]))
                }
                new_data.append(new_entry)

            # 변환된 데이터를 Pandas DataFrame으로 변환
            df = pd.DataFrame(new_data)

            return_data.append(df)
            namelist.append(file_name)

    return (return_data, namelist)


def getOne(forder, name):
    return_data = []

    # expert 폴더 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, forder)

    # 파일이 JSON 파일인지 확인
    file_name = name + ".json"

    # JSON 파일 경로 설정
    file_path = os.path.join(folder_path, file_name)

    # JSON 파일 읽기
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # 빈 리스트 생성
    new_data = []

    # JSON 데이터를 원하는 형태로 변환
    for entry in data:
        new_entry = {
            "timestamp": entry["timestamp"],
            "x": float(entry["userAcceleration"]["x"]),
            "y": float(entry["userAcceleration"]["y"]),
            "z": float(entry["userAcceleration"]["z"]),
            "p": np.sin(float(entry["attitude"]["pitch"])),
            "r": np.sin(float(entry["attitude"]["roll"])),
            #"w": np.sin(float(entry["attitude"]["yaw"]))
        }
        new_data.append(new_entry)

    # 변환된 데이터를 Pandas DataFrame으로 변환
    df = pd.DataFrame(new_data)

    return_data.append(df)

    return return_data


def makeGraph(forder):
    # expert 폴더 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, forder)

    # 폴더 내의 모든 파일에 대해 작업 수행
    for file_name in os.listdir(folder_path):
        file_name = nml("NFC", file_name)

        # 파일이 JSON 파일인지 확인
        if file_name.endswith('.json'):

            # JSON 파일 경로 설정
            file_path = os.path.join(folder_path, file_name)

            # JSON 파일 읽기
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # 빈 리스트 생성
            new_data = []

            # JSON 데이터를 원하는 형태로 변환
            for entry in data:
                new_entry = {
                    "timestamp": entry["timestamp"],
                    "x_acceleration": float(entry["userAcceleration"]["x"]),
                    "y_acceleration": float(entry["userAcceleration"]["y"]),
                    "z_acceleration": float(entry["userAcceleration"]["z"]),
                    "roll": float(entry["attitude"]["roll"]),
                    #"yaw": float(entry["attitude"]["yaw"]),
                    "pitch": float(entry["attitude"]["pitch"])
                }
                new_data.append(new_entry)

            # 변환된 데이터를 Pandas DataFrame으로 변환
            df = pd.DataFrame(new_data)

            # 그래프 생성 및 저장
            plt.figure(figsize=(10, 6))

            plt.subplot(2, 1, 1)
            plt.plot(df['timestamp'], np.sin(df['pitch']), label='Sin(Pitch)')
            plt.plot(df['timestamp'], np.sin(df['roll']), label='Sin(Roll)')
            #plt.plot(df['timestamp'], np.sin(df['yaw']), label='Sin(Yaw)')
            plt.title('Sin Transformed Attitude Data')
            plt.xlabel('Timestamp')
            plt.ylabel('Values')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(df['timestamp'], df['x_acceleration'], label='X')
            plt.plot(df['timestamp'], df['y_acceleration'], label='Y')
            plt.plot(df['timestamp'], df['z_acceleration'], label='Z')
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

    return


def cutData(all_data):
    return_data = []

    for data in all_data:
        max_index = data['z'].idxmax()

        start_index = max_index - 11
        end_index = max_index + 6

        sliced_df = data.iloc[start_index:end_index].reset_index(drop=True)

        return_data.append(sliced_df)

    return return_data


def normalize(all_data):
    return_data = []

    for data in all_data:
        result = data.copy()

        for feature_name in data.columns:
            max_value = data[feature_name].max()
            min_value = data[feature_name].min()

            result[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)

        return_data.append(result)

    return return_data


def avgData(all_data):
    sum_df = all_data[0].copy()
    for df in all_data[1:]:
        sum_df += df

    average_df = sum_df / len(all_data)

    return [average_df]


# 데이터에서 사용할 속성 선택
def extract_feature(data):
    features = []
    for entry in data:
        feature = [
            float(entry['userAcceleration']['x']),
            float(entry['userAcceleration']['y']),
            float(entry['userAcceleration']['z']),
            float(entry['attitude']['pitch']),
            #float(entry['attitude']['yaw']),
            float(entry['attitude']['roll'])
        ]
        features.append(feature)
    return features


# DTW를 사용하여 두 데이터 사이의 유사성 계산
def calculate_dtw_distance(data1, data2):
    #data1_features = extract_feature(data1)
    #data2_features = extract_feature(data2)
    distance, _ = fastdtw(data1, data2)
    return distance


def getPersonList(list):
    return_data = []

    for name in list:
        # expert 폴더 경로 설정
        folder_path = name[0]
        same_name = []
        # 폴더 내의 모든 파일에 대해 작업 수행
        for file_name in os.listdir(folder_path):
            file_name = nml("NFC", file_name)
            # 파일이 JSON 파일인지 확인
            if file_name.endswith('.json') and file_name.startswith(name[1]):
                # JSON 파일 경로 설정
                file_path = os.path.join(folder_path, file_name)
                # JSON 파일 읽기
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                # 빈 리스트 생성
                new_data = []
                # JSON 데이터를 원하는 형태로 변환
                for entry in data:
                    new_entry = {
                        "timestamp": entry["timestamp"],
                        "x": float(entry["userAcceleration"]["x"]),
                        "y": float(entry["userAcceleration"]["y"]),
                        "z": float(entry["userAcceleration"]["z"]),
                        "p": np.sin(float(entry["attitude"]["pitch"])),
                        "r": np.sin(float(entry["attitude"]["roll"])),
                        #"w": np.sin(float(entry["attitude"]["yaw"]))
                    }
                    new_data.append(new_entry)
                # 변환된 데이터를 Pandas DataFrame으로 변환
                df = pd.DataFrame(new_data)

                same_name.append(cutData([df])[0])

        return_data.append(avgData(same_name)[0])

    return return_data


# data 읽기 및 dataframe으로 변환
(ex_data, ex_namelist) = getData("expert")
ex_cut_data = cutData(ex_data)
ex_avg = avgData(ex_cut_data)
ex_normalize_data = normalize(ex_cut_data)
ex_avg_normalize_data = normalize(ex_avg)

# data 읽기 및 dataframe으로 변환
(wg_data, wg_namelist) = getData("wg")
(ulsan_data, ulsan_namelist) = getData("intermediate")
low_data = wg_data + ulsan_data
low_namelist = wg_namelist + ulsan_namelist
low_cut_data = cutData(low_data)
low_normalize_data = normalize(low_cut_data)


nameList1 = [["expert", "A남10"],
             ["expert", "A남11"],
             ["expert", "A남12"],
             ["intermediate", "C남10"],
             ["intermediate", "C남2년"],
             ["intermediate", "C남1년"],
             ["intermediate", "D남1."],
             ["intermediate", "D여3년"],
             ["intermediate", "D남1년"],
             ["intermediate", "D여6개"]]
nameList2 = [["wg", "이윤신"],
             ["wg", "정보석"],
             ["wg", "권윤호"],
             ["wg", "신도영"],
             ["wg", "강송모"],
             ["wg", "이태경"],
             #["wg", "정석훈"],
             ["wg", "박지환"],
             ["wg", "최지웅"],
             ["wg", "이신우"],
             #["wg", "김유영"],
             #["wg", "김도규"],
             ["wg", "김민주"],
             ["wg", "서예은"],
             ["wg", "박소현"],
             ["wg", "김송은"],
             ["wg", "이가은"],]

#순위 매길 집단 선택
nameList = nameList2
personList = getPersonList(nameList)
normalized_personlist_data = normalize(personList)

expertfeatures = (ex_avg_normalize_data[0])[['x', 'y', 'z', 'p', 'r']].values.tolist()
dtws = []
names = []
actual_ranks = []
cnt = 1
# DTW 거리 계산 및 이름 리스트 작성
for data in normalized_personlist_data:
    feature = data[['x', 'y', 'z', 'p', 'r' ]].values.tolist()
    dtw_distance = calculate_dtw_distance(feature, expertfeatures)
    dtws.append(dtw_distance)

for namepack in nameList:
    club, name = namepack
    names.append(name)
    actual_ranks.append(cnt)
    cnt += 1

# 이름과 DTW 거리를 zip하여 세트로 만들기
name_dtw_pairs = list(zip(names, dtws, actual_ranks))

# DTW 거리를 기준으로 정렬
name_dtw_pairs.sort(key=lambda pair: pair[1])

# 정렬된 결과 출력
cnt = 1
rank_distance = 0
for name, dtw_distance, actual_rank in name_dtw_pairs:
    print(f"{cnt}: {name}: {dtw_distance} ({actual_rank})")
    rank_distance += abs(cnt - actual_rank)
    cnt += 1

print(f"평균절대오차(MAE): {rank_distance}, 평균: {rank_distance/(cnt-1)} ")