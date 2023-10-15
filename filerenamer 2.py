import os
import json
import shutil

# 원본 데이터 폴더 경로와 대상 폴더 경로 설정
original_data_folder = 'originaldata'
renamed_data_folder = 'renameddata'

# 대상 폴더가 없으면 생성
if not os.path.exists(renamed_data_folder):
    os.makedirs(renamed_data_folder)

# 원본 데이터 폴더 내의 모든 폴더에 대해 작업 수행
for folder_name in os.listdir(original_data_folder):
    folder_path = os.path.join(original_data_folder, folder_name)

    # 폴더가 아닌 파일은 무시
    if not os.path.isdir(folder_path):
        continue

    # MotionData.json 파일 경로 설정
    json_file_path = os.path.join(folder_path, 'MotionData.json')

    # MotionData.json 파일이 존재하는 경우에만 작업 수행
    if os.path.exists(json_file_path):
        # 폴더 이름과 MotionData.json 파일 이름을 동일하게 만듭니다.
        new_json_file_name = folder_name + '.json'

        # 대상 폴더로 복사
        new_json_file_path = os.path.join(renamed_data_folder, new_json_file_name)
        shutil.copy(json_file_path, new_json_file_path)

print("작업 완료")
