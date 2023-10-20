import os

from unicodedata import normalize as nml
# MLP 폴더 경로
mlp_folder = 'mlp'

# MLP 폴더 내의 모든 파일 목록 가져오기
file_list = os.listdir(mlp_folder)

# 파일 이름 수정 및 이동
for file_name in file_list:
    file_name = nml("NFC", file_name)
    # 파일 이름에서 '남 ' 또는 '여 '를 빈 문자열로 대체
    new_name = file_name.replace('남 ', '').replace('여 ', '')
    # 파일 이름에서 '년 '을 '_'로 대체
    new_name = new_name.replace('년 ', '_')

    # 새로운 파일 이름 생성
    new_path = os.path.join(mlp_folder, new_name)

    # 파일 이동 (이름 변경)
    os.rename(os.path.join(mlp_folder, file_name), new_path)

    # 수정된 파일 이름 출력
    print(f"수정된 파일 이름: {new_name}")
