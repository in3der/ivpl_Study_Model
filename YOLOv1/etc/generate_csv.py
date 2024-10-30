import csv

# "train.txt" 파일을 읽어 각 줄을 리스트로 저장합니다.
read_train = open("train.txt", "r").readlines()

# "train.csv" 파일을 쓰기 모드로 열고, newline을 비워둡니다.
with open("train.csv", mode="w", newline="") as train_file:
    # "train.txt"의 각 줄에 대해 반복합니다.
    for line in read_train:
        # 줄에서 파일 이름을 추출하고, 개행 문자를 제거합니다.
        image_file = line.split("/")[-1].replace("\n", "")
        # 이미지 파일 이름에서 ".jpg" 확장자를 ".txt"로 변경하여 텍스트 파일 이름을 생성합니다.
        text_file = image_file.replace(".jpg", ".txt")
        # 이미지 파일 이름과 텍스트 파일 이름을 리스트로 만듭니다.
        data = [image_file, text_file]
        # CSV 작성기를 생성합니다.
        writer = csv.writer(train_file)
        # 리스트 데이터를 CSV 파일에 한 줄로 씁니다.
        writer.writerow(data)

# "test.txt" 파일을 읽어 각 줄을 리스트로 저장합니다.
read_train = open("test.txt", "r").readlines()

# "test.csv" 파일을 쓰기 모드로 열고, newline을 비워둡니다.
with open("test.csv", mode="w", newline="") as train_file:
    # "test.txt"의 각 줄에 대해 반복합니다.
    for line in read_train:
        # 줄에서 파일 이름을 추출하고, 개행 문자를 제거합니다.
        image_file = line.split("/")[-1].replace("\n", "")
        # 이미지 파일 이름에서 ".jpg" 확장자를 ".txt"로 변경하여 텍스트 파일 이름을 생성합니다.
        text_file = image_file.replace(".jpg", ".txt")
        # 이미지 파일 이름과 텍스트 파일 이름을 리스트로 만듭니다.
        data = [image_file, text_file]
        # CSV 작성기를 생성합니다.
        writer = csv.writer(train_file)
        # 리스트 데이터를 CSV 파일에 한 줄로 씁니다.
        writer.writerow(data)
