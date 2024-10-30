import os
import xml.etree.ElementTree as ET

# 경로 설정
#annotation_dir = '/home/ivpl-d29/dataset/VOC/VOC2007/Annotations'
#image_dir = '/home/ivpl-d29/dataset/VOC/VOC2007/JPEGImages'
#output_file = '/home/ivpl-d29/dataset/VOC/VOC2007/VOC2007.txt'

annotation_dir = '/home/ivpl-d29/dataset/VOC/VOC2012/Annotations'
image_dir = '/home/ivpl-d29/dataset/VOC/VOC2012/JPEGImages'
output_file = '/home/ivpl-d29/dataset/VOC/VOC2012/VOC2012.txt'

# 클래스 이름과 해당하는 class_id를 정의
classes = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", 
           "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", 
           "sofa", "tvmonitor"]

# 클래스 이름을 class_id로 변환하는 함수
def get_class_id(class_name):
    if class_name in classes:
        return classes.index(class_name)
    else:
        return -1

# VOC XML 파일 파싱
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    image_data = []
    image_filename = root.find('filename').text

    # 객체 정보 추출
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = get_class_id(class_name)

        if class_id == -1:
            continue

        # 바운딩 박스 정보 추출
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        # 이미지 파일 이름을 제외한 객체 정보(x1, y1, x2, y2, class_id)를 추가
        image_data.append(f"{xmin} {ymin} {xmax} {ymax} {class_id}")
    
    return image_filename, image_data

# 모든 XML 파일을 순회하며 파싱한 결과를 저장
def parse_annotations():
    with open(output_file, 'w') as f:
        for xml_file in os.listdir(annotation_dir):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(annotation_dir, xml_file)
                image_filename, image_data = parse_xml(xml_path)

                # 이미지 이름과 그에 속한 모든 객체 정보를 하나의 줄에 기록
                if image_data:
                    f.write(f"{image_filename} " + " ".join(image_data) + "\n")

if __name__ == "__main__":
    parse_annotations()
