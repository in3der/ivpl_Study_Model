import sys
import cv2
import matplotlib.pyplot as plt

def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs

def config(gs, img, strategy='q'):
    gs.setBaseImage(img)

    if strategy == 's':
        gs.switchToSingleStrategy()
    elif strategy == 'f':
        gs.switchToSelectiveSearchFast()    # R-CNN에서 사용하는 Select Search mode
    elif strategy == 'q':
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)

def get_rects(gs, max_rects=2000):
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]
    return rects[:max_rects]  # 상위 max_rects 개수만 반환

def plot_image_with_boxes(img, rects):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
    ax = plt.gca()

    for (x1, y1, x2, y2) in rects:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    """
    Selective search 알고리즘 동작 
    """
    gs = get_selective_search()

    img = cv2.imread('/home/ivpl-d29/myProject/Study_Model/R-CNN/pet.jpg', cv2.IMREAD_COLOR)
    config(gs, img, strategy='f')

    rects = get_rects(gs, max_rects=2000)  # 2000개의 사각형만 가져오기
    print(rects)

    # Plotting the image with bounding boxes
    plot_image_with_boxes(img, rects)
