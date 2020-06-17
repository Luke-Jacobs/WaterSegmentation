from water_detection import *
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage import io

if __name__ == '__main__':
    plt.switch_backend('Qt5Agg')
    img = io.imread('/home/ljacobs/Argonne/water-data/example_cases/heavy_flooding.png')
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    entropy_img = entropy(img, disk(5))
    io.imshow(entropy_img)
    input('?')
