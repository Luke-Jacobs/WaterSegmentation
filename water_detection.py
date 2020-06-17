import cv2
import numpy as np
from PIL import Image
import os.path
from typing import Tuple
from matplotlib import pyplot as plt
import pickle
import argparse
from skimage.filters import rank
import skimage.morphology
import skimage.io


# Constants
BALLFIELD_MASK = cv2.cvtColor(cv2.imread('/home/ljacobs/Argonne/water-data/ball_park_1/bin_mask_ballpark1.png'),
                              cv2.COLOR_BGR2GRAY)
FOLDER_1_021A = '/home/ljacobs/Argonne/water-data/ball_park_0/bp1_021A'
FOLDER_0_016A = '/home/ljacobs/Argonne/water-data/ball_park_0/bp0_016A'
EXAMPLES_FOLDER = '/home/ljacobs/Argonne/water-data/example_cases'
ZOOMED_EXAMPLES_FOLDER = '/home/ljacobs/Argonne/water-data/example_cases/zoomed'
FOLDER_FLOODED = '/home/ljacobs/Argonne/water-data/water_labeled/flooded'
FOLDER_SOME_WATER = '/home/ljacobs/Argonne/water-data/water_labeled/some_water'
FOLDER_NO_WATER = '/home/ljacobs/Argonne/water-data/water_labeled/no_water'

# Crops: X, Y, Width, Height
BALLFIELD_1_CROP = (319, 382, 794, 189)
BALLFIELD_1_TIGHT_CROP = (537, 441, 534, 104)
BALLFIELD_0_CROP = (384, 354, 870, 206)
BALLFIELD_0_TIGHT_CROP = (574, 379, 581, 113)
BALLFIELD_TILTED_VIEW_CROP = (476, 343, 754, 162)

# People detection initialization
hog_classify = cv2.HOGDescriptor()
hog_classify.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
plt.switch_backend('Qt5Agg')


def apply_crop(img: np.ndarray, crop: Tuple[int, int, int, int]):
    x, y, w, h = crop
    return img[y:y+h, x:x+w]

def show_img(img: np.ndarray):
    Image.fromarray(img).show()

def stitch_together(*imgs, text: str = None) -> np.ndarray:
    height = max([img.shape[0] for img in imgs])
    width = sum([img.shape[1] for img in imgs])
    output_canvas = np.zeros((height, width, 3)).astype(np.uint8)
    prev_marker = 0
    for img in imgs:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        output_canvas[:, prev_marker:prev_marker+img.shape[1]] = img
        prev_marker = prev_marker+img.shape[1]
    if text is not None:
        cv2.putText(output_canvas, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return output_canvas

def get_images_from(folder: str, first_n: int = None, exclude_containing: str = None, no_load: bool = False, sort=True) -> Tuple[str, np.ndarray]:
    img_formats = ['.png', '.jpg']
    img_paths = [os.path.join(folder, img_path) for img_path in os.listdir(folder)
                 if os.path.splitext(img_path)[-1] in img_formats]
    if exclude_containing is not None:
        assert type(exclude_containing) == str
        img_paths = [img_path for img_path in img_paths if img_path.count(exclude_containing) == 0]

    def sort_by_frame(img_path):
        return int(os.path.basename(img_path)[:-4])

    if sort:
        key_func = sort_by_frame
    else:
        key_func = None

    for i, img_path in enumerate(sorted(img_paths, key=key_func)):
        if no_load:
            out = img_path
        else:
            out = (img_path, cv2.imread(img_path))

        if first_n is None:
            yield out
        else:
            if i < first_n:
                yield out
            else:
                return
    return

def apply_and_show_detection(img: np.ndarray, bg_sub: cv2.BackgroundSubtractorMOG2):
    # Apply mask to the image to extract the ballfield
    bg_sub_mask = bg_sub.apply(img)
    ret, fg_mask = cv2.threshold(bg_sub_mask, 200, 255, cv2.THRESH_BINARY)
    ret, shadow_mask = cv2.threshold(bg_sub_mask, 100, 200, cv2.THRESH_BINARY)

    # Edge detection to find dynamic-textured puddles
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    # People detection to ignore those edges
    # rects, weights = hog_classify.detectMultiScale(img, winStride=(1, 1), scale=1.01)
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(masked_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     print('Detected person at %s' % str((x, y)))
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    sigma = 0.33
    med = np.median(img)
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    edges = cv2.Canny(img, lower, upper)
    edges = cv2.dilate(edges, (2, 2))

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Background subtractor gives very accurate shadow detection
    shadows = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, np.ones((5, 5)))
    shadows = cv2.morphologyEx(shadows, cv2.MORPH_CLOSE, np.ones((5, 5)))

    # Foreground objects are not water
    # TODO Ignore the area around foreground objects (people), since they are definitely not water

    # Different edge detection techniques
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # sobelx = cv2.Sobel(hsv_img, cv2.CV_8U, 1, 0, ksize=5)
    # sobely = cv2.Sobel(hsv_img, cv2.CV_8U, 0, 1, ksize=5)
    # gradient_img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    # Give output
    # value_map = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 2]
    debug_img = stitch_together(img, edges, shadows, fg_mask, text='Contours: %d' % len(contours))
    # show_img(debug_img)
    # input('CONTINUE')

    return debug_img, contours

def apply_pipeline_to(folder: str, figures_folder: str, crop: tuple):
    # ----- Gradient Script -----
    bg_subtract = cv2.createBackgroundSubtractorMOG2(history=10, detectShadows=True)
    # os.makedirs(folder, exist_ok=True)
    # os.chdir(folder)
    img_pipeline_debug_output = []

    print('Training background subtractor...', end='')
    for path, img in get_images_from(folder, first_n=100):
        bg_subtract.apply(img)
    print('DONE')

    for i, (path, img) in enumerate(get_images_from(folder, exclude_containing='_gradient')):
        # Give full, blurred image to background subtractor so that it is not limited to one section
        img: np.ndarray = cv2.GaussianBlur(img, (7, 7), 0)
        fg_mask = bg_subtract.apply(img)

        # Crop, theshold, and reduce shadow noise
        fg_mask = apply_crop(fg_mask, crop)
        ret, fg_mask = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, (5, 5))

        # Crop and blur base image
        img = apply_crop(img, crop)
        img: np.ndarray = cv2.GaussianBlur(img, (5, 5), 0)

        # Compute the Laplacian gradient of this cropped image
        # gradient = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_16S, ksize=5)
        # gradient_abs: np.ndarray = np.absolute(gradient).astype(np.uint8)

        # Compute the entropy of the ballfield view to find its texture roughness
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        entropy_img = rank.entropy(img_gray, skimage.morphology.disk(5))
        # skimage.io.imshow(entropy_img)

        # Remove the parts of this image that are shadow or foreground
        entropy_img = cv2.bitwise_and(entropy_img, cv2.bitwise_not(fg_mask))

        gradient_file = 'gradient_' + os.path.basename(path)
        if i % 1000 == 0:
            print('Iteration: %d' % i)

        # Ballfield isolating
        # flat_img = img.reshape((img.shape[0] * img.shape[1], 3))
        # clt = KMeans()
        # clt.fit(flat_img)
        # show_img(clt.labels_.reshape((img.shape[0], img.shape[1])) * 50)

        gradient_sum = entropy_img.sum()
        img_pipeline_debug_output.append([i, os.path.basename(path), gradient_sum])
        # debug_img = stitch_together(img, gradient_abs, fg_mask, text=str(gradient_sum))
        # cv2.imwrite(gradient_file, debug_img)

    os.makedirs(figures_folder, exist_ok=True)

    # Histogram to show the most frequent gradient sum values
    plt.hist([img[2] for img in img_pipeline_debug_output], bins=20)
    plt.title('Histogram of the gradient counts of %s' % (folder,))
    plt.xlabel('Gradient count')
    plt.ylabel('Frequency')
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.savefig('%s/gradient_counts_hist.png' % (figures_folder,), dpi=200)
    plt.show()

    # Line graph to show gradient sum values over time
    plt.plot([img[2] for img in img_pipeline_debug_output])
    plt.title('Gradient count for each image over time')
    plt.xlabel('Frame #')
    plt.ylabel('Gradient count of that frame')
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.savefig('%s/gradient_over_time.png' % (figures_folder,), dpi=200)
    plt.show()

    # Dump results
    pickle.dump(img_pipeline_debug_output, open('%s/data.bin' % (figures_folder,), 'wb+'))
    return img_pipeline_debug_output

def parse_args():
    # TODO Make an argument parser that has intuitive options that labelers could use to assist them in labeling
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', dest='input_folder', help='Input folder to unprocessed image folder')
    ap.add_argument('-d', dest='debug_folder', help='Debug folder that will be used to store graphs, computed gradient '
                                                    'values, and a JSON file of tags')


if __name__ == '__main__':
    def item_select(array, indx):
        return np.array(array)[:, indx]

    flooded_data = item_select(
        apply_pipeline_to(FOLDER_FLOODED, 'flooded_water_output', BALLFIELD_TILTED_VIEW_CROP), 2)
    some_water_data = item_select(
        apply_pipeline_to(FOLDER_SOME_WATER, 'some_water_output', BALLFIELD_TILTED_VIEW_CROP), 2)
    no_water_data = item_select(
        apply_pipeline_to(FOLDER_NO_WATER, 'no_water_output', BALLFIELD_TILTED_VIEW_CROP), 2)

    no_water_grads = [int(item) for item in no_water_data]
    some_water_grads = [int(item) for item in some_water_data]
    flooded_grads = [int(item) for item in flooded_data]

    plt.boxplot([no_water_grads, some_water_grads, flooded_grads])
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()
    input('>')
