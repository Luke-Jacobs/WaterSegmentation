from water_detection import *
import os
from typing import Tuple

class Frame:

    CAT_FLOODED = 2
    CAT_SOME_WATER = 1
    CAT_NO_WATER = 0
    CAT_UNKNOWN = -1

    def __init__(self, frame_n, perspective, cat=None, data=None):
        self.tags = {"gray_entropy": [], "gradient_sum": [], "color_entropy": [],
                     "avg_value_variance": [], "avg_value": [], "avg_hue_variance": [],
                     "avg_sat_variance": [], "avg_sat": [], "avg_hue": []}  # Pipeline metrics
        self.cat = cat
        self.perspective = perspective
        self.frame_n = frame_n
        self.data = data

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['data']
        return d

    def __setstate__(self, state):
        self.__dict__ = state

    def sum_diff_with(self, other: 'Frame') -> int:
        assert self.data is not None and other.data is not None
        return cv2.absdiff(self.data, other.data).sum()

    def contains_water(self):
        if self.cat >= 0:
            return bool(self.cat)
        raise RuntimeError('Not labeled')

    def set_tag(self, key: str, value):
        self.tags[key] = value

    def get_tag(self, key: str):
        return self.tags[key]

    def compute_metrics(self):
        # Standardization
        mask_pixels = MASK_PIXELS[self.perspective]
        # Preprocessing
        img = cv2.GaussianBlur(self.data, (3, 3), 0)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Compute the area of this cropped section so that these metrics can be standardized
        # cropped_img_area = img.shape[0] * img.shape[1]

        # Compute the Laplacian gradient of this cropped image
        hsv_hue_img = hsv_img[:, :, 0]
        hsv_sat_img = hsv_img[:, :, 1]
        hsv_value_img = hsv_img[:, :, 2]
        # gradient = cv2.Laplacian(hsv_value_img, cv2.CV_16S, ksize=5)
        # gradient_abs_img: np.ndarray = np.absolute(gradient).astype(np.uint8)

        # Compute my metrics (Difference in value across a resized image)
        avg_value = hsv_value_img.sum() / mask_pixels
        avg_hue = hsv_hue_img.sum() / mask_pixels
        avg_sat = hsv_sat_img.sum() / mask_pixels
        avg_value_variance = np.absolute(hsv_value_img - avg_value).sum() / mask_pixels
        avg_hue_variance = np.absolute(hsv_hue_img - avg_hue).sum() / mask_pixels
        avg_sat_variance = np.absolute(hsv_sat_img - avg_sat).sum() / mask_pixels
        self.tags["avg_value"].append(avg_value)
        self.tags["avg_value_variance"].append(avg_value_variance)
        self.tags["avg_hue"].append(avg_hue)
        self.tags["avg_hue_variance"].append(avg_hue_variance)
        self.tags["avg_sat"].append(avg_sat)
        self.tags["avg_sat_variance"].append(avg_sat_variance)

        # Compute the entropy of the ballfield view to find its texture roughness
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        entropy_img = rank.entropy(img_gray, skimage.morphology.disk(5))
        gradient = cv2.Laplacian(hsv_value_img, cv2.CV_16S, ksize=5)
        gradient_abs_img = np.absolute(gradient).astype(np.uint8)
        self.tags["gray_entropy"].append(entropy_img.sum() / mask_pixels)
        self.tags["gradient_sum"].append(gradient_abs_img.sum() / mask_pixels)

    @staticmethod
    def load_from(folder: str, category: int, perspective: int) -> List['Frame']:
        img_nums = sorted([int(item[:-4]) for item in os.listdir(folder) if os.path.splitext(item)[-1] == '.jpg'])

        frames = []
        for frame_n in img_nums:
            img = cv2.imread(os.path.join(folder, '%d.jpg' % frame_n))
            img = apply_crop(apply_crop(img, MASKS[perspective][0]), MASKS[perspective][1])
            frames.append(Frame(frame_n, perspective, cat=category, data=img))
        return frames

def get_motion_amount_between_frames(frames: List[Frame], mask_pixels: int):
    ordered_frames = sorted(frames, key=lambda item: item.frame_n)

    # Iterate over all the Frame objects and tag each object with the absdiff sum between them
    # and the next frame
    diffs = []
    for i in range(len(ordered_frames)-1):
        this_frame = ordered_frames[i]
        next_frame = ordered_frames[i+1]

        if i % 1000 == 0:
            print(i)

        if abs(this_frame.frame_n - next_frame.frame_n) > 1:
            print('Frame jump between %d and %d' % (this_frame.frame_n, next_frame.frame_n))

        diff = cv2.absdiff(this_frame.data, next_frame.data).sum() / mask_pixels
        this_frame.tags['next_diff'] = diff
        diffs.append((this_frame.frame_n, diff, this_frame.cat))

def compute_every_metric(frames: List[Frame]):
    for frame in frames:
        frame.compute_metrics()

if __name__ == '__main__':
    MASKS = [
        [cv2.imread('/home/cc/images/tilted_view_field_binmask.png', cv2.IMREAD_GRAYSCALE), (476, 343, 754, 162)],
        [cv2.imread('/home/cc/images/ball_park_1_mask.png', cv2.IMREAD_GRAYSCALE), (537, 441, 534, 104)]
    ]
    MASK_PIXELS = [
        cv2.countNonZero(MASKS[0][0]),
        cv2.countNonZero(MASKS[1][0])
    ]

    # FLOODED_IMAGES = Frame.load_from('/home/cc/images/flooding', Frame.CAT_FLOODED, 0)
    # SOME_WATER_IMAGES = Frame.load_from('/home/cc/images/some_water', Frame.CAT_SOME_WATER, 0)
    # NO_WATER_IMAGES = Frame.load_from('/home/cc/images/no_water', Frame.CAT_NO_WATER, 0)
    # ALL_LABELED_IMGS = FLOODED_IMAGES + SOME_WATER_IMAGES + NO_WATER_IMAGES
    UNKNOWN_IMGS_1 = Frame.load_from('/home/cc/images/unknown/bp1_020A', Frame.CAT_UNKNOWN, 1)
    UNKNOWN_IMGS_2 = Frame.load_from('/home/cc/images/unknown/bp1_021A', Frame.CAT_UNKNOWN, 1)
    print('Loaded all the images')

    # get_motion_amount_between_frames(ALL_LABELED_IMGS, mask_pixels=MASK_PIXELS[0])
    get_motion_amount_between_frames(UNKNOWN_IMGS_1, mask_pixels=MASK_PIXELS[1])
    get_motion_amount_between_frames(UNKNOWN_IMGS_2, mask_pixels=MASK_PIXELS[1])
    # compute_every_metric(ALL_LABELED_IMGS)
    compute_every_metric(UNKNOWN_IMGS_1)
    compute_every_metric(UNKNOWN_IMGS_2)

    # ALL_IMGS = ALL_LABELED_IMGS + UNKNOWN_IMGS_1 + UNKNOWN_IMGS_2
    pickle.dump(UNKNOWN_IMGS_2 + UNKNOWN_IMGS_1, open('/home/cc/checkpoint2_unknowns.bin', 'wb+'))

    input('?')
