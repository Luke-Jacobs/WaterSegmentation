import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['USE_SIMPLE_THREADED_LEVEL3'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
from backup.water_detection import *
from sklearn.svm import SVC
from multiprocessing import set_start_method
import cv2

set_start_method("spawn")
cv2.setNumThreads(4)
N_PROCESSES = 10
# import cupy as cp
plt.ioff()


def get_imgs_from(folder: str, sort=True):
    out = [os.path.join(folder, file) for file in os.listdir(folder) if file[-4:] in ['.png', '.jpg']]
    if sort:
        out = sorted(out, key=lambda item: int(os.path.basename(item)[:-4]))
    return out

def build_spatial_features():
    # Load LBP descriptors
    sources = {
        'W_flooded': lbp_descriptors_from_folder('/home/ljacobs/Argonne/water_data/water_labeled/water/flooded',
                                                 crop=BALLFIELD_TILTED_VIEW_CROP),
        'N_no_water': lbp_descriptors_from_folder('/home/ljacobs/Argonne/water_data/water_labeled/no_water',
                                                  crop=BALLFIELD_TILTED_VIEW_CROP),
        'W_flooded2': lbp_descriptors_from_folder('/home/ljacobs/Argonne/water_data/ball_park_0/bp0_016A/flooded',
                                                  crop=BALLFIELD_0_VIEW_CROP),
        'N_no_water2': lbp_descriptors_from_folder('/home/ljacobs/Argonne/water_data/ball_park_0/bp0_016A/no_water',
                                                   crop=BALLFIELD_0_VIEW_CROP),
        'W_some_water2': lbp_descriptors_from_folder('/home/ljacobs/Argonne/water_data/ball_park_0/bp0_016A/some_water',
                                                     crop=BALLFIELD_0_VIEW_CROP),
        'U_3': lbp_descriptors_from_folder('/home/ljacobs/Argonne/water_data/ball_park_0/bp0_030A',
                                           crop=BALLFIELD_0_VIEW_CROP)
    }

    all_x = np.vstack([value[0] for value in tuple(sources.values())])
    labels = tuple([np.ones((len(sources[key][0]), 1)) if key[0] == 'W' else np.zeros((len(sources[key][0]), 1))
                    for key in list(sources.keys())])  # A tuple of 1-D ndarray's of varying lengths
    all_y = np.vstack(labels)
    dataset = np.hstack((all_x, all_y))

    ninth = int(len(dataset)//10*9)
    train = dataset[:ninth]
    test = dataset[ninth:]

    classify = SVC()
    classify.fit(train[:, :-1], train[:, -1])

    predictions = classify.predict(test[:, :-1])
    correct = predictions == test[:, -1]
    print('# of test points: %d, percent correct: %02f' % (len(correct), correct.sum() / len(correct)))

    pca_fit = PCA(n_components=2)
    pca_fit.fit(dataset[:, :-1])

    for key in sources:
        group_pts = pca_fit.transform(sources[key])
        plt.scatter(group_pts[:, 0], group_pts[:, 1], label=key)
    plt.legend()
    plt.show()

    input('WAIT')

def crop_slice_through_vid(img_ar: np.ndarray, crop: tuple, width, height):
    x, y = crop
    return img_ar[:, y:y+height, x:x+width]

def crop_f(img: np.ndarray, crop) -> np.ndarray:
    if crop is None:
        return img
    x, y, w, h = crop
    return img[y:y + h, x:x + w]

def compute_fft_array(img_array, boxsize=5):
    cutoff_half = int(np.ceil(img_array.shape[0]/2))
    # fft_in = img_array - mode(img_array, axis=0).mode

    # An averaging kernel (low-pass)
    # kernel1 = np.ones((boxsize, boxsize)) / (boxsize * boxsize)
    # Another recommended kernel
    # kernel2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Apply kernels
    # for i in range(len(fft_in)):
    #     fft_in[i] = cv2.filter2D(fft_in[i], -1, kernel1)
    #     fft_in[i] = cv2.filter2D(fft_in[i], -1, kernel2)

    # fft_img_ar[:, y, x] = np.fft.fft(img_array[:, y, x, :] - img_array[:, y, x, :].mean(axis=0), axis=0)
    fft_out = np.zeros((cutoff_half, img_array.shape[1], img_array.shape[2]))
    for y in range(img_array.shape[1]):
        for x in range(img_array.shape[2]):
            fft_pix = np.fft.fftshift(np.fft.fft(img_array[:, y, x]))
            fft_out[:, y, x] = abs(fft_pix)[:cutoff_half]

    # APPEARS TO NOT WORK
    # fft_out = abs(np.fft.fft2(img_array))[:cutoff_half]

    peaks_img = np.argmax(fft_out, axis=0)  # Computes the dominant frequency for each pixel
    plt.imshow(peaks_img)
    plt.show()

    return fft_out

def get_intensity_img_of_video_at_freq(fft_ar, freq_low, freq_high=None):
    return fft_ar[freq_low:freq_high, :, :, :].sum(axis=0)

def show_avg_fft_waveform_at_locs(fft_ar: np.ndarray, locs: List[Tuple[tuple, str]], width: int, height: int):
    """
    Computes the average FFT waveform of a certain rectangular view-slice of a video and graphs those waveforms on top
    of each other so that patterns can become visible.

    @:param fft_ar: An ndarray of shape (# of frames, height, width). This implies that it takes in only grayscale
                    images
    @:param locs: A list of (x,y) coordinate tuples
    """

    # Allocate space for <N locs> FFT waveforms
    averaged_waveforms = np.zeros((len(locs), fft_ar.shape[0]))
    for i, loc in enumerate(locs):
        (x, y), label = loc
        this_loc_waveforms = fft_ar[:, y:y + height, x:x + width]
        averaged_waveforms[i] = this_loc_waveforms.mean(axis=(1, 2))

    plt.title('DFT Waveform Describing the Predominant Pixel-change Frequencies For 4 Different Regions In Video')
    plt.xlabel('DFT Frequency')
    plt.ylabel('Intensity')
    for i in range(len(averaged_waveforms)):
        plt.plot(abs(averaged_waveforms[i]) - abs(averaged_waveforms[i]).mean(), label=locs[i][-1])
    plt.legend()
    plt.show()

def array_from_img_folder(folder: str):
    """
    Returns a numpy data array of shape:
        (n_of_imgs_in_folder, images_y, images_x, images_color_channels)
    Works on a folder with images in the format of <number>.jpg
    """
    frames = sorted([file for file in os.listdir(folder) if file.find('.png') != -1],
                    key=lambda item: int(item[:-4]))
    first_frame = cv2.imread(os.path.join(folder, frames[0]))
    height, width, channels = first_frame.shape
    SCALING = 0.5
    img_array = np.zeros((len(frames), int(height * SCALING), int(width * SCALING)))

    for file in frames:
        # img = cv2.cvtColor(cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2HSV)
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        dim = (int(img.shape[1] * SCALING), int(img.shape[0] * SCALING))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_array[int(file[:-4]) - 1] = img

    return img_array

def highlight_water_segments_in_video(img_ar: np.ndarray):
    water_mask = cv2.imread('./mask_vid_puddles_bin.png', cv2.IMREAD_GRAYSCALE)
    water_mask = (water_mask > 50).reshape(water_mask.shape[0] * water_mask.shape[1])

    fft_ar_flat = compute_fft_array(img_ar).reshape((64, -1))
    fft_w = fft_ar_flat[:, water_mask].T
    np.random.shuffle(fft_w)
    fft_n = fft_ar_flat[:, ~water_mask].T
    np.random.shuffle(fft_n)

    plt.bar(np.arange(64), fft_w[:, :1000].mean(axis=0), alpha=0.5, label='Water FFT Spectrum')
    plt.bar(np.arange(64), fft_n[:, :1000].mean(axis=0), alpha=0.5, label='No water FFT Spectrum')
    plt.show()

    input('WAIT')


VIDEO_FILE = "/home/ljacobs/Argonne/water_data/twitter_rain/street_webcam_clip.mp4"
FOLDER_SPLIT_IMGS = "/home/ljacobs/Argonne/water_data/twitter_rain/street_webcam_split"
FOLDER_SPLIT_IMGS_FULL = "/home/ljacobs/Argonne/water_data/twitter_rain/street_webcam_split_full"
CROP_REGION_1 = (835, 430, 280, 214)


def compute_on_chameleon():
    FOLDER_WATER_VIDEO_DATASET = '/home/cc/ww/VideoWaterDatabase/'
    VIDEOS_FOLDER = FOLDER_WATER_VIDEO_DATASET + 'videos'
    MASKS_FOLDER = FOLDER_WATER_VIDEO_DATASET + 'masks'

    WATER_VIDEO_DATASET = TrainingSet.init_many(VIDEOS_FOLDER, MASKS_FOLDER)

    def quick_stats(name: str, ar: np.ndarray) -> str:
        fourth = len(ar) // 4
        ar.sort()
        return '----- %s -----\nMin: %f\n1st quarter: %f\nMedian: %f\n3rd quarter: %f\nMax: %f\nStd: %f\n' % \
               (name, ar.min(), ar[fourth], float(np.median(ar)), ar[int(fourth*3)], ar.max(), ar.std())

    def compare_sat_val_ratio():
        ...

    WATER_VIDEO_DATASET.train_glcm_classifier()
    WATER_VIDEO_DATASET.train_gmm_classifier(visual=True)
    WATER_VIDEO_DATASET.visualize_glcm_classifier()
    # compare_sat_val_ratio()

    input('WAIT')


if __name__ == '__main__':
    # build_spatial_features()
    # show_avg_fft_waveform_at_locs(fft_ar, [PUDDLE_1, PUDDLE_2, MOTION_1, MOTION_2], 70, 50)
    compute_on_chameleon()

