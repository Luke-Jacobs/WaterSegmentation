from backup.metrics_linear_classifier import *

# # Constants
# BALLFIELD_MASK = cv2.cvtColor(cv2.imread('/home/ljacobs/Argonne/water-data/ball_park_1/bin_mask_ballpark1.png'),
#                               cv2.COLOR_BGR2GRAY)
# FOLDER_1_021A = '/home/ljacobs/Argonne/water-data/ball_park_1/bp1_021A'
# FOLDER_0_016A = '/home/ljacobs/Argonne/water-data/ball_park_0/bp0_016A'
# EXAMPLES_FOLDER = '/home/ljacobs/Argonne/water-data/example_cases'
# ZOOMED_EXAMPLES_FOLDER = '/home/ljacobs/Argonne/water-data/example_cases/zoomed'
# FOLDER_FLOODED = '/home/ljacobs/Argonne/water-data/water_labeled/flooded'
# FOLDER_SOME_WATER = '/home/ljacobs/Argonne/water-data/water_labeled/some_water'
# FOLDER_NO_WATER = '/home/ljacobs/Argonne/water-data/water_labeled/no_water'
#
# # Crops: X, Y, Width, Height
# BALLFIELD_1_CROP = (319, 382, 794, 189)
# BALLFIELD_1_TIGHT_CROP = (537, 441, 534, 104)
# BALLFIELD_0_CROP = (384, 354, 870, 206)
# BALLFIELD_0_TIGHT_CROP = (574, 379, 581, 113)
BALLFIELD_TILTED_VIEW_CROP = (476, 343, 754, 162)

BALLFIELD_TILTED_FIELD_MASK = cv2.imread('/home/ljacobs/Argonne/water-data/water_labeled/tilted_view_field_binmask.png',
                                         cv2.IMREAD_GRAYSCALE)
BALLFIELD_0_VIEW_CROP = (577, 380, 567, 132)

# BALLFIELD_TILTED_FIELD_MASKCROP = (387, 312, 837, 188)
# BALLFIELD_1_MASK = cv2.imread('/home/ljacobs/Argonne/water-data/ball_park_1_mask.png', cv2.IMREAD_GRAYSCALE)
# BALLFIELD_1_MASK_CROP = (386, 403, 810, 209)
#
# # People detection initialization
# hog_classify = cv2.HOGDescriptor()
# hog_classify.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# plt.switch_backend('Qt5Agg')

def apply_crop(img: np.ndarray, crop) -> np.ndarray:
    if type(crop) == tuple:
        x, y, w, h = crop
        return img[y:y+h, x:x+w]
    else:
        return cv2.bitwise_and(img, img, mask=crop)

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
    # TODO Generalize the sort argument

    img_formats = ['.png', '.jpg']
    img_paths = [os.path.join(folder, img_path) for img_path in os.listdir(folder)
                 if (os.path.splitext(img_path)[-1] in img_formats) and img_path != 'mask.png']
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

def graph_binning_bar_chart(n_bins: int, water_data: List[int], no_water_data: List[int]):
    widest_distribution = water_data if max(water_data) > max(no_water_data) else no_water_data

    n, bin_edges, patches = plt.hist(widest_distribution, bins=n_bins)
    plt.clf()
    plt.title('Histogram describing the effectiveness of this metric (brown is no water and blue is water)')
    plt.xlabel('Metric value (in bins)')
    plt.ylabel('Percentage')
    for i in range(n_bins):
        water_in_this_bar = len([1 for item in water_data if bin_edges[i] <= item < bin_edges[i + 1]])
        no_water_in_this_bar = len([1 for item in no_water_data if bin_edges[i] <= item < bin_edges[i + 1]])
        items_in_bar = water_in_this_bar + no_water_in_this_bar
        plt.bar(i, water_in_this_bar / items_in_bar, color='blue', alpha=0.5, label='Water')
        plt.bar(i, no_water_in_this_bar / items_in_bar, color='brown', alpha=0.5,
                bottom=(water_in_this_bar / items_in_bar), label='No water')
    plt.show()

def parse_args():
    # TODO Make an argument parser that has intuitive options that labelers could use to bin images accurately

    ap = argparse.ArgumentParser()
    sub_ap = ap.add_subparsers(title='Command', dest='command',
                               help='Main command, either "process", "train", or "bin"')

    process_ap = sub_ap.add_parser('process', help='Scan through the input folder and compute various metrics for each'
                                                   'input image, outputting these metrics to a specified file')
    process_ap.add_argument('-i', dest='input_folder', help='Input folder of images to process')
    # process_ap.add_argument('-o', dest='output_file', help='Output file to store a pickled dictionary containing image'
    #                                                        'metrics')
    process_ap.add_argument('-m', dest='field_mask_file', help='Path to field mask file')

    train_ap = sub_ap.add_parser('train', help='Train a simple linear SVM (classifier) by feeding it examples of images '
                                               'of water and images without water')
    train_ap.add_argument('--water', required=True, help='Folder of processed metrics of water images')
    train_ap.add_argument('--dry', required=True, help='Folder of processed metrics of dry images')
    train_ap.add_argument('-o', dest='output_file', required=True, help='Output file that will contain a trained '
                                                                        'classifier')

    bin_ap = sub_ap.add_parser('bin', help='Bin images using an input classifier and output them to their respective'
                                           'folders, "water" and "dry"')
    bin_ap.add_argument('--model', required=True, help='Path to the classifier model')
    bin_ap.add_argument('--source', required=True, help='Path to the source folder of the unclassified images')
    bin_ap.add_argument('--water', required=True, help='Path to a folder where water images will be placed')
    bin_ap.add_argument('--dry', required=True, help='Path to a folder where images without water will be placed')

    return ap.parse_args()

def process_command(input_folder, field_mask_file):
    metrics = {
        # "gray_entropy": [], "gradient_sum": [], "color_entropy": [],
               "avg_value_variance": [], "avg_value": [], "avg_hue_variance": [], "avg_sat_variance": [], "avg_sat": [],
               "avg_hue": []}  # Pipeline metrics

    if "metrics.bin" in os.listdir(input_folder):
        char = input('[Warning] Do you want to process (%s) and overwrite the existing metrics.bin file? [Y/n]'
                     % input_folder)
        if char not in ['Y', '']:
            return

    print('Select a crop around your mask')
    ret, field_mask = cv2.threshold(cv2.imread(field_mask_file, cv2.IMREAD_GRAYSCALE), 200, 255, cv2.THRESH_BINARY)
    roi = cv2.selectROI('Select field crop', field_mask)
    cv2.destroyAllWindows()

    # Standardization
    mask_pixels = cv2.countNonZero(field_mask)

    total_imgs = len(os.listdir(input_folder))
    for i, (path, img) in enumerate(get_images_from(input_folder)):
        # Crop and blur base image
        img = apply_crop(img, field_mask)
        img = apply_crop(img, roi)

        # Preprocessing
        img = cv2.GaussianBlur(img, (3, 3), 0)
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
        metrics["avg_value"].append(avg_value)
        metrics["avg_value_variance"].append(avg_value_variance)
        metrics["avg_hue"].append(avg_hue)
        metrics["avg_hue_variance"].append(avg_hue_variance)
        metrics["avg_sat"].append(avg_sat)
        metrics["avg_sat_variance"].append(avg_sat_variance)

        # Compute the entropy of the ballfield view to find its texture roughness
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # entropy_img = rank.entropy(img_gray, skimage.morphology.disk(5))
        # metrics["gray_entropy"].append(entropy_img.sum() / mask_pixels)
        # metrics["gradient_sum"].append(gradient_abs_img.sum() / mask_pixels)

        if i % 1000 == 0:
            print('Iteration: %d/%d' % (i, total_imgs))

    # Dump results
    results = {'metrics': metrics, 'input_folder': input_folder}
    print('Dumping the metrics.bin into %s' % input_folder)
    try:
        fp = open(os.path.join(input_folder, 'metrics.bin'), 'wb+')
    except IOError:
        alternative_path = input('Unable to save into (%s), where else should metrics.bin be saved? Enter path:')
        os.makedirs(alternative_path, exist_ok=True)
        fp = open(os.path.join(alternative_path, 'metrics.bin'), 'wb+')
    pickle.dump(results, fp)

    return results

def train_command(water_folder, dry_folder, output_file):
    water_metrics = pickle.load(open(os.path.join(water_folder, 'metrics.bin'), 'rb'))
    dry_metrics = pickle.load(open(os.path.join(dry_folder, 'metrics.bin'), 'rb'))
    water_pts = dict_to_points(water_metrics['metrics'])
    dry_pts = dict_to_points(dry_metrics['metrics'])

    classifier = train_classifier(water_pts, dry_pts)
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pickle.dump(classifier, open(output_file, 'wb+'))

    return classifier

def bin_command(model_path, source_path, water_path, dry_path, copy_images=True):
    if not os.path.exists(water_path):
        os.makedirs(water_path, exist_ok=True)
    if not os.path.exists(dry_path):
        os.makedirs(dry_path, exist_ok=True)
    assert os.path.isdir(source_path) and os.path.isdir(water_path)

    cmd = 'cp' if copy_images else 'mv'
    classifier = pickle.load(open(model_path, 'rb'))
    unclassified_pts = dict_to_points(pickle.load(open(os.path.join(source_path, 'metrics.bin'), 'rb'))['metrics'])

    labeled_pts = classifier.predict(unclassified_pts)

    # TODO Make sure that labels get matched with the right images (assuming that the get_images function will be in order)
    for i, img_path in enumerate(get_images_from(source_path, no_load=True)):
        output_path = water_path if labeled_pts[i] == 1 else dry_path
        os.system('%s %s %s' % (cmd, img_path, output_path))

def get_motion_amount_between_frames():
    def filt(ar):
        return [int(item[:-4]) for item in ar if item[-4:] == '.jpg']

    flooded_frames = np.array(filt(os.listdir('../water-data/water_labeled/water/flooded')))
    some_water_frames = np.array(filt(os.listdir('../water-data/water_labeled/water/some_water')))
    nowater_frames = np.array(filt(os.listdir('../water-data/water_labeled/no_water/')))

    all_frames_n = np.vstack((
        np.dstack((flooded_frames, np.full((len(flooded_frames),), 0)))[0],
        np.dstack((some_water_frames, np.full((len(some_water_frames),), 1)))[0],
        np.dstack((nowater_frames, np.full((len(nowater_frames),), 2)))[0],
    ))

    def get_img(n, id):
        path = '../water-data/water_labeled/%s/%d.jpg' % (['water/flooded', 'water/some_water', 'no_water'][id],
                                                          n)
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    for i, (frame_n, category_id) in enumerate(all_frames_n):
        if i == 0:
            continue
        img_prev = apply_crop(get_img(*all_frames_n[i-1]), BALLFIELD_TILTED_FIELD_MASK)
        img_prev = apply_crop(img_prev, BALLFIELD_TILTED_VIEW_CROP)
        img_current = get_img(frame_n, category_id)
        img_current = apply_crop(img_current, BALLFIELD_TILTED_FIELD_MASK)
        img_current = apply_crop(img_current, BALLFIELD_TILTED_VIEW_CROP)
        show_img(stitch_together(cv2.absdiff(img_prev, img_current), img_current))
        input('WAIT')

    # plt.figure()
    # plt.plot([item[0] for item in diffs], color=[['blue', 'purple', 'brown'][item[-1]] for item in diffs])
    # plt.show()
    #
    # return diffs

if __name__ == '__main__':
    # [no_water, some_water, flooded] = [pickle.load(
    #     open('/home/ljacobs/Argonne/water-pipeline/%s_water_output/results_metrics_standardized.bin' % item,
    #          'rb'))
    #     for item in ['flooded', 'some', 'no']]
    # no_water_pts = dict_to_points(no_water)
    # some_water_pts = dict_to_points(some_water)
    # flooded_pts = dict_to_points(flooded)
    # water_pts = np.vstack((some_water_pts, flooded_pts))
    # flooding_classifier = train_classifier(water_pts, no_water_pts)

    args = parse_args()

    if args.command == 'process':
        process_command(args.input_folder, args.field_mask_file)
    elif args.command == 'train':
        train_command(args.water, args.dry, args.output_file)
    elif args.command == 'bin':
        bin_command(args.model, args.source, args.water, args.dry)
    else:
        raise RuntimeError('Unknown command (%s)' % args.command)

    # flooded_results = apply_pipeline_to(FOLDER_FLOODED, 'flooded_water_output',
    #                                     crop_tuple=BALLFIELD_TILTED_FIELD_MASKCROP,
    #                                     crop_mask=BALLFIELD_TILTED_FIELD_MASK)
    # some_water_results = apply_pipeline_to(FOLDER_SOME_WATER, 'some_water_output',
    #                                        crop_tuple=BALLFIELD_TILTED_FIELD_MASKCROP,
    #                                        crop_mask=BALLFIELD_TILTED_FIELD_MASK)
    # no_water_results = apply_pipeline_to(FOLDER_NO_WATER, 'no_water_output',
    #                                      crop_tuple=BALLFIELD_TILTED_FIELD_MASKCROP,
    #                                      crop_mask=BALLFIELD_TILTED_FIELD_MASK)
    # input('WAIT')

    # Load a new set of images to inference on and test the algorithm's accuracy manually
    # test_batch = apply_pipeline_to(FOLDER_1_021A, 'cam_1_021A', crop_tuple=BALLFIELD_1_MASK_CROP,
    #                                crop_mask=BALLFIELD_1_MASK)

    # test_batch_labels = flooding_classifier.predict(test_batch_pts)

    # for metric in range(len(test_batch)):
    #     no_water_metrics = np.array(no_water_results[metric])
    #     some_water_metrics = np.array(some_water_results[metric])
    #     flooded_metrics = np.array(flooded_results[metric])
    #
    #     plt.title('Metric Accuracy (%d)' % metric)
    #     plt.hist(no_water_metrics, bins=20, label='No flooding')
    #     plt.hist(some_water_metrics, bins=20, label='Partial flooding')
    #     plt.hist(flooded_metrics, bins=20, label='Heavy flooding')
    #     plt.legend()
    #     fig = plt.gcf()
    #     fig.set_size_inches(8, 8)
    #     plt.show()
    #
    #     plt.boxplot([no_water_metrics, some_water_metrics, flooded_metrics])
    #     fig = plt.gcf()
    #     fig.set_size_inches(8, 8)
    #     plt.show()

    input('>')

