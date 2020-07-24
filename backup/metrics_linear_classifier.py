from water_detection import *
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage import io
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import itertools
from sklearn import svm


def quick_stats(*metrics_distributions: np.ndarray):
    n_metrics = metrics_distributions[0].shape[1]
    for metric_n in range(n_metrics):
        print('------ Metric (%d) ------' % metric_n)
        for i, dist in enumerate(metrics_distributions):
            dist_slice = dist[:, metric_n]
            print('Dist %d | Mean: %f, Max: %f, Min: %f' % (i, dist_slice.mean(), dist_slice.max(), dist_slice.min()))

def dict_to_points(metrics_dict: dict):
    # Quick filter in case of entries that will derail the list to ndarray conversion
    for key in metrics_dict.keys():
        if type(metrics_dict[key]) != list:
            print('[Warning] Deleting key %s from this dictionary because it is not a list' % key)
            del metrics_dict[key]

    pts = np.array(list(metrics_dict.values())).transpose()
    return pts

def make_label_set(water_pts: np.ndarray, nowater_pts: np.ndarray):
    return np.vstack((np.hstack((water_pts, np.ones((len(water_pts), 1)))),
                     np.hstack((nowater_pts, np.zeros((len(nowater_pts), 1))))))

def load_pts(path: str) -> np.ndarray:
    return dict_to_points(pickle.load(open(path, 'rb'))['metrics'])

def train_classifier(water_pts: np.ndarray, no_water_pts: np.ndarray) -> svm.SVC:
    classify = svm.SVC(probability=True)
    water_pts = np.hstack((water_pts, np.ones((len(water_pts), 1))))
    no_water_pts = np.hstack((no_water_pts, np.zeros((len(no_water_pts), 1))))
    dataset = np.vstack((water_pts, no_water_pts))
    np.random.shuffle(dataset)

    train = dataset[:int(dataset.shape[0])]

    train_x = train[:, :-1]
    train_y = train[:, -1]

    classify.fit(train_x, train_y)
    return classify

def graph_groups_by_metrics(flooding: dict, some_water: dict, no_water: dict):
    print('Using PCA to graph %d metrics' % len(flooding))
    flooding_pts = dict_to_points(flooding)
    no_water_pts = dict_to_points(no_water)
    partial_water_pts = dict_to_points(some_water)
    pca_fit = PCA(n_components=2)
    pca_fit.fit(np.vstack((flooding_pts, no_water_pts, partial_water_pts)))
    new_flooding_pts = pca_fit.transform(flooding_pts)
    new_no_water_pts = pca_fit.transform(no_water_pts)
    new_partial_water_pts = pca_fit.transform(partial_water_pts)
    np.random.shuffle(new_flooding_pts)
    np.random.shuffle(no_water_pts)
    np.random.shuffle(partial_water_pts)
    plt.scatter(new_flooding_pts[:, 0], new_flooding_pts[:, 1], c='blue')
    plt.scatter(new_no_water_pts[:, 0], new_no_water_pts[:, 1], c='brown')
    plt.scatter(new_partial_water_pts[:, 0], new_partial_water_pts[:, 1], c='purple')
    plt.show()

def test_combinations_of_metric_pairs(flooding: dict, some_water: dict, no_water: dict):
    flooding_pts = np.array(list(flooding.values())).transpose()
    no_water_pts = np.array(list(no_water.values())).transpose()
    partial_water_pts = np.array(list(some_water.values())).transpose()
    np.random.shuffle(no_water_pts)
    np.random.shuffle(partial_water_pts)
    np.random.shuffle(flooding_pts)

    for x, y in itertools.combinations(range(flooding_pts.shape[1]), 2):
        plt.scatter(flooding_pts[:, x], flooding_pts[:, y], color='blue')
        plt.scatter(partial_water_pts[:, x], partial_water_pts[:, y], color='purple')
        plt.scatter(no_water_pts[:, x], no_water_pts[:, y], color='brown')
        plt.show()

if __name__ == '__main__':
    [no_water, some_water, flooded] = [pickle.load(
        open('/home/ljacobs/Argonne/water-pipeline/%s_water_output/results_many_metrics_effectiveness.bin' % item, 'rb'))
                                       for item in ['flooded', 'some', 'no']]
    # graph_groups_by_metrics(flooded, some_water, no_water)
    # test_combinations_of_metric_pairs(flooded, some_water, no_water)
    no_water_pts = dict_to_points(no_water)
    some_water_pts = dict_to_points(some_water)
    flooded_pts = dict_to_points(flooded)

    train_classifier(np.vstack((some_water_pts, flooded_pts)), no_water_pts)

