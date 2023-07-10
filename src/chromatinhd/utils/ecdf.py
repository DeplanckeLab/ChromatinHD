import numpy as np


def ecdf(data):
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    sorted_data = np.vstack([sorted_data - 1e-5, sorted_data]).T.flatten()
    y = np.vstack([np.hstack([[0], y[:-1]]), y]).T.flatten()
    return sorted_data, y


def area_under_ecdf(list1):
    x1, y1 = ecdf(list1)
    area = np.trapz(y1, x1)
    return area


def area_between_ecdfs(list1, list2):
    x1, y1 = ecdf(list1)
    x2, y2 = ecdf(list2)

    combined_x = np.concatenate(
        [[-0.001], np.sort(np.unique(np.concatenate((x1, x2))))]
    )

    y1_interp = np.interp(combined_x, x1, y1, left=0, right=1)
    y2_interp = np.interp(combined_x, x2, y2, left=0, right=1)

    ecdf_diff = y1_interp - y2_interp
    area = np.trapz(ecdf_diff, combined_x)
    return area


def relative_area_between_ecdfs(list1, list2):
    x1, y1 = ecdf(list1)
    area = area_between_ecdfs(list1, list2) / np.trapz(y1, x1)
    return area
