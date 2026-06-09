import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def get_affine_transformation(points_in, points_out):
    """
    TODO for students:
    Estimate the affine transformation matrix mapped from points_in to points_out.
    Transform the input points to homogenous coordinates and solve the least-squares problem.
    """
    # transform to homogenous coordinates
    points_in = np.asarray(points_in)
    points_out = np.asarray(points_out)

    # (x, y) -> (x, y, 1)
    ones = np.ones((points_in.shape[0], 1))
    points_in_h = np.hstack((points_in, ones))

    # solve the least-squares problem A.T@Ax = A.Tb
    transformation, _, _, _ = np.linalg.lstsq(points_in_h, points_out, rcond=None)

    # Convert from 2x3-style affine parameters to a 3x3 homogeneous matrix
    matrix = np.eye(3)
    matrix[:2, :] = transformation.T

    return matrix


def transform_points(points, matrix):
    """
    TODO for students:
    Given a set of 2D points, apply the transformation matrix.
    Return the new (x, y) coordinates.
    """
    # homogneous coordinates
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])

    # transform the points
    transformed_h = points_h @ matrix.T
    transformed = transformed_h[:, :2] / transformed_h[:, 2:3]

    return transformed

def backwards_mapping(image, output_shape, transformation, background=0):
    """
    TODO for students:
    Apply a backward mapping transformation to the input image.
    """
    # create the points of the new image
    h_out, w_out = output_shape[:2]
    h_in, w_in = image.shape[:2]

    yy, xx = np.indices((h_out, w_out))
    output_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # transform the points into the original image
    input_points = transform_points(output_points, transformation)

    input_x = np.round(input_points[:, 0]).astype(int)
    input_y = np.round(input_points[:, 1]).astype(int)

    output_x = output_points[:, 0].astype(int)
    output_y = output_points[:, 1].astype(int)

    # remove all points, that land outside the original image
    valid = (
        (input_x >= 0) & (input_x < w_in) &
        (input_y >= 0) & (input_y < h_in)
    )

    # write the pixel values at the correct location
    output_image = np.zeros(output_shape, dtype=image.dtype)

    if image.ndim == 2:
        output_image[output_y[valid], output_x[valid]] = image[input_y[valid], input_x[valid]]
    else:
        output_image[output_y[valid], output_x[valid], :] = image[input_y[valid], input_x[valid], :]

    return output_image

def downsample_bilinear(img, factor):
    """
    TODO for students:
    Downsample a grayscale image by the given factor using bilinear interpolation.
    """
    h, w = img.shape

    new_h = int(h / factor)
    new_w = int(w / factor)

    downsampled = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    return downsampled
    """
    h, w = img.shape

    new_h = int(h // factor)
    new_w = int(w // factor)

    output = np.zeros((new_h, new_w), dtype=img.dtype)

    for y_out in range(new_h):
        for x_out in range(new_w):
            x = (x_out + 0.5) * factor - 0.5
            y = (y_out + 0.5) * factor - 0.5

            x0 = int(np.floor(x))
            y0 = int(np.floor(y))

            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)

            x0 = max(x0, 0)
            y0 = max(y0, 0)

            dx = x - x0
            dy = y - y0

            top = (1 - dx) * img[y0, x0] + dx * img[y0, x1]
            bottom = (1 - dx) * img[y1, x0] + dx * img[y1, x1]

            value = (1 - dy) * top + dy * bottom

            output[y_out, x_out] = value

    return output
    """

def histogram_equalization(img):
    """
    TODO for students:
    Compute histogram equalization mapping from an image.
    Return the equalized image.
    """
    img = np.asarray(img)

    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])

    cdf = hist.cumsum()

    cdf_masked = np.ma.masked_equal(cdf, 0)

    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()

    cdf_equalized = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)

    cdf_equalized = np.ma.filled(cdf_equalized, 0).astype(np.uint8)

    equalized_img = cdf_equalized[img]

    return equalized_img

def convolve2d(img, kernel):
    """
    TODO: Apply a 2D convolution (without padding, assumes odd kernel).
    
    Parameters:
        img (np.ndarray): Grayscale image.
        kernel (np.ndarray): 2D filter kernel.
    
    Returns:
        np.ndarray: Convolved image (same size as input, zero-padded).
    """
    img = np.asarray(img)
    kernel = np.asarray(kernel, dtype=np.float64)

    kernel = np.flipud(np.fliplr(kernel))

    output = cv2.filter2D(
        src=img,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT
    )

    return output

def sobel_filter(img):
    """
    TODO: Apply Sobel edge detection filter (magnitude of gradients).
    
    Parameters:
        img (np.ndarray): Grayscale image.
    
    Returns:
        np.ndarray: Sobel gradient magnitude image.
    """
    img = np.asarray(img, dtype=np.float64)

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    gx = convolve2d(img, sobel_x)
    gy = convolve2d(img, sobel_y)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255

    return magnitude.astype(np.uint8)

def filter_wrapper_fn(img, mode, kernel_size=3, sigma=1.0):
    """
    TODO Apply one of the following filters to the image: mean, median, gaussian, sobel.

    Parameters:
        img (np.ndarray): Grayscale image.
        mode (str): One of "mean", "median", "gaussian", "sobel"
        kernel_size (int): Kernel size (must be odd)
        sigma (float): Gaussian std dev (only used for gaussian)
    
    Returns:
        np.ndarray: Filtered image.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    if mode == "mean":
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
        kernel = kernel / kernel.sum()
        filtered_img = convolve2d(img, kernel)

    elif mode == "median":
        filtered_img = cv2.medianBlur(img.astype(np.uint8), kernel_size)

    elif mode == "gaussian":
        filtered_img = gaussian_filter(img, sigma=sigma)

    elif mode == "sobel":
        filtered_img = sobel_filter(img)

    else:
        raise ValueError("mode must be one of: mean, median, gaussian, sobel")

    return filtered_img

def get_keypoints(image, filtering=True, sigma=3):
    """
    Extracts keypoints from the image. 
    You might want to optionally smooth the image with a gaussian filter first.
    """
    if filtering:
        image = gaussian_filter(image, sigma=sigma)

    image = np.asarray(image)

    if image.dtype != np.uint8:
        image_min = image.min()
        image_max = image.max()
        image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors

def intersect2d(array1, array2):
    """ Helper to get intersection of row matches """
    test = array1[:, None] == array2
    return array2[np.all(test.mean(0) > 0, axis=1)]

def matching(descriptors_1, descriptors_2, max_ratio=0.7, cross_checking=True):
    """
    TODO:
    Matches the descriptors against each other.
    Returns the best match for each descriptor, if it is significant.
    The significance is defined by the max_ratio: distance_1 / distance_2 < max_ratio.
    Optional cross-checking of matches.
    """
    distances = cdist(descriptors_1, descriptors_2)

    matches1 = np.argsort(distances, axis=1)[:, :2]

    best_dist1 = distances[np.arange(distances.shape[0]), matches1[:, 0]]
    second_best_dist1 = distances[np.arange(distances.shape[0]), matches1[:, 1]]

    mask1 = best_dist1 / second_best_dist1 < max_ratio

    final_matches = np.stack(
        (np.arange(descriptors_1.shape[0])[mask1], matches1[mask1, 0])
    ).T

    distances2 = cdist(descriptors_2, descriptors_1)

    matches2 = np.argsort(distances2, axis=1)[:, :2]

    best_dist2 = distances2[np.arange(distances2.shape[0]), matches2[:, 0]]
    second_best_dist2 = distances2[np.arange(distances2.shape[0]), matches2[:, 1]]

    mask2 = best_dist2 / second_best_dist2 < max_ratio

    # cross_checking
    if cross_checking:
        final_matches2 = np.stack(
            (np.arange(descriptors_2.shape[0])[mask2], matches2[mask2, 0]),
        ).T
    if cross_checking:
        # return the intersection of the two matches arrays (invert final_matches2 to point in the same direction)
        final_matches = intersect2d(final_matches, final_matches2[:, ::-1])
    return final_matches

def ransac(points_in, points_out, matches, percentage_outliers=0.5, probability=0.99, cutoff=20, k=3):
    """
    TODO:
    Implement Random Sample Consensus to predict a robust affine model on a set with outliers.
    """
    points_in = np.asarray(points_in)
    points_out = np.asarray(points_out)
    matches = np.asarray(matches)

    n_matches = matches.shape[0]

    if n_matches < k:
        return matches, 0

    inlier_best = np.zeros(n_matches, dtype=bool)
    support_best = 0

    p_inlier = 1 - percentage_outliers
    num_iterations = np.log(1 - probability) / np.log(1 - p_inlier ** k)
    num_iterations = int(np.ceil(num_iterations))

    for _ in range(num_iterations):
        sample_indices = np.random.choice(n_matches, size=k, replace=False)
        sample_matches = matches[sample_indices]

        sample_points_in = points_in[sample_matches[:, 0]]
        sample_points_out = points_out[sample_matches[:, 1]]

        try:
            transformation = get_affine_transformation(sample_points_in, sample_points_out)
        except:
            continue

        matched_points_in = points_in[matches[:, 0]]
        matched_points_out = points_out[matches[:, 1]]

        transformed_points = transform_points(matched_points_in, transformation)

        errors = np.linalg.norm(transformed_points - matched_points_out, axis=1)

        inlier_current = errors < cutoff
        support_current = np.sum(inlier_current)

        if support_current > support_best:
            support_best = support_current
            inlier_best = inlier_current

    # return the inliers
    return matches[inlier_best], support_best

def helper_plot_fn(img1, img2, transformed_img2):
    """ Plotting helper """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title("Source Image")
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title("Destination Image")
    axes[2].imshow(transformed_img2, cmap='gray')
    axes[2].set_title("Transformed Image")
    plt.tight_layout()
    plt.show()
