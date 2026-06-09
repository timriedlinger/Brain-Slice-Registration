# Brain Slice Registration and 3D Reconstruction

This project was developed as part of a university Computer Vision course.
The goal was to build an image registration pipeline that aligns 2D brain slice images and combines them into a simple 3D volume.

The project uses classical computer vision techniques such as affine transformations, image filtering, SIFT feature detection, descriptor matching, RANSAC, and backward image warping.

---

## What the Project Does

The input consists of multiple 2D brain slice images. These slices are visually similar, but they are not perfectly aligned. Some slices are shifted, rotated, or slightly sheared compared to neighboring slices.

The project implements a pipeline to:

1. Load and inspect 2D brain slice images.
2. Manually align two slices using selected landmark points.
3. Apply image preprocessing and filtering operations.
4. Automatically detect and match image features using SIFT.
5. Remove incorrect matches using RANSAC.
6. Estimate affine transformations between neighboring slices.
7. Warp the images into a common coordinate system.
8. Stack the aligned slices into a simple 3D volume.

---

## What I Learned

This project helped me understand how classical computer vision methods can be combined into a complete registration pipeline.

Key topics I worked with:

* representing grayscale images as NumPy arrays
* estimating affine transformations from corresponding points
* using homogeneous coordinates for matrix-based transformations
* applying inverse transformations for backward image mapping
* implementing image preprocessing methods such as downsampling, histogram equalization, smoothing, and edge detection
* detecting local features with SIFT
* matching descriptors using distance-based matching, ratio test, and cross-checking
* using RANSAC to remove outlier matches
* chaining pairwise transformations to align a complete image stack
* building a simple 3D volume from aligned 2D slices

---
