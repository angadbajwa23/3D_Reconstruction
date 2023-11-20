"""
Different feature matching techniques:

1. SIFT
2. ORB
3. Multi View Stereo (MVS) (not feature matching but we wish to use this in the Sfm process)


"""
import cv2
import numpy as np
import os


cv2.namedWindow("Image 1")
cv2.namedWindow("Image 2")

# change the path of the images
img1 = cv2.imread(os.path.join("Gustav","Gustav1.jpg"))
img2 = cv2.imread(os.path.join("Gustav","Gustav2.jpg"))


# ----------------------------------------SIFT-RANSAC---------------------------------------------------



# sift = cv2.xfeatures2d.SIFT_create()
# key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
# key_points_2, desc_2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

# # print(key_points_1)
# print(desc_1.shape)

# # Brute Force Matches using Euclidean Distance
# bf = cv2.BFMatcher()

# # Matching descriptors using KNN ( K = 2 )
# matches = bf.knnMatch(desc_1, desc_2, k=5)
# """
# knnMatch returns a list of lists consisting of 
# distance: A float representing the distance between the descriptors. A lower distance indicates a better match.
# trainIdx: The index of the descriptor in the training set (desc_2).
# queryIdx: The index of the descriptor in the query set (desc_1).
# imgIdx: The index of the image in the dataset (not always relevant).
# """

# # Apply ratio test
# #For a smaller threshold we get fewer points (sparser representation)
# good_matches = []
# for m, n,o,p,q in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)


# # RANSAC to estimate the homography
# src_pts = np.float32([key_points_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# dst_pts = np.float32([key_points_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# # RANSAC parameters
# ransac_reproj_threshold = 4.0
# homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)

# # Apply the homography to the first image
# matches_mask = mask.ravel().tolist()
# h, w,c = img1.shape
# pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# dst_pts_transformed = cv2.perspectiveTransform(pts, homography)

# # Draw the matches
# img_matches = cv2.drawMatches(img1, key_points_1, img2, key_points_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Draw the bounding box around the matched region
# img_matches = cv2.polylines(img_matches, [np.int32(dst_pts_transformed)], True, (0, 255, 0), 2, cv2.LINE_AA)

# # Display the result
# cv2.imshow('Matches', img_matches)
# cv2.imwrite("SIFT-RANSAC.png",img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ------------------------------------------  ORB  -------------------------------------------------


#Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
key_points_1, desc_1 = orb.detectAndCompute(img1, None)
key_points_2, desc_2 = orb.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.knnMatch(desc_1, desc_2, k=2)

# Apply ratio test
good_matches = []
for m,n in matches:
    if m.distance < 0.95 * n.distance:
        good_matches.append(m)

# Draw matches
img_matches = cv2.drawMatches(img1, key_points_1, img2, key_points_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the result
cv2.imshow('Matches', img_matches)
cv2.imwrite("ORB.png",img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()


# -----------------------------------------   Multi-View Stereo (MVS)  --------------------------------------------------


# # StereoBM (Block Matching) is a simple stereo algorithm in OpenCV
# stereo = cv2.StereoBM_create()

# # Compute the disparity map
# disparity = stereo.compute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

# # Normalize the disparity map for visualization
# normalized_disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)

# # Display the images and disparity map
# cv2.imshow('Image 1', img1)
# cv2.imshow('Image 2', img2)
# cv2.imshow('Disparity Map', normalized_disparity)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


