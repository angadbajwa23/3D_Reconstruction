"""

Outline of all the function which will be used in the reconstruction process ( for 2 images of now)

1. Finding camera matrix for image 1 (using intrinsic and calculating extrinsic parameters) and initializing camera matrix for image 2
2. Finding common features between image 1 and image 2
3. Finding the transformation matrix (rotation and translation) between camera matrix for image 1 and image 2
4. Finding camera matrix for image 2 using camera matrix of image 1 and the calculated transformation matrix
5. Using triangulation to find the 3d positions given the common features and the camera poses(matrices)
6. Calculating the reprojection error of these 3d points

"""
import cv2
import os
import numpy as np
from utils import triangulation,reprojection_error,find_features,to_ply,PnP


img_dir = "Gustav"
#print(os.path.join(img_dir, 'intrinsic_parameters.txt'))
with open(os.path.join(img_dir, 'intrinsic_parameters.txt')) as f:
    K = np.array(list((map(lambda x:list(map(lambda x:float(x), x.strip().split(' '))),f.read().split('\n')))))
    image_list = []
# Loading the set of images
for image in sorted(os.listdir(img_dir)):
    if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.png':
        image_list.append(os.path.join(img_dir, image))

print(image_list)

total_points = np.zeros((1, 3))
total_colors = np.zeros((1, 3))

# Getting the intrinsic camera parameters
# 3X3 matrix
pose_array = K.ravel()

# Transform matrix  consists of Rotation and Translation Matrix which are the external camera parameters
# Taking the first image as the base and assuming the first camera is at the origin
# 3X4 matrix just like we did in class
transform_matrix_0 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]])

# This is the camera matrix for the second image, we plan on updating the matrix itself for the remaining images(eg: for image3,image 4...)
transform_matrix_1 =  np.empty((3, 4))


# The camera matrix for camera 1 ( associated with image 1) : combining intrinsic and extrinsic parameters
pose_0 = np.matmul(K, transform_matrix_0)
# Initializing camera matrix for camera 2
pose_1 = np.empty((3, 4)) 

image_0 = cv2.imread(image_list[0])
image_1 = cv2.imread(image_list[1])


# Finding the common features using either SIFT-RANSAC/ ORB / MVS 
# returns a list of features in the source image( image 1) and destination image ( image 2)

feature_0, feature_1 = find_features(image_0, image_1)

print(len(feature_0))
print(len(feature_1))

# Finding the essential matrix ( transformation matrix between 2 cameras. Relates motion b/w 2 cameras and used for pose estimation )
essential_matrix, em_mask = cv2.findEssentialMat(feature_0, feature_1, K, method=cv2.RANSAC)
#feature_0 = feature_0[em_mask.ravel() == 1]
#feature_1 = feature_1[em_mask.ravel() == 1]
# Recovering camera pose
# R- rotation matrix, t- translation vector, mask- inlier points mask, retval-boolean value whether pose recovery was successful
retval, R, t, mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, K)

# Keeping only those features which are greater than 0 in the mask
#features_0 = feature_0[em_mask.ravel() > 0]
#features_1 = feature_1[em_mask.ravel() > 0]

# Finding the transform matrix for image 2 (extrinsic parameters) : By multiplying the Rotation matrix we obtained with the rotation component of camera 1
# And adding the translation vector with the translation component of camera 1
transform_matrix_1[:3, :3] = np.matmul(R, transform_matrix_0[:3, :3])
transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], t.ravel())

# The camera matrix for camera 2 ( associated with image 2) : combining intrinsic and extrinsic parameters
pose_1 = np.matmul(K, transform_matrix_1)

#print("new length of feature_0",len(feature_0))
# Having found the camera matrix for both the cameras we do triangulation and then find the reprojection error
#feature_0, feature_1, points_3d = triangulation(feature_0, feature_1,pose_0,pose_1)
points_3d = triangulation(feature_0, feature_1,pose_0,pose_1)
error, points_3d = reprojection_error(points_3d, feature_1, transform_matrix_1, K, homogenity = 1)

print("REPROJECTION ERROR: ", error)
print("new length of feature_0",len(feature_0))
#_, _, feature_1, points_3d, _ = PnP(points_3d, feature_1, K, np.zeros((5, 1), dtype=np.float32), feature_0, initial=1)

total_images = len(image_list) - 2 
pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))

#threshold = 0.5
#for i in tqdm(range(total_images)):
print("Points 3d",points_3d.shape)
total_points = np.vstack((total_points, points_3d[:,0,:]))
colors = np.array([image_0[int(pt[1]), int(pt[0])] for pt in feature_0])
total_colors = np.vstack((total_colors,colors))
#total_colors  = np.array([image_0[int(pt[1]), int(pt[0])] for pt in feature_0])

#points_left = np.array(cm_mask_1, dtype=np.int32)
#color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
#total_colors = np.vstack((total_colors, color_vector)) 
print("Printing to .ply file")
print(total_points)
print(total_colors)
print(total_points.shape, total_colors.shape)
to_ply(total_points, total_colors,image_list)
print("Completed Exiting ...")
np.savetxt('res/' +image_list[0].split('/')[-2]+'_pose_array.csv', pose_array, delimiter = '\n')
