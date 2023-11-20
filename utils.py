"""
Helper functions for Structure for Motion Tasks
"""
import cv2
import numpy as np 
import os


def triangulation(point_2d_img1, point_2d_img2, projection_matrix1, projection_matrix2):
        '''
        Triangulates 3d points from 2d vectors and projection matrices
        returns projection matrix of first camera, projection matrix of second camera, point cloud 
        '''
        # cv2.traingulatePoints returns the traingulated 3D points of all the corresponding features between image 1 and image 2
        pt_cloud = cv2.triangulatePoints(projection_matrix1, projection_matrix2, point_2d_img1.T, point_2d_img2.T)

        # pt_cloud consists of [X,Y,Z,W] where W is a non-scalar. So we want to convert these homogeneuos coordinates to Euclidean coordinates by divides X/W, Y/W AND Z/W
        # returning the transpose of projection matrices and the euclidean coordinates
        #return projection_matrix1.T, projection_matrix2.T, (pt_cloud / pt_cloud[3])  
        return (pt_cloud / pt_cloud[3])    


def reprojection_error(object_points, image_points, transformation_matrix, K, homogenity):
        '''
        Calculates the reprojection error ie the distance between the projected points and the actual points.
        returns total error, object points
        '''
        rot_matrix = transformation_matrix[:3, :3]
        tran_vector = transformation_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        if homogenity == 1:
            object_points = cv2.convertPointsFromHomogeneous(object_points.T)
        image_points_calc, _ = cv2.projectPoints(object_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        total_error = cv2.norm(image_points_calc, np.float32(image_points) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error / len(image_points_calc), object_points



def PnP(object_point, image_point , K, dist_coeff, rot_vector, initial):
        '''
        Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
        returns rotational matrix, translational matrix, image points, object points, rotational vector
        '''
        if initial == 1:
            object_point = object_point[:, 0 ,:]
            image_point = image_point.T
            rot_vector = rot_vector.T 
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(object_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        # Converts a rotation matrix to a rotation vector or vice versa
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        if inlier is not None:
            image_point = image_point[inlier[:, 0]]
            object_point = object_point[inlier[:, 0]]
            rot_vector = rot_vector[inlier[:, 0]]
        return rot_matrix, tran_vector, image_point, object_point, rot_vector


def find_features(image_0, image_1):
        '''
        Feature detection using the sift algorithm and KNN
        return keypoints(features) of image1 and image2
        '''

        sift = cv2.xfeatures2d.SIFT_create()
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                feature.append(m)

        return np.float32([key_points_0[m.queryIdx].pt for m in feature]), np.float32([key_points_1[m.trainIdx].pt for m in feature])


def to_ply(point_cloud, colors,image_list):
        '''
        Generates the .ply which can be used to open the point cloud
        '''
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        #print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])


        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open('res/' + image_list[0].split('/')[-2] + '.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')