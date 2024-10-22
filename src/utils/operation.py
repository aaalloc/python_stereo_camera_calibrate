import numpy as np
from scipy import linalg

# Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.


def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2, :] - P1[1, :],
         P1[0, :] - point1[0]*P1[2, :],
         point2[1]*P2[2, :] - P2[1, :],
         P2[0, :] - point2[0]*P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)

    # print('Triangulated point: ')
    # print(Vh[3,0:3]/Vh[3,3])
    return Vh[3, 0:3]/Vh[3, 3]

# Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix


def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1

    return P
# Turn camera calibration data into projection matrix


def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    return P
