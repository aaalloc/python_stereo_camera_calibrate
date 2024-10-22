import cv2 as cv
import numpy as np


def get_world_space_origin(cmtx, dist, img_path, calibration_settings):

    frame = cv.imread(img_path, 1)

    # calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
    cv.putText(frame, "If you don't see detected points, try with a different image",
               (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv.imshow('img', frame)
    cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    # rvec is Rotation matrix in Rodrigues vector form
    R, _ = cv.Rodrigues(rvec)

    return R, tvec


def _project_points_to_frame(R, T, cmtx, dist, points, frame, colors):
    points, _ = cv.projectPoints(points, R, T, cmtx, dist)
    points = points.reshape((-1, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame, origin, _p, col, 2)


def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0,
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):

    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                [0, 0, 1]], dtype='float32').reshape((4, 1, 3))
    # axes colors are RGB format to indicate XYZ axes.
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # project origin points to frame 0
    _project_points_to_frame(R_W0, T_W0, cmtx0, dist0,
                             unitv_points, frame0, colors)
    # points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    # points = points.reshape((4, 2)).astype(np.int32)
    # origin = tuple(points[0])
    # for col, _p in zip(colors, points[1:]):
    #     _p = tuple(_p.astype(np.int32))
    #     cv.line(frame0, origin, _p, col, 2)

    # project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01

    _project_points_to_frame(R_W1, T_W1, cmtx1, dist1,
                             unitv_points, frame1, colors)
    # points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    # points = points.reshape((4, 2)).astype(np.int32)
    # origin = tuple(points[0])
    # for col, _p in zip(colors, points[1:]):
    #     _p = tuple(_p.astype(np.int32))
    #     cv.line(frame1, origin, _p, col, 2)

    cv.imshow('frame0', frame0)
    cv.imshow('frame1', frame1)
    cv.waitKey(0)

    return R_W1, T_W1
