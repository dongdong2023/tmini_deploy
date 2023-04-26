# encoding=utf-8
import numpy as np
from scipy.linalg import lstsq
from math import degrees
from sklearn.linear_model import RANSACRegressor
import numpy as np
import open3d as o3d


# 输入是3个点
def point_angle(points):
    # do fit

    X0 = np.ones([points.shape[0], 1])
    tmp_A = points[:, :2]
    tmp_A = np.hstack((tmp_A, X0))
    tmp_b = points[:, 2]

    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    # Or use Scipy

    fit, residual, rnk, s = lstsq(A, b)

    # Or use numpy
    # (a,B,C),resid,rank,s =  np.linalg.lstsq(A, b)

    errors = b - A * fit

    plane_normal = [fit[0][0], fit[1][0], -1]

    X_normal = [1, 0, 0]
    Y_normal = [0, 1, 0]
    Z_normal = [0, 0, 1]

    dist1 = np.dot(plane_normal, X_normal) / (np.linalg.norm(plane_normal) * np.linalg.norm(X_normal))
    radio = np.arccos(dist1)
    X_angel = degrees(radio)

    dist1 = np.dot(plane_normal, Y_normal) / (np.linalg.norm(plane_normal) * np.linalg.norm(Y_normal))
    radio = np.arccos(dist1)
    Y_angel = degrees(radio)

    dist1 = np.dot(plane_normal, Z_normal) / (np.linalg.norm(plane_normal) * np.linalg.norm(Z_normal))
    radio = np.arccos(dist1)
    Z_angel = degrees(radio)

    # X_angel = np.degrees(np.arcsin(np.abs(-1) / np.linalg.norm(plane_normal)))
    # Y_angel = np.degrees(np.arcsin(np.abs(plane_normal[0]) / np.linalg.norm(plane_normal)))
    # Z_angel = np.degrees(np.arcsin(np.abs(plane_normal[1]) / np.linalg.norm(plane_normal)))
    # if X_angel>90:
    #     X_angel-=180
    return np.array([X_angel, Y_angel, Z_angel, 90 - X_angel])


def fit_face(points):
    X0 = np.ones([points.shape[0], 1])
    tmp_A = points[:, :2]
    tmp_A = np.hstack((tmp_A, X0))
    tmp_b = points[:, 2]

    R = np.mat(tmp_A)
    A = np.dot(np.dot(np.linalg.inv(np.dot(R.T, R)), R.T), tmp_b)
    A = np.array(A, dtype='float32').flatten()
    print('平面拟合结果为：z = %.3f * x + %.3f * y + %.3f' % (A[0], A[1], A[2]))
    print('法向量为：（%.3f, %.3f, -1）' % (A[0], A[1]))

    X_angel = np.degrees(np.arcsin(np.abs(-1) / np.sqrt(np.power(A[0], 2) + np.power(A[1], 2) + np.power(-1, 2))))
    Y_angel = np.degrees(np.arcsin(np.abs(A[0]) / np.sqrt(np.power(A[0], 2) + np.power(A[1], 2) + np.power(-1, 2))))
    Z_angel = np.degrees(np.arcsin(np.abs(A[1]) / np.sqrt(np.power(A[0], 2) + np.power(A[1], 2) + np.power(-1, 2))))

    return np.array([X_angel, Y_angel, Z_angel])


def open3d_segment(points):
    # RANSAC 剔除离群点
    model = RANSACRegressor(residual_threshold=0.5)
    model.fit(points[:, :2], points[:, 2])
    inliers = model.inlier_mask_

    # 最小二乘法拟合平面
    X = np.column_stack((points[:, :2], np.ones(points.shape[0])))
    coefficients, _, _, _ = lstsq(X[inliers, :], points[inliers, 2])

    import csv
    fo = open("info.csv", "a", newline='')
    writer = csv.writer(fo)
    res = points[inliers, 2].astype(int).astype(str)
    writer.writerow(res)
    # 关闭文件
    fo.close()

    plane_normal = [coefficients[0], coefficients[1], -1]

    X_normal = [1, 0, 0]
    Y_normal = [0, 1, 0]
    Z_normal = [0, 0, 1]

    dist1 = np.dot(plane_normal, X_normal) / (np.linalg.norm(plane_normal) * np.linalg.norm(X_normal))
    radio = np.arccos(dist1)
    X_angel = degrees(radio)

    dist1 = np.dot(plane_normal, Y_normal) / (np.linalg.norm(plane_normal) * np.linalg.norm(Y_normal))
    radio = np.arccos(dist1)
    Y_angel = degrees(radio)

    dist1 = np.dot(plane_normal, Z_normal) / (np.linalg.norm(plane_normal) * np.linalg.norm(Z_normal))
    radio = np.arccos(dist1)
    Z_angel = degrees(radio)

    # X_angel = np.degrees(np.arcsin(np.abs(-1) / np.linalg.norm(plane_normal)))
    # Y_angel = np.degrees(np.arcsin(np.abs(plane_normal[0]) / np.linalg.norm(plane_normal)))
    # Z_angel = np.degrees(np.arcsin(np.abs(plane_normal[1]) / np.linalg.norm(plane_normal)))
    # if X_angel>90:
    #     X_angel-=180
    return np.array([X_angel, Y_angel, Z_angel, 90 - X_angel]), inliers


def fit(points):
    # 生成点云数据
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 使用RANSAC算法拟合平面
    distance_threshold = 10
    ransac_n = 3
    num_iterations = 100
    probability = 0.99
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations, probability)

    # 获取平面法向量和单位向量
    plane_normal = np.array(plane_model[:3])
    plane_normal /= np.linalg.norm(plane_normal)
    X_normal = [1, 0, 0]
    Y_normal = [0, 1, 0]
    Z_normal = [0, 0, 1]

    # 计算夹角（单位为弧度）
    angle = np.arccos(np.dot(plane_normal, X_normal))
    # 将夹角转换为角度
    X_angel = degrees(angle)

    # 计算夹角（单位为弧度）
    angle = np.arccos(np.dot(plane_normal, Y_normal))
    # 将夹角转换为角度
    Y_angel = degrees(angle)

    # 计算夹角（单位为弧度）
    angle = np.arccos(np.dot(plane_normal, Z_normal))
    # 将夹角转换为角度
    Z_angel = degrees(angle)

    return np.array([X_angel, Y_angel, Z_angel, 90 - X_angel]), inliers
