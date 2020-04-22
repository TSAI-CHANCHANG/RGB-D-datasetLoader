import cv2
import numpy as np


def tran_matrix_2_vec(transfomation_matrix):
    rot_vec, jacb = cv2.Rodrigues(transfomation_matrix[0:3, 0:3])
    trans_vec = (np.linalg.inv(transfomation_matrix[0:3, 0:3]) @ (-transfomation_matrix[0:3, 3]).T).T
    return rot_vec.T, trans_vec


def vec2T(rotvec, tranvec):
    T = np.zeros([4, 4])
    T[0:3, 0:3] = cv2.Rodrigues(rotvec)[0]
    T[0:3, 3] = - (T[0:3, 0:3] @ tranvec.T).T
    T[3, 3] = 1.0
    return T


# a = np.array([0.3, 0.5, 0.7])
#
# print(vec2T(a, a))
# print(cv2.Rodrigues(a)[0])
# print(cv2.Rodrigues(cv2.Rodrigues(a)[0])[0])
# rot_vec, trans_vec, = tran_matrix_2_vec(vec2T(a, a))
# print(rot_vec)
# print(trans_vec)
# print("----------------------------------")
# a = np.array((0.2, 0.4, 0.8))
# b = np.array((0.1, 0.3, 0.5))
# print("a" + str(a))
# R = cv2.Rodrigues((0.2, 0.4, 0.8))
# print("R[0]" + str(R[0]))
# print(vec2T(a, b))
# print(tran_matrix_2_vec(vec2T(a, b)))
# print(cv2.Rodrigues(R[0])[0])
# print("----------------------------------")
# a = np.array([[9.3306959e-001, -1.6964546e-001, 3.1708300e-001, -9.4861656e-001],
#               [1.7337367e-001, 9.8468649e-001, 1.6643673e-002, -5.6749225e-001],
#               [-3.1506166e-001, 3.9447054e-002, 9.4821417e-001, 7.1117169e-001],
#               [0.0000000e+000, 0.0000000e+000, 0.0000000e+000, 1.0000000e+000]])
# rot_vec, trans_vec = tran_matrix_2_vec(a)
# print(rot_vec)
# print(trans_vec)
# print(vec2T(rot_vec, trans_vec))
# print(tran_matrix_2_vec(vec2T(rot_vec, trans_vec)))