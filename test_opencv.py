import cv2 as cv
import os

import numpy
import numpy as np
import torch


# uv = torch.tensor([[[360, 270],
#                     [330, 300],
#                     [255, 330],
#                     [210, 345],
#                     [150, 375],
#                     [225, 270],
#                     [150, 240],
#                     [120, 225],
#                     [75, 195],
#                     [255, 225],
#                     [195, 180],
#                     [165, 165],
#                     [180, 165],
#                     [285, 195],
#                     [225, 135],
#                     [195, 120],
#                     [180, 105],
#                     [315, 180],
#                     [285, 135],
#                     [270, 90],
#                     [255, 60]]], device='cuda:0')
def paint_hand(uv, img):
    # img = np.ones((480, 480, 3), np.uint8)
    # img[:] = [255, 255, 255]
    uv = uv[0].cpu().numpy()
    uv = numpy.flip(uv, -1)
    for test in uv:
        xy = (test[0], test[1])
        cv.circle(img, xy, 4, (0, 0, 255), 1)
    cv.line(img, uv[0], uv[1], (255, 0, 0), 2)
    # print(uv[0:5])
    cv.polylines(img, [uv[0:5]], False, (255, 0, 0), 2)
    cv.polylines(img, [uv[5:9]], False, (255, 0, 0), 2)
    cv.polylines(img, [uv[9:13]], False, (255, 0, 0), 2)
    cv.polylines(img, [uv[13:17]], False, (255, 0, 0), 2)
    cv.polylines(img, [uv[17:21]], False, (255, 0, 0), 2)
    cv.line(img, uv[0], uv[5], (255, 0, 0), 2)
    cv.line(img, uv[0], uv[9], (255, 0, 0), 2)
    cv.line(img, uv[0], uv[13], (255, 0, 0), 2)
    cv.line(img, uv[0], uv[17], (255, 0, 0), 2)
    return img

# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# a = np.random.randint(1, 10, size=15).reshape((3, 5))
# print(a)
# print(a.shape)
# print(np.flip(a, -1))
# print(np.flip(a, 0))
# print(np.flip(a, 1))

# 读取图像
# print(os.getcwd())
# img = cv.imread('./materials/demo2.jpeg')
# cv.imshow('image', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 读取视频
# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     # 逐帧捕获
#     ret, frame = cap.read()
#     # 如果正确读取帧，ret为True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # 我们在框架上的操作到这里
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # 显示结果帧e
#     cv.imshow('frame', gray)
#     if cv.waitKey(1) == ord('q'):
#         break
# # 完成所有操作后，释放捕获器
# cap.release()
# cv.destroyAllWindows()
