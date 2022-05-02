import cv2 as cv
import os
import numpy
import numpy as np
import torch

# uv = torch.tensor([[[345, 240],  # 0
#                     [300, 225],
#                     [255, 195],
#                     [210, 180],
#                     [180, 180],
#                     [195, 255],  # 5
#                     [120, 255],
#                     [90, 255],
#                     [75, 255],  # 8 食指指尖
#                     [210, 300],
#                     [135, 315],
#                     [90, 330],
#                     [45, 345],
#                     [225, 330],
#                     [165, 345],
#                     [120, 375],
#                     [90, 390],
#                     [240, 360],
#                     [210, 375],
#                     [180, 390],
#                     [150, 405]]], device='cuda:0')
# uv = uv[0].cpu().numpy()
# uv = numpy.flip(uv, -1)


def paint_hand(uv, img):
    # img = np.ones((480, 480, 3), np.uint8)
    # img[:] = [255, 255, 255]
    # uv = uv[0].cpu().numpy()
    # uv = numpy.flip(uv, -1)
    for test in uv:
        xy = (test[0], test[1])
        cv.circle(img, xy, 4, (0, 0, 255), 1)
    cv.line(img, uv[0], uv[1], (255, 0, 0), 2)
    # print(uv[0:5])
    # 颜色顺序为BGR
    cv.polylines(img, [uv[0:5]], False, (0, 0, 255), 2)  # 大拇指
    cv.polylines(img, [uv[5:9]], False, (255, 0, 0), 2)  # 食指
    cv.polylines(img, [uv[9:13]], False, (255, 0, 0), 2)  # 中指
    cv.polylines(img, [uv[13:17]], False, (255, 0, 0), 2)  # 无名指
    cv.polylines(img, [uv[17:21]], False, (255, 0, 0), 2)  # 小拇指
    cv.line(img, uv[0], uv[5], (255, 0, 0), 2)
    cv.line(img, uv[0], uv[9], (255, 0, 0), 2)
    cv.line(img, uv[0], uv[13], (255, 0, 0), 2)
    cv.line(img, uv[0], uv[17], (255, 0, 0), 2)
    return img


def get_included_angle(coords1, coords2, coords3):
    # 通过斜率计算夹角
    # k1 = (coords2[1] - coords1[1]) / (coords2[0] - coords1[0])
    # k2 = (coords2[1] - coords3[1]) / (coords2[0] - coords3[0])
    #
    # x = np.array([1, k1])
    # y = np.array([1, k2])
    # Lx = np.sqrt(x.dot(x))
    # Ly = np.sqrt(y.dot(y))
    # Cobb = int((np.arccos(x.dot(y) / (float(Lx * Ly))) * 180 / np.pi) + 0.5)

    # 向量计算夹角
    arr_0 = np.array([(coords2[0] - coords1[0]), (coords2[1] - coords1[1])])
    arr_1 = np.array([(coords3[0] - coords2[0]), (coords3[1] - coords2[1])])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))
    # cos_value = format((float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1)))), '.9f')
    if cos_value > 1:
        cos_value = 1
    Cobb = np.arccos(cos_value) * (180 / np.pi)

    return Cobb


def judge_posture(uv):
    # uv = uv[0].cpu().numpy()
    # uv = numpy.flip(uv, -1)
    flag_thumb = 0
    flag_forefinger = 0
    flag_medius = 0
    flag_ring_finger = 0
    flag_little_finger = 0
    flag_judge = 0
    # 拇指弯曲角度
    angle0_1_2 = get_included_angle(uv[0], uv[1], uv[2])
    angle1_2_3 = get_included_angle(uv[1], uv[2], uv[3])
    angle2_3_4 = get_included_angle(uv[2], uv[3], uv[4])
    angle3_0_4 = get_included_angle(uv[3], uv[0], uv[4])

    # 食指弯曲角度
    angle0_5_6 = get_included_angle(uv[0], uv[5], uv[6])
    angle5_6_7 = get_included_angle(uv[5], uv[6], uv[7])
    angle6_7_8 = get_included_angle(uv[6], uv[7], uv[8])
    angle7_0_8 = get_included_angle(uv[7], uv[0], uv[8])

    # 中指弯曲角度
    angle0_9_10 = get_included_angle(uv[0], uv[9], uv[10])
    angle9_10_11 = get_included_angle(uv[9], uv[10], uv[11])
    angle10_11_12 = get_included_angle(uv[10], uv[11], uv[12])
    angle11_0_12 = get_included_angle(uv[11], uv[0], uv[12])

    # 无名指指弯曲角度
    angle0_13_14 = get_included_angle(uv[0], uv[13], uv[14])
    angle13_14_15 = get_included_angle(uv[13], uv[14], uv[15])
    angle14_15_16 = get_included_angle(uv[14], uv[15], uv[16])
    angle15_0_16 = get_included_angle(uv[15], uv[0], uv[16])

    # 小拇指弯曲角度
    angle0_17_18 = get_included_angle(uv[0], uv[17], uv[18])
    angle17_18_19 = get_included_angle(uv[17], uv[18], uv[19])
    angle18_19_20 = get_included_angle(uv[18], uv[19], uv[20])
    angle19_0_20 = get_included_angle(uv[19], uv[0], uv[20])
    if angle0_1_2 < 20 and angle1_2_3 < 20 and angle2_3_4 < 20 and angle3_0_4 > 175:
        flag_thumb = 1
    if angle0_5_6 < 20 and angle5_6_7 < 20 and angle6_7_8 < 20 and angle7_0_8 > 175:
        flag_forefinger = 1
    if angle0_9_10 < 20 and angle9_10_11 < 20 and angle10_11_12 < 20 and angle11_0_12 > 175:
        flag_medius = 1
    if angle0_13_14 < 25 and angle13_14_15 < 20 and angle14_15_16 < 20 and angle15_0_16 > 175:
        flag_ring_finger = 1
    if angle0_17_18 < 25 and angle17_18_19 < 20 and angle18_19_20 < 20 and angle19_0_20 > 175:
        flag_little_finger = 1

    if flag_thumb == 1 and flag_forefinger == 1 and flag_medius == 1 and flag_ring_finger == 1 and flag_little_finger == 1:
        flag_judge = 1
    # if flag_judge == 1:
    #     print("识别成功\n")
    return flag_judge


# 失败的拖尾效果QAQ
def show_special_effects(uv, img):
    height = 30
    width = 10
    # uv = uv[0].cpu().numpy()
    # uv = numpy.flip(uv, -1)
    tail = cv.imread('./tail.png')
    tail = cv.resize(tail, (width, height))
    x = uv[8][1]
    y = uv[8][0]
    if x - height >= 0 and y - (width // 2) >= 0 and y + (width // 2) <= img.shape[0]:
        img[x - height:x, y - (width // 2):y + (width // 2)] = tail
    return img


def show_switch_effects(img):
    pass


# 手指进入四角方块的互动
def click_box(uv, img, flag_ul, flag_ur, flag_ll, flag_lr):
    x = uv[8][1]
    y = uv[8][0]
    length = 80
    if flag_ul == 0:
        img[0:length, 0:length] = [0, 0, 0]
    if flag_ur == 0:
        img[0:length, 480 - length:480] = [80, 80, 80]
    if flag_ll == 0:
        img[480 - length:480, 0:length] = [160, 160, 160]
    if flag_lr == 0:
        img[480 - length:480, 480 - length:480] = [255, 255, 255]
    if x <= length:
        if y <= length:
            flag_ul = 1
        elif y >= 480 - length:
            flag_ur = 1
    elif x >= 480 - length:
        if y <= length:
            flag_ll = 1
        elif y >= 480 - length:
            flag_lr = 1
    return img, flag_ul, flag_ur, flag_ll, flag_lr


# # 测试函数功能
# img = np.zeros((480, 480, 3), np.uint8)
# img[:] = [200, 200, 200]
# paint_hand(uv, img)
# click_box(uv, img)
# # show_special_effects(uv, img)
# cv.imshow('img', img)
# cv.waitKey(0)
# judge_posture(uv)
# cv.destroyAllWindows()

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
