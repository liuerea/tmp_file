# -*- codeing = utf-8 -*-
# @Time : 2024/3/5 23:34
# @Author : 陈思源
# @File : right_hand_test.py
# @Software : PyCharm
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
import time
import numpy as np

mp_holistic = mp.solutions.holistic

def draw3d(ax, world_landmarks, connection=mp_holistic.HAND_CONNECTIONS):
    ax.clear()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    landmarks = []
    for index, landmark in enumerate(world_landmarks.landmark):
        landmarks.append([landmark.x, landmark.z, landmark.y * (-1)])
    landmarks = np.array(landmarks)

    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='r', s=10)
    for _c in connection:
        ax.plot([landmarks[_c[0], 0], landmarks[_c[1], 0]],
                [landmarks[_c[0], 1], landmarks[_c[1], 1]],
                [landmarks[_c[0], 2], landmarks[_c[1], 2]], 'k')

    plt.pause(0.001)  # 在每次绘制后暂停一小段时间以更新图形

# 初始化视频捕获
cap = cv2.VideoCapture(0)

# 初始化 MediaPipe Holistic 模型
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as holistic:

    # 创建 Matplotlib 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 开始视频捕获和处理循环
    while cap.isOpened():
        # 读取视频帧
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 将视频帧转换为 RGB 格式
        start = time.time()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用 MediaPipe Holistic 模型处理视频帧
        results = holistic.process(image)

        # 将视频帧恢复为 BGR 格式
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        end = time.time()
        fps = 1 / (end - start)
        fps = "%.2f fps" % fps
        # 实时显示帧数
        image = cv2.flip(image, 1)
        cv2.putText(image, "FPS {0}".format(fps), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)

        # 绘制右手的 3D 散点图
        if results.right_hand_landmarks:
            draw3d(ax, results.right_hand_landmarks)

        # 显示视频帧
        cv2.imshow('MediaPipe Holistic', image)

        # 如果按下 ESC 键，则退出循环
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 释放视频捕获对象并关闭窗口
cap.release()
cv2.destroyAllWindows()
