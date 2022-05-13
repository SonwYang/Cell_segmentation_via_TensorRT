import ctypes
from ctypes import *
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # dllPath = r"D:\2022\3\medicalSeg\code\cpp_medical_dll_python\build\Release\seg_lib.dll"
    # ll = ctypes.cdll.LoadLibrary
    #
    # pDll = ll(dllPath)
    #
    # enginePath = "model_trt_int8_2.engine"
    # imgPath = r"D:\2022\3\medicalSeg\images\rec_20190101_024426_0001.png"
    #
    # img = cv2.imread(imgPath)
    # img = cv2.resize(img.copy(), (1792, 1024))
    # (rows, cols) = (img.shape[0], img.shape[1])
    # ret_img = np.zeros(dtype=np.uint8, shape=(rows, cols, 3))
    #
    # p_str = ctypes.c_char_p(b'model_trt_int8_2.engine')
    #
    # pDll.Prediction(p_str, rows, cols, img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    #                 ret_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
    #
    # ###### 从指针指向的地址中读取数据，并转为numpy array
    # plt.imshow(ret_img)
    # plt.show()

    dllPath = "/home/scell/medicalSeg/cpp_dll/build/libseg_lib.so"
    ll = ctypes.cdll.LoadLibrary

    pDll = ll(dllPath)

   # imgPath = r"D:\2022\3\medicalSeg\images\rec_20190101_024426_0001.png"

    #img = cv2.imread(imgPath)
   # img = cv2.resize(img.copy(), (1792, 1024))
   # (rows, cols) = (img.shape[0], img.shape[1])

    engine_str = ctypes.c_char_p(b'linknet_trt_int8.engine')

    seg = pDll.init(engine_str)

    cap = cv2.VideoCapture("/home/scell/medicalSeg/cpp/rec_20190101_024426.avi")
    #cap = cv2.VideoCapture("/dev/video2")
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()  # 是否成功打开
    # 1080, 1920
    while suc:
        suc, frame = cap.read()
        print(f"frame size : {frame.shape}")
        frame = cv2.resize(frame, (1920, 1024))
        w, h = frame.shape[:2]
        
        ret_img = np.zeros(dtype=np.uint8, shape=(w, h, 3))
        pDll.prediction(w, h, frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                        ret_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        cv2.namedWindow("Image")
        #cv2.resizeWindow("Image", 1024, 1792)
        #plt.imshow(ret_img)
        #plt.show()
        #del ret_img
        cv2.imshow("Image", ret_img)
        cv2.waitKey(delay=2)
       
    cv2.destroyWindow("Image")

      


