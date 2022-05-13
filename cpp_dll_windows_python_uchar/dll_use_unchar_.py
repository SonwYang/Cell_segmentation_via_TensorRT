import ctypes
from ctypes import *
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dllPath = r"D:\2022\handover_material\cell_seg\cpp_dll_windows_uchar\build\Release\seg_lib.dll"
    ll = ctypes.cdll.LoadLibrary

    pDll = ll(dllPath)

    engine_str = ctypes.c_char_p(b'model_trt_int8_2.engine')

    seg = pDll.init(engine_str)

    cap = cv2.VideoCapture(r"D:\2022\3\medicalSeg\video\kidney\rec_20190101_024426.avi")
    suc = cap.isOpened()  # 是否成功打开
    # 1080, 1920
    while suc:
        suc, frame = cap.read()
        frame2 = cv2.resize(frame.copy(), (1792, 1024))
        rows, cols = frame2.shape[:2]
        ret_img = np.zeros(dtype=np.uint8, shape=(rows, cols, 3))
        # 设置输出数据类型为uint8的指针
        pDll.prediction.restype = ctypes.POINTER(ctypes.c_uint8)

        pointer = pDll.prediction(rows, cols, frame2.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        np_reuslt = np.array(np.fromiter(pointer, dtype=np.uint8, count=1792 * 1024 * 3)).reshape((1024, 1792, 3))

        count = pDll.getCount()
        percent = pDll.getPercent()
        print(f"count : {count}, percent : {percent}")
        cv2.namedWindow("Image")
        cv2.imshow("Image", np_reuslt)
        cv2.waitKey(delay=2)

    cv2.destroyWindow("Image")


