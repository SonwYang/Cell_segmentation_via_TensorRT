#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger
import torch
import tensorrt as trt
from torch2trt import torch2trt

@logger.catch
def main():
    model = torch.load("./checkpoints/best.pth", map_location='cpu')
    # ckpt = torch.load("resnet18-5c106cde.pth", map_location="cpu")
    # model.load_state_dict(ckpt)

    logger.info("loaded checkpoint done.")
    model.eval()
    model.cuda()
    x = torch.ones(1, 3, 1024, 1792).cuda()

    print("Torch2TensorRT")
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << 32)
    )
    #torch.save(model_trt.state_dict(), "model_trt.pth")
    logger.info("Converted TensorRT model done.")

    y = model(x)
    y_trt = model_trt(x)

    # check the output against PyTorch
    print(f"difference: {torch.max(torch.abs(y - y_trt))}")

    engine_file = "./checkpoints/model_trt.engine"

    print("generate engine file")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


## python tools/trt.py -n yolox-s -c D:\MyWorkSpace\git\YOLOX-main\yolox_s.pth
if __name__ == "__main__":
    main()

