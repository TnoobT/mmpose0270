'''
计算onnx耗时
'''

import cv2
import numpy as np
import onnxruntime
import torch
import time

onnx_path = './15.onnx'

times = []
def get_onnx_output(img):
    sess = onnxruntime.InferenceSession(onnx_path)
    start = time.time()
    output_onnx = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: np.array(img,dtype=np.float32),})
    end = time.time()
    times.append(end-start)
    return output_onnx

if __name__ == '__main__':

    # 数据预处理
    img = cv2.imread('./test.jpg')
    ori_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    img = cv2.resize(img, (256, 256))
    img = (img / 127.5) - 1.0
    img = img.transpose(2,0,1)[None,...]

    for i in range(1000):
        output_onnx = get_onnx_output(img)
    print(times)
    print('mean time: {}'.format(np.mean(times[50:950])))
