#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import tensorflow as tf
import numpy as np
import time

INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
INPUT_SIZE = 513

def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

def main():

    # 動画の読み込み
    cap = cv2.VideoCapture("input.mp4")

    with tf.gfile.FastGFile('deeplabv3.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        color_label = create_pascal_label_colormap()

        # 動画終了まで繰り返し
        while(cap.isOpened()):

            # フレームを取得
            ret, frame = cap.read()

            if ret == False:
                break

            width, height, ch = frame.shape
            resize_ratio = 1.0 * INPUT_SIZE / max(width, height)

            inp = cv2.resize(frame, (int(resize_ratio * height), int(resize_ratio * width)))
            inp = inp[:, :, [2, 1, 0]]

            start = time.time()
            batch_seg_map = sess.run(
                OUTPUT_TENSOR_NAME,
                feed_dict={INPUT_TENSOR_NAME: inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
            elapsed_time = time.time() - start
            print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

            seg_map = batch_seg_map[0]

            # フレームを表示
            inp = cv2.cvtColor(inp, cv2.COLOR_RGB2BGR)
            cv2.imshow("Flame", inp)

            seg_image = color_label[seg_map].astype(np.uint8)
            cv2.imshow("deeplabv3", seg_image)

            # qキーが押されたら途中終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()