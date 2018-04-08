#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import tensorflow as tf


INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
INPUT_SIZE = 513

def main():

    # 動画の読み込み
    cap = cv2.VideoCapture("/Users/tatsunori/work/opencv/input.mp4")

    with tf.gfile.FastGFile('/Users/tatsunori/work/opencv/deeplabv3.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # 動画終了まで繰り返し
        while(cap.isOpened()):

            # フレームを取得
            ret, frame = cap.read()

            if ret == False:
                break

            inp = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            inp = inp[:, :, [2, 1, 0]]
            batch_seg_map = sess.run(
                OUTPUT_TENSOR_NAME,
                feed_dict={INPUT_TENSOR_NAME: inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
            seg_map = batch_seg_map[0]

            # フレームを表示
            cv2.imshow("Flame", seg_map)

            # qキーが押されたら途中終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()