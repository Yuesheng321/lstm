import numpy as np
import tensorflow as tf
import model
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=10, help='# of data')
    parser.add_argument('--hidden_size', type=int, default=32, help='# of batch')
    parser.add_argument('--seq_length', type=int, default=40, help='length of each sequence')
    parser.add_argument('--embedding_size', type=int, default=50, help='size of embedding')
    parser.add_argument('--checkpoint_path', type=str, default='save/', help='path of model`s parameter')
    args = parser.parse_args()
    test(args)


# 预测
def test(args):
    input_data = np.random.randint(0, args.embedding_size, size=(args.input_size, args.seq_length))
    sess = tf.Session()
    lstm = model.Rnn(args, isTrain=False)
    lstm.input_data = input_data
    # 定义预测模型
    pre_list = lstm.sample()
    # 定义模型加载器
    saver = tf.train.Saver(tf.global_variables())
    # 恢复模型
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        pre_val = sess.run(pre_list)
        print(np.array(pre_val))


if __name__ == '__main__':
    main()
