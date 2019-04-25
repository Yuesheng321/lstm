import numpy as np
import tensorflow as tf
import model
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=32, help='# of data')
    parser.add_argument('--batch_size', type=int, default=16, help='# of batch')
    parser.add_argument('--hidden_size', type=int, default=32, help='# of batch')
    parser.add_argument('--seq_length', type=int, default=40, help='length of each sequence')
    parser.add_argument('--embedding_size', type=int, default=50, help='size of embedding')
    parser.add_argument('--checkpoint_path', type=str, default='save/', help='path of model`s parameter')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='# of train')
    parser.add_argument('--print_every', type=int, default=10, help='# of print every')
    parser.add_argument('--logs', type=str, default='logs/', help='path of logs')
    args = parser.parse_args()
    train(args)


def train(args):
    input_data = np.random.randint(0, args.embedding_size, size=(args.input_size, args.seq_length))  # shape(N, 40)
    targets = np.random.randint(0, args.embedding_size, size=(args.input_size, args.seq_length))

    tf.reset_default_graph()
    sess = tf.Session()
    # 定义图结构
    lstm = model.Rnn(args)
    lstm.input_data = input_data
    lstm.targets = targets

    total_loss, train_step = lstm.loss()
    sess.run(tf.global_variables_initializer())
    # 保存图结构
    tf.summary.FileWriter(args.logs, sess.graph)
    # 定义参数保存器
    saver = tf.train.Saver(tf.global_variables())
    # 训练
    for epoch in range(args.epochs):
        np_loss, _ = sess.run([total_loss, train_step])
        if epoch % args.print_every == 0:
            print('epoch{}:{}'.format(epoch, np_loss))
    checkpoint_path = os.path.join(args.checkpoint_path, 'model'+str(args.epochs)+'.ckpt')
    saver.save(sess, checkpoint_path, global_step=args.epochs)


if __name__ == '__main__':
    main()

