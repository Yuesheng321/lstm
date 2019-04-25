import tensorflow as tf
import tensorflow.nn.rnn_cell as rnn
from loader import Dataloader


class Rnn:
    def __init__(self, args, isTrain=True):
        self.hidden_size = args.hidden_size
        self.seq_length = args.seq_length
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        # 输入数据和输出数据
        self.input_data = tf.placeholder(tf.int32, shape=[self.input_size, self.seq_length])
        if isTrain is True:
            self.targets = tf.placeholder(tf.int32, shape=[self.input_size, self.seq_length])
            self.batch_size = args.batch_size
        # self.input_data = args.input_data
        # self.targets = args.targets
        with tf.name_scope("lstm"):
            self.lstm = rnn.LSTMCell(self.hidden_size)
        # 初始化嵌入矩阵
        with tf.name_scope("input"):
            self.weight_input = tf.Variable(tf.random_uniform([self.embedding_size, self.hidden_size]))
        # 初始化输入到预测的矩阵
        with tf.name_scope("output"):
            self.weight_out = tf.Variable(tf.random_uniform([self.hidden_size, self.embedding_size]))
            self.b_out = tf.Variable(tf.zeros(self.embedding_size))

    def loss(self):
        # 初始化隐藏层状态，不是权重
        state = self.lstm.zero_state(self.batch_size, tf.float32)
        # 将输入数据表示成向量
        inputs = tf.nn.embedding_lookup(self.weight_input, self.input_data)  # shape(None, seq_length, hidden_size)
        # 分割成每一个时序
        # inputs = tf.split(inputs, self.seq_length, axis=1)
        loss = 0.0
        # batch加载器
        batch = Dataloader(inputs, self.targets, self.batch_size)
        for batch_x, batch_y in batch:
            # print(batch_x.shape, batch_y.shape)
            for t in range(self.seq_length):
                output, state = self.lstm(inputs=tf.reshape(batch_x[:, t, :], [-1, self.hidden_size]), state=state)
                scores = tf.matmul(output, self.weight_out) + self.b_out  # shape(batch_size, embedding_size)
                loss += tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(batch_y[:, t],
                                                                   depth=self.embedding_size, axis=1),
                                                                   logits=scores)
        total_loss = tf.reduce_mean(loss) / self.seq_length
        train_step = tf.train.AdamOptimizer(0.1).minimize(total_loss)
        update = tf.get_collection(tf.GraphKeys.UPDATE_OPS, tf.GraphKeys.TRAINABLE_VARIABLES)
        return total_loss, train_step

    def sample(self):
        # 初始化状态
        state = self.lstm.zero_state(self.input_size, dtype=tf.float32)  # h:(N, hidden_size), c:(N, hidden_size)
        # 初始化预测序列
        pre_list = []
        # 将输入数据表示成向量
        inputs = tf.nn.embedding_lookup(self.weight_input, self.input_data)
        # print(inputs)  # shape=(10, 40, 32)
        x = tf.reshape(inputs[:, 0, :], [-1, self.hidden_size])  # 仅使用第一个时间节点
        # print(x)  # shape=(10, 32)
        for t in range(self.seq_length):
            output, state = self.lstm(inputs=x, state=state)
            # print(output, state)  # shape=(10, 32), shape=(10, 32), shape=(10, 32)
            scores = tf.matmul(output, self.weight_out) + self.b_out
            prob = tf.nn.softmax(scores, axis=1)
            # 获取最大概率的下标
            pre_val = tf.argmax(prob, axis=1)
            # print(pre_val)
            pre_list.append(pre_val)
            x = output
        return pre_list
