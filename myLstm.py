import tensorflow as tf
import tensorflow.nn.rnn_cell as rnn
import numpy as np


batch_size = 16
hidden_state = 32
input_size = 26
seq_length = 40
embedding_size = 50

with tf.name_scope("weight_lstm"):
    lstm = rnn.LSTMCell(hidden_state)
# inputs = tf.placeholder(dtype=tf.int32, shape=[batchsize, 26])

# 定义数据，标签
# inputs = tf.placeholder(tf.int32, [batch_size, seq_length])
# targets = tf.placeholder(tf.int32, [batch_size, seq_length])
input_data = np.random.randint(0, 50, size=(batch_size, seq_length))  # shape(16, 40)
targets = np.random.randint(0, 50, size=(batch_size, seq_length))

# 初始化状态
state = lstm.zero_state(batch_size, tf.float32)
print(state)

# 初始化嵌入矩阵
with tf.name_scope("weight_input"):
    weight = tf.Variable(tf.random_uniform([embedding_size, hidden_state]))  # shape(50, 32)
# print(weight)

# 初始化输入到预测的矩阵
with tf.name_scope("weight_output"):
    fc_w = tf.Variable(tf.random_uniform([hidden_state, embedding_size]))
    fc_b = tf.Variable(tf.zeros(embedding_size))

# 将输入数据表示成向量
inputs = tf.nn.embedding_lookup(weight, input_data)
# 分割成每一个时序
inputs = tf.split(inputs, seq_length, axis=1)
print(inputs)

'''
loss = 0.0
for t in range(seq_length):
    output, state = lstm(inputs=tf.reshape(inputs[t], [-1, hidden_state]), state=state)
    scores = tf.matmul(output, fc_w) + fc_b  # shape(batch_size, embedding_size)
    loss += tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(targets[:, t],
                                                    depth=embedding_size, axis=1),
                                                    logits=scores)
total_loss = tf.reduce_mean(loss)/seq_length
train = tf.train.AdamOptimizer(0.1).minimize(total_loss)
update = tf.get_collection(tf.GraphKeys.UPDATE_OPS, tf.GraphKeys.TRAINABLE_VARIABLES)

sess = tf.Session()
writer = tf.summary.FileWriter("./logs/", sess.graph)
sess.run(tf.global_variables_initializer())
for step in range(100):
    np_loss, _ = sess.run([total_loss, train])
    print(np_loss)
'''