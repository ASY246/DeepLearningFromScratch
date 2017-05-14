import tensorflow as tf

IMAGE_PIXELS = 28

filenames = ['hdfs://default/user/bdusr01/asy/mergeOneHot.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)  #读入文件名序列
reader = tf.TextLineReader()  #读取器，用于输出由换行符分隔的行，读文件名
key, value = reader.read(filename_queue)  #返回reader产生的下一个记录
lines = tf.decode_csv(value, record_defaults=[[0] for i in range(794)])
features = tf.pack([*[lines[:-10]]])
labels = tf.pack([*lines[-10:]])

'''--------------------------------------------------------------------------------------'''
W = tf.Variable(tf.zeros([784, 10]))
W = tf.to_float(W)
b = tf.Variable(tf.zeros([10]))
b = tf.to_float(b)

x = tf.reshape(features, [1, IMAGE_PIXELS*IMAGE_PIXELS])  #直接把变量传进去
x = tf.to_float(x)
y = tf.nn.softmax(tf.matmul(x, W) + b)

#y_ = tf.placeholder(tf.float32, [None, 10])
y_ = tf.reshape(labels,[1,10])
y_ = tf.to_float(y_)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  coord = tf.train.Coordinator()  #创建一个协调器，管理线程
  threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。
  for i in range(1000):
    sess.run(train_step)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy))
  coord.request_stop()
  coord.join(threads)