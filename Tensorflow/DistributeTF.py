import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data
# 命令行参数传入ClusterSpec定义
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")

tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# 命令行参数传入Server定义，jobname和taskindex
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

#  Tensorflow Server模型描述信息，包括隐含层神经元数量，MNIST数据目录以及每次训练数据大小（默认一个批次为100个图片）
tf.app.flags.DEFINE_integer("hidden_units", 100, "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "MNIST", "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = 28

def main(_):
    # 解析ps和worker对应机器和端口
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # 从参数服务器和worker参数创建集群的描述对象ClusterSpec
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # 为该节点运行的服务创建一个server
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index) #cluster中的ps还是worker，其中的第几个任务

    if FLAGS.job_name == "ps":  # 如果是参数服务器，执行参数join
        server.join()
    elif FLAGS.job_name == "worker":  # 如果是worker
        # 默认的方式对该本地worker分配op，默认为该节点上的cpu0
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)): #传入worker的第几个task
            ##################################################################################
            # 定义TensorFlow隐含层参数变量，为全连接神经网络隐含层
            hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units], stddev=1.0/IMAGE_PIXELS), name = "hid_w")
            hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name = "hid_b")
            
            # 定义TensorFlow softmax回归层的参数变量
            sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10], stddev = 1.0/math.sqrt(FLAGS.hidden_units)), name = "sm_w")
            sm_b = tf.Variable(tf.zeros([10]), name = "sm_b")
            
            # 定义模型输入数据变量（x为图片像素数据，y_为手写数字分类）
            x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
            y_ = tf.placeholder(tf.float32, [None, 10])
            
            # 定义隐含层及神经元计算模型
            hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
            hid = tf.nn.relu(hid_lin)
            
            #定义softmax回归模型，及损失方程
            y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
            loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

            global_step = tf.Variable(0)

            train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
            
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            ################################################################################
            saver = tf.train.Saver() #对模型定期做checkpoint，用于模型回复
            summary_op = tf.summary.merge_all()   #定义收集模型统计信息的操作
            init_op = tf.global_variables_initializer()  # 初始化所有变量

        # sv负责监控训练过程，构建模型检查点以及计算模型统计信息
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/tmp/train_logs",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        # 在sv中启动session
        #读入mnist训练数据
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        with sv.managed_session(server.target) as sess:  #以这种方式启动session
            # Loop until the supervisor shuts down or 1000000 steps have completed.
            step = 0
            while not sv.should_stop() and step < 100000:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # 在该处执行同步或异步运算
                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                train_feed = {x: batch_xs, y_:batch_ys}
                
                # 执行分布式TensorFlow模型训练
                _, step = sess.run([train_op, global_step], feed_dict=train_feed)
                
                #每隔100步长，验证模型精度
                if step % 100 == 0:
                    print("Done step %d" % step)
                    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
        
        # 停止tensorflow session
        sv.stop()

if __name__ == "__main__":
    tf.app.run()
