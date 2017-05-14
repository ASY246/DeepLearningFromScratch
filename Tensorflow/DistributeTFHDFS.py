import tensorflow as tf
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
#            filenames = ['hdfs://default/user/bdusr01/asy/mergeOneHot.csv']
            filenames = ['/home/bdusr01/asy/DL/DistributeTF/mergeOneHot.csv']
            filename_queue = tf.train.string_input_producer(filenames, shuffle=False)  #读入文件名序列
            reader = tf.TextLineReader()  #读取器，用于输出由换行符分隔的行，读文件名
            key, value = reader.read(filename_queue)  #返回reader产生的下一个记录
            lines = tf.decode_csv(value, record_defaults=[[0] for i in range(794)])
            features = tf.pack([*[lines[:-10]]])
            labels = tf.pack([*lines[-10:]])
            
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
            
            global_step = tf.Variable(0)
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
        with sv.managed_session(server.target) as sess:  #以这种方式启动session
            # Loop until the supervisor shuts down or 1000000 steps have completed.
            while not sv.should_stop():
#                coord = tf.train.Coordinator()  #创建一个协调器，管理线程
#                threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。
                for i in range(1000):
                    sess.run(train_step)
                    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    if i % 10 == 0:
                        print(sess.run(accuracy))
#                coord.request_stop()
#                coord.join(threads)

        # 停止tensorflow session
        sv.stop()

if __name__ == "__main__":
    tf.app.run()
