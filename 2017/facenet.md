# Facenet Source Code

```
Get Facenet(python) [here](https://github.com/davidsandberg/facenet)
```

Facenet主要源代码位于repo下的src目录，facenet.py为base文件，别的为其他算法或功能封装。

## Softmax Train

```
train_softmax.py
```
main函数，标注入口函数，整体运行流程都在里边，本文件中除了parse_arguments函数之外，都在其中调用。

parse_arguments函数，处理main函数所接收的参数以及其默认值。


## Main

一开始总是些各种定义和准备的方法，列举如下:

```
    # 加载关于引入的module
    network = importlib.import_module(args.model_def)
    # 获取当前时间作为记录的训练model的文件名
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    # 创建log文件
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    # 创建训练model文件
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    # 判断model文件夹是否存在
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # 调用facenet方法，将参数信息写入文件
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    # 制造随机种子数据
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    # 获取训练数据
    train_set = facenet.get_dataset(args.data_dir)
    if args.filter_filename:
        train_set = filter_dataset(train_set, os.path.expanduser(args.filter_filename), 
            args.filter_percentile, args.filter_min_nrof_images_per_class)
    # 训练数据种类
    nrof_classes = len(train_set)
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    # 新建model还是基于原有model
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)
    
    # 使用lfw的文件位置
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

```
接下来，开始构建图表，主要是对操作的定义:

```
    with tf.Graph().as_default():
        #根据之前保存的种子数据生成图级别的种子数据
        tf.set_random_seed(args.seed)
        # 定义global_step来记录训练步骤
        global_step = tf.Variable(0, trainable=False)
        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        # 如果数据为空，跳出
        assert len(image_list)>0, 'The dataset should not be empty'
        
        # 将labels转为tensor，用来与logits计算cross_entropy
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        # 生成队列index
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)
        # 定义出列op
        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')

        # placeholders
        # 学习曲线
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        # 每次train的样本数量
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        # 样本路径
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        # labels
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
        # 定义FIFO队列
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(1,), (1,)],
                                    shared_name=None, name=None)
        # 定义入列op，将样本和对应label入列
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')
        
        # 图片数据扩充处理
        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            # 图片出列
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                if args.random_rotate:
                    image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
    
                #pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        # 用images_and_labels来填充／新建队列
        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder, 
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        # 新建对象
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        
        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))
        
        print('Building training graph')
        
        # Build the inference graph
        # prelogits：inference的预测结果
        prelogits, _ = network.inference(image_batch, args.keep_probability, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
            weight_decay=args.weight_decay)
        # logits：prelogits经过全连接层之后的输出
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(args.weight_decay),
                scope='Logits', reuse=False)

        # embeddings：prelogtis做L2 Norm操作。
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Add center loss
        # prelogits_center_loss：添加center_loss，softmax_loss之上加入正则项，使得样本特征向量可以尽量聚集
        if args.center_loss_factor>0.0:
            prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
            # 保存regularization_loss
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        # 得到指数衰减之后的learning_rate
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # 根据label和logits 计算交叉熵
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        # 平均交叉熵
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        # 将平均交叉熵保存名为losses的collection中
        tf.add_to_collection('losses', cross_entropy_mean)
        
        # Calculate the total losses
        # prelogits_center_loss * center_loss_factor + cross_entropy_mean
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        # 定义训练操作
        train_op = facenet.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)
        
        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # GPU设置
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
```

## 接下来就是是在session中运行定义好的操作

```
        # 会话中运行操作
        with sess.as_default():
            # 使用model
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)

            # Training and validation loop
            print('Running training')
            epoch = 0
            # 定义训练循环和操作
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # 执行训练
                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
                    total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file)

                # 保存训练当前epoch结束时的model
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # 每一个epoch结束对训练结果进行评估
                if args.lfw_dir:
                    evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, 
                        embeddings, label_batch, lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer)

```
最终的返回结果是 训练完之后model的文件位置。

其次，train函数具体如下：

```
# 传入op
# index_dequeue_op
# enqueue_op
# global_step
# loss
# train_op
# summary_op
# regularization_losses
# learning_rate_schedule_file

def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
      loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    # 获取图片index队列
    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    
    # Enqueue one epoch of image paths and labels
    # expand_dims: 数据维度扩展，插入一个新坐标轴
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    # 维度扩展之后，执行入列操作
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        if (batch_number % 100 == 0):
            # 根据feed_dict传入参数变化，各个op的结果会产品相应变化
            err, _, step, reg_loss, summary_str = sess.run([loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step
```
在最后一行，有这样一行代码：

```
if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))
```

如果本文件被作为源文件来运行 __name__会被设置为 __main__，如果此文件被引入，那么__name__会被设置为引入的module name，这样就确保作为源文件运行时候运行main函数，但是被引入的情况下不会执行。也就是说，这行代码是文件执行的入口。

总结来说：训练流程如上所述，evaluate函数与train函数大同小异，执行文件之后，开始以main函数入口开始顺序执行：    
* main函数后，定义图之前：做引入，文件导入等准备操作  
* 图之后，会话之前：定义各种变量，操作以及操作所用到的参数，可以理解为做完所有准备工作。主要包括：  

```
1，图片，标签数据格式标准化，出列，入列。  
2，train_op，以及train_op的必要参数：total_loss, global_step等，及其计算过程中的变量。summary_op。  
3，tf.train.Saver新建。  
```
* session创建之后，开始运行：
```
1，运行各种初始化操作。  
2，tf.train.start_queue_runners 运行所有collection中的queue runner。  
3，一个epoch，epoch_size控制的循环，开始运行train方法。  
```
这就是整个softmax_train的流程。

## train_softmax.py 注解。
1，
```
# 避免已存在工具对import声明解读的混乱。
from __future__ import absolute_import 
# 引入python将来要实现的除法操作
from __future__ import division
# 将print方法引入低于3.0的python脚本
from __future__ import print_function

# 基本时间和日期类型
from datetime import datetime
# 各种操作系统接口
import os.path
# 实时时间获取以及转换
import time
# 允许使用某些属于系统或者解释器的变量或操作
import sys
# 产品伪随机数
import random
# 引入tensorflow
import tensorflow as tf
# 引入numpy
import numpy as np
# 更全的引入模块
import importlib
# 可以接收处理命令行输入
import argparse
# 引入facenet
import facenet
# 引入lfw
import lfw
# HDF5文件python接口，可以使用numpy处理大型文件
import h5py
# tensorflow的轻量级model定义，训练，评估库
import tensorflow.contrib.slim as slim
# 引入tensorflow.python中的 data_flow_ops模块 和 array_ops模块
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import array_ops
# 引入 tensorflow.python.framework 中的 ops模块
from tensorflow.python.framework import ops
```

2，
```
if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))
```

如果本文件被作为源文件来运行 __name__会被设置为 __main__，如果此文件被引入，那么__name__会被设置为引入的module name，这样就确保作为源文件运行时候运行main函数，但是被引入的情况下不会执行。也就是说，这行代码是文件执行的入口。

3，

```
  # Build the inference graph
  # inference 返回预测结果
  prelogits, _ = network.inference(image_batch, args.keep_probability, 
      phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
      weight_decay=args.weight_decay)
  # 增加全链接层，输出最终预测结果。
  logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None, 
          weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
          weights_regularizer=slim.l2_regularizer(args.weight_decay),
          scope='Logits', reuse=False)
  # 用预测值做L2 norm。
  embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

```

这段代码，产生了相当重要的3个值 prelogits, logits 和 embeddings。

### prelogtis 来源于network建立的推理的结果，netowrk的定义为：
```
network = importlib.import_module(args.model_def)
parser.add_argument('--model_def', type=str,
    help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
```
由此可见，network使用model文件中的inception_resnet_v1model建立推理从而获得初步预测值 prelogtis。

### logits： 是prelogti通过一个新建立的全连接层的计算结果。  
### embeddings：对prelogits做 l2_normalize 操作之后的结果。  
#### l2_normalize 使用L2范数对指定维度dim进行标准化，如果是多维，则对每个维度独立进行标准化。标准化可以有效防止模型过度拟合。  

4，
```
  # Add center loss
  if args.center_loss_factor>0.0:
      prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
      # 将regularization_loss加入collection
      tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

  # 根据label和logits 计算交叉熵
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=label_batch, logits=logits, name='cross_entropy_per_example')

  # 平均交叉熵
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  # 将平均交叉熵保存名为losses的collection中
  tf.add_to_collection('losses', cross_entropy_mean)

  # Calculate the total losses
  # prelogits_center_loss * center_loss_factor + cross_entropy_mean
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
```

这里通过计算得出total_loss  
total_loss => cross_entropy_mean, regularization_loss  
cross_entropy_mean, cross_entropy => labels, logits  
regularization_loss => prelogits_center_loss * args.center_loss_factor  


5， train_op 就是将上边计算出来的结果作为步骤保存起来，以便在session中调用。

```
  train_op = facenet.train(total_loss, global_step, args.optimizer, 
      learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)
```

6，新建inference的时候使用的是 inception_resnet_v1 model，这个model定义在当前 model/inception_resnet_v1.py中  
这个model应该是整个算法中的核心点。