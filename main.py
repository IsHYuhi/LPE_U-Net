import argparse
import random
import tensorflow as tf
import numpy as np
import os
import easydict

from util import loader as ld
from util import model
from util import repoter as rp


#check pointをセーブするかどうか
SAVE = True

def load_dataset(train_rate, size=(128, 128)):
    loader = ld.Loader(dir_original="data_set/train_images",
                       dir_segmented="data_set/train_annotations",
                       init_size=size)
    return loader.load_train_test(train_rate=train_rate, shuffle=False)


def train(parser):
    assert parser.num != None, 'please input the number of images. i.e. python3 main.py -n xxx'
    #imageの枚数
    NUM = parser.num
    #画像の学習サイズ
    size = tuple(parser.size)
    #trainrate
    trainrate = parser.trainrate

    train, test = load_dataset(train_rate=trainrate, size=size)

    valid = train.devide(int(NUM * (trainrate - ((1-trainrate)))), int(NUM*trainrate))
    test = test.devide(0, int(NUM * (1-trainrate)))

    #保存ファイル
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure("Accuracy", ("epoch", "accuracy"), ["train", "test"])
    loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

    #restoreするモデル名
    CONTINUE = parser.restore

    #GPU
    gpu = parser.gpu
    print(gpu)

    #model
    model_unet = model.UNet(size=size, l2_reg=parser.l2reg).model

    #誤差関数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    #精度
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #gpu config
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7, visible_device_list="", allow_growth=True), #device_count={'GPU': 0},
                                log_device_placement=False, allow_soft_placement=True)
    if gpu:
        sess = tf.InteractiveSession(config=gpu_config)
        print("gpu mode")
    else:
        sess = tf.InteractiveSession()
        print("cpu mode")

    tf.global_variables_initializer().run()
    #parameter
    epochs = parser.epoch
    batch_size = parser.batchsize

    t_images_original = test.images_original

    saver = tf.train.Saver(max_to_keep=100)
    if  not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

    if CONTINUE is not None:
        saver.restore(sess, "./checkpoint/"+ CONTINUE)
        print("restored")

    for epoch in range(epochs):
        for batch in train(batch_size=batch_size):
            # バッチデータ
            images_original = batch.images_original
            inputs = images_original
            teacher = batch.images_segmented

            sess.run(train_step, feed_dict={model_unet.inputs: inputs, model_unet.teacher: teacher,
                                            model_unet.is_training: True})

        train_images_original = train.images_original

        # 評価
        accuracy_train = 0
        loss_train = 0
        accuracy_test =0
        loss_test = 0

        if epoch % 1 == 0:
            num_batch = 0
            for batchs in valid(batch_size=batch_size):
                num_batch += 1
                images = batchs.images_original
                segmented = batchs.images_segmented
                accuracy_train += sess.run(accuracy, feed_dict={model_unet.inputs: images, model_unet.teacher: segmented,
                  model_unet.is_training: False})
                loss_train += sess.run(cross_entropy, feed_dict={model_unet.inputs: images, model_unet.teacher: segmented,
                  model_unet.is_training: False})
            accuracy_train /= num_batch
            loss_train /= num_batch

            num_batch = 0
            for batchs in test(batch_size=batch_size):
                num_batch += 1
                images = batchs.images_original
                segmented = batchs.images_segmented
                accuracy_test += sess.run(accuracy, feed_dict={model_unet.inputs: images, model_unet.teacher: segmented,
                 model_unet.is_training: False})
                loss_test += sess.run(cross_entropy, feed_dict={model_unet.inputs: images, model_unet.teacher: segmented,
                 model_unet.is_training: False})
            accuracy_test /= num_batch
            loss_test /= num_batch

            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
            accuracy_fig.add([accuracy_train, accuracy_test], is_update=True)
            loss_fig.add([loss_train, loss_test], is_update=True)
            if epoch % 1 == 0:
                idx_train = random.randrange(int(NUM*trainrate))
                idx_test = random.randrange(int(NUM*(1-trainrate)))
                outputs_train = sess.run(model_unet.outputs,
                                         feed_dict={model_unet.inputs: [train_images_original[idx_train]],
                                                    model_unet.is_training: False})
                outputs_test = sess.run(model_unet.outputs,
                                        feed_dict={model_unet.inputs: [t_images_original[idx_test]],
                                                   model_unet.is_training: False})
                train_set = [train_images_original[idx_train], outputs_train[0], train.images_segmented[idx_train]]
                test_set = [t_images_original[idx_test], outputs_test[0], test.images_segmented[idx_test]]
                reporter.save_image_from_ndarray(train_set, test_set, train.palette, epoch,
                                                 index_void=0)#index_void = background
        if epoch % 10 == 0:
            if SAVE:
                save_path = saver.save(sess, "./checkpoint/save_model_epoch_"+str(epoch)+"_.ckpt")
    save_path = saver.save(sess, "./checkpoint/save_model_done.ckpt")

    #modelの評価

    for batchs in test(batch_size=batch_size):
        images = batchs.images_original
        segmented = batchs.image_segmented
        accuracy_test += sess.run(accuracy, feed_dict={model_unet.inputs: images, model_unet.teacher: segmented,
        model_unet.is_training: False})
        loss_test += sess.run(cross_entropy, feed_dict={model_unet.inputs: images, model_unet.teacher: segmented,
        model_unet.is_training: False})
    accuracy_test /= num_batch
    loss_test /= num_batch

    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)

    sess.close()


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPUs')
    parser.add_argument('-e', '--epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=4, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.7, help='Training rate')
    parser.add_argument('-l', '--l2reg', type=float, default=0.0001, help='L2 regularization')
    parser.add_argument('-n', '--num', type=int, default=None, help='the number of images')
    parser.add_argument('-s', '--size', nargs=2, type=int, default=(128, 128), help='image size')
    parser.add_argument('-r', '--restore', default=None, help='name of the checkpoint file')
    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
