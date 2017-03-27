import tensorflow as tf
import numpy as np
import data_helper
from model import Model

if __name__ == '__main__':
    imgs, tags = data_helper.load()
    model = Model()
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())

        for j in range(100):
            loss_sum = 0.0
            for i in range(len(imgs)):
                feed_dict = {
                    model.x: np.asarray([np.expand_dims(imgs[i], -1)]),
                    model.y: np.asarray([data_helper.indexTo01(tags[i])])
                }
                _loss, _ = sess.run([model.loss, model.optimizer], feed_dict=feed_dict)
                loss_sum += _loss
            print "iter: ", j, "loss: ", loss_sum
        
        saver = tf.train.Saver()
        saver.save(sess, "train_result.ckpt")