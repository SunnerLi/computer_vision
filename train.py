import tensorflow as tf
import numpy as np
import data_helper
from model import Model

loss_list = []

def record():
    with open('loss.txt', 'w') as f:
        for _loss in loss_list:
            f.write(str(_loss) + '\n')

if __name__ == '__main__':
    imgs, tags = data_helper.load()
    model = Model()
    sess = tf.Session()
    with sess.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(0.01)
        grad_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grad_and_vars, global_step=global_step)

        loss_summary = tf.scalar_summary("loss", model.loss)
        train_summary_op = tf.merge_summary([loss_summary])
        train_summary_writer = tf.train.SummaryWriter('summary/train', sess.graph)

        sess.run(tf.initialize_all_variables())

        for j in range(100):
            loss_sum = 0.0
            for i in range(len(imgs)):
                feed_dict = {
                    model.x: np.asarray([np.expand_dims(imgs[i], -1)]),
                    model.y: np.asarray([data_helper.indexTo01(tags[i])])
                }
                _loss, _, _summary, _step = sess.run([model.loss, train_op, train_summary_op, global_step], feed_dict=feed_dict)
                loss_sum += _loss

            current_step = tf.train.global_step(sess, global_step)
            train_summary_writer.add_summary(_summary, global_step=current_step)
            print "iter: ", j, "loss: ", loss_sum
            loss_list.append(loss_sum)
        
        saver = tf.train.Saver()
        saver.save(sess, "train_result.ckpt")

    record()