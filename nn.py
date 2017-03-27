import tensorflow as tf
import numpy as np
import data_helper
from model import Model
import cv2

test_img_name = 'zero.png'

def formTensor(img=cv2.imread(test_img_name, 0)):
    height, width = np.shape(img)
    x_stride_num = (width - 8) / 2 + 1
    y_stride_num = (height - 8) / 2 + 1
    
    tensor = []
    print "please wait for a while to form the tensor..."
    for i in range(y_stride_num):
        for j in range(x_stride_num):
            roi = img[2*i:2*i+8, 2*j:2*j+8]
            # print "y: ", 2*i, '~', 2*i+8, '\tx: ', 2*j, '~', 2*j+8
            tensor.append(np.expand_dims(roi, -1))

            # Copy Image
            draw_paper = np.zeros([np.shape(img)[0], np.shape(img)[1]])
            for k in range(np.shape(img)[0]):
                for l in range(np.shape(img)[1]):
                    draw_paper[k][l] = img[k][l]

            cv2.rectangle(draw_paper, (2*i, 2*j), (2*i+8, 2*j+8), (0, 0, 200))
            #cv2.waitKey(0)

    print "Finish forming!"
    print "Tensor size: ", np.shape(np.asarray(tensor))
    return np.asarray(tensor)

def test():
    imgs, tags = data_helper.load()
    model = Model()

    saver = tf.train.Saver()
    sess = tf.Session()
    with sess.as_default():
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        sess.run(tf.global_variables_initializer())

        # Testing
        imgs, tags = data_helper.load_test()
        for i in range(len(imgs)):
            feed_dict = {
                model.x: np.asarray([np.expand_dims(imgs[i], -1)]),
                model.y: np.asarray([data_helper.indexTo01(tags[i])])
            }
            _output = sess.run(model.output, feed_dict=feed_dict)
            print "tag: ", tags[i], "\toutput: ", _output

def work(tensor):
    result = [0] * len(tensor)
    model = Model()
    saver = tf.train.Saver()
    sess = tf.Session()
    with sess.as_default():
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        feed_dict = {
            model.x: tensor
        }
        _output = sess.run(model.output, feed_dict=feed_dict)

        for i in range(len(_output)):
            if _output[i][0] > _output[i][1]:
                result[i] = 0
            else:
                result[i] = 1
    return result

def draw(result, img):
    result_img = np.zeros([np.shape(img)[0], np.shape(img)[1], 3])
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            result_img[i][j] = img[i][j]

    height, width = np.shape(img)
    x_stride_num = (width - 8) / 2 + 1
    y_stride_num = (height - 8) / 2 + 1

    count = 0
    for i in range(y_stride_num):
        for j in range(x_stride_num):
            #print "point1: (", i*2, ",", j*2, "\tpoint2: (", 2*i+8, ',', 2*j+8
            if result[count] == 1:
                #cv2.rectangle(result_img, (i*2, j*2), (2*i+8, 2*j+8), (0, 0, 200))
                cv2.rectangle(result_img, (j*2, i*2), (2*j+8, 2*i+8), (0, 0, 200))
                #cv2.imwrite('wrong'+str(i*100+j)+'.png', img[2*i:2*i+8, 2*j:2*j+8])
            else:
                #cv2.rectangle(result_img, (i*2, j*2), (2*i+8, 2*j+8), (200, 0, 0))
                pass
            #cv2.imshow('result', result_img)
            #cv2.waitKey(0)
            count += 1

    cv2.imwrite('draw_result.png', result_img)

if __name__ == '__main__':
    tensor = formTensor()
    result_list = work(tensor)
    draw(result_list)