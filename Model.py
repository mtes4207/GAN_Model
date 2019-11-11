import tensorflow as tf
from tensorflow.layers import batch_normalization
from os import listdir
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage import transform



def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def W_var( name, Size):
    init = tf.truncated_normal_initializer(stddev=0.2)
    return tf.get_variable('g_W' + name,dtype=tf.float32,shape=Size,initializer=init)

def b_var(name, Size):
    init = tf.truncated_normal_initializer(stddev=0)
    return tf.get_variable('g_b' + name,dtype=tf.float32,shape=Size,initializer=init)

def Dnn(x, layer_units, name, normalization=True,is_training=True):
    input_dim = x.get_shape()[1]
    g_w = W_var(name , [input_dim ,layer_units])
    g_b = b_var(name , [layer_units])
    g = tf.matmul( x, g_w) + g_b
    if normalization:
      g = batch_normalization( g, training=is_training)
    return tf.sigmoid(g)


def discriminator(images, reuse1=None):
    with tf.variable_scope("discriminator", reuse=reuse1):
        keep_prob = 1
        init = tf.truncated_normal_initializer(stddev=0.4)
        ## conv1 layer ##
        W_conv1 = tf.get_variable('d_wc1', [5, 5, 3, 60], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.4))
        b_conv1 = tf.get_variable('d_bc1', [60], dtype=tf.float32, initializer=init)
        h_conv1 = conv2d(images, W_conv1) + b_conv1
        h_conv1 = tf.nn.relu(batch_normalization(h_conv1, name='cn1',training =True))
        h_pool1 = max_pool_2x2(h_conv1)  # 38*38*30

        ## conv2 layer ##
        W_conv2 = tf.get_variable('d_wc2', [5, 5, 60, 120], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.4))
        b_conv2 = tf.get_variable('d_bc2', [120], dtype=tf.float32, initializer=init)
        h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
        h_conv2 = tf.nn.relu(batch_normalization(h_conv2, name='cn2',training =True))
        h_pool2 = max_pool_2x2(h_conv2)  # 19*19*120

        ## func1 layer ##
        W_fc1 = tf.get_variable('d_wf1', [19 * 19 * 120, 1000], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc1 = tf.get_variable('d_bf1', [1000], dtype=tf.float32, initializer=init)
        h_pool1_flat = tf.reshape(h_pool2, [-1, 19 * 19 * 120])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)


        ## func2 layer ##
        W_fc2 = tf.get_variable('d_wf2', [1000, 500], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc2 = tf.get_variable('d_bf2', [500], dtype=tf.float32, initializer=tf.constant_initializer(0))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc2 = tf.nn.dropout(h_fc2, keep_prob)

        ## func3 layer ##
        W_fc3 = tf.get_variable('d_wf3', [500, 1], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
        b_fc3 = tf.get_variable('d_bf3', [1], dtype=tf.float32, initializer=tf.constant_initializer(0))
        h_fc3 = tf.matmul(h_fc2 ,W_fc3) + b_fc3
        y_hat = tf.sigmoid(h_fc3)
        return y_hat

def generator(z, batch_size, z_dim, test_train, reuse1=None):
    with tf.variable_scope("generator",reuse=reuse1):
        init = tf.truncated_normal_initializer(stddev=0.1)
        layer1 = Dnn( z, 500,"1",is_training=test_train)
        layer2 = Dnn( layer1, 1000, "2",is_training=test_train)
        layer3 = Dnn( layer2, 10000, "3",is_training=test_train)
        g = tf.reshape( layer3, [-1,10,10,100])

        # Generate 50 features
        g_w1 = tf.get_variable('g_cw1', [2, 2, 50, 100], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.5))
        g_b1 = tf.get_variable('g_cb1', [50], initializer=init)
        g1 = tf.nn.conv2d_transpose(g, g_w1, [batch_size, 19, 19, 50], strides=[1, 2, 2, 1], padding='SAME')
        g1 = g1 + g_b1
        g1 = tf.nn.relu(g1)
        g1 = batch_normalization(g1, name='bn1', training=test_train)

        # Generate 25 features
        g_w2 = tf.get_variable('g_cw2', [2, 2, 25, 50], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.5))
        g_b2 = tf.get_variable('g_cb2', [25], initializer=init)
        g2 = tf.nn.conv2d_transpose(g1, g_w2, [batch_size, 38, 38, 25], strides=[1, 2, 2, 1], padding='VALID')
        g2 = g2 + g_b2
        g2 = tf.nn.relu(g2)
        g2 = batch_normalization(g2, name='bn2', training = test_train)

        # final features
        g_w3 = tf.get_variable('g_cw3', [2, 2, 3, 25], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=1))
        g_b3 = tf.get_variable('g_cb3', [3], initializer=init)
        g3 = tf.nn.conv2d_transpose(g2, g_w3, [batch_size, 76, 76, 3], strides=[1, 2, 2, 1], padding='VALID')
        g3 = g3 + g_b3
        g3 = tf.sigmoid(g3)
        return g3

def load_data(mypath):
    file=listdir(mypath)
    for i in file:
        aa=img.imread(mypath+"/%s"%i)
        aa=transform.resize(aa,(76,76), mode='constant')[np.newaxis,:,:,:]
        if i is file[0]:
            out=aa
        else:
            out=np.concatenate((out, aa))
    print(out.shape)
    return out


def Generate_img(num, z_beta, name, path, test_train):
    z = tf.placeholder(tf.float32, [None,80], name='z1')
    Gz = generator(z, 225, 80,  test_train=test_train,reuse1=tf.AUTO_REUSE)
    count = 1
    saver = tf.train.Saver()
    plt.figure(figsize=[40, 40])
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, path)
        G_image = sess.run(Gz, {z: z_beta})
        for i in G_image:
            a = transform.resize(i, (512, 512), mode='constant')
            plt.subplot(25, 25, count).imshow(a)
            plt.axis("off")
            count += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    #plt.savefig("drive/My Drive/GAN_model/G_image/%s%s.png"%(name,num), bbox_inches='tight')