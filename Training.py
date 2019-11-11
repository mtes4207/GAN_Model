from Model import *

if __name__=="__main__":
    tf.reset_default_graph()
    z = tf.placeholder(tf.float32, [None, 80], name='z')
    x = tf.placeholder(tf.float32, shape=[None, 76, 76, 3], name='x')
    z_batch = np.load("drive/My Drive/GAN_model/z_batch1.npy")
    x_batch = load_data("drive/My Drive/GAN_model/data")

    l_path = "Path"  ##load tf path
    s_path = "Path"  ##save tf path

    Gz = generator(z, 225, 80, True, reuse1=tf.AUTO_REUSE)
    Dg = discriminator(Gz, reuse1=tf.AUTO_REUSE)
    Dx = discriminator(x, reuse1=tf.AUTO_REUSE)

    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if 'd_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]

    ## train discriminator
    lossG = -tf.reduce_mean(tf.log(1 - Dg))
    lossR = -tf.reduce_mean(tf.log(Dx))
    loss = lossG + lossR

    ## train generator
    G_loss = -tf.reduce_mean(tf.log(Dg))

    with tf.variable_scope("Loss_Train", reuse=tf.AUTO_REUSE):
        D_loss_train = tf.train.AdamOptimizer(0.0003).minimize(loss, var_list=d_vars)
        G_loss_train = tf.train.AdamOptimizer(0.001).minimize(G_loss, var_list=g_vars)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            training_op = G_loss_train

    ### Start Training
    num = 1
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, l_path)
        for i in range(1, 6):

            # training Discrimator
            while np.mean(sess.run(Dg, {z: z_batch})) > 0.05 or np.mean(sess.run(Dx, {x: x_batch})) < 0.95:
                sess.run(D_loss_train, {x: x_batch, z: z_batch})
                print("Real imagine score : %.2f " % (np.mean(sess.run(Dx, {x: x_batch})) * 100))
                print("Generate imagine score : %.2f " % (np.mean(sess.run(Dg, {z: z_batch})) * 100), "\n")

            # training Generator
            for u in range(1500):
                g_score = sess.run(Dg, {z: z_batch})
                sess.run(training_op, {z: z_batch})
                sess.run(extra_update_ops, {x: x_batch, z: z_batch})
                print("Generate imagine score : %.2f " % (g_score[102] * 100))
                print("Generate imagine score : %.2f " % (np.mean(g_score) * 100))
                print("-------------------------------")

                #show  img
                if u % 20 == 0:
                    G_image = sess.run(Gz, {z: z_batch})
                    a = G_image[102, :, :, :]
                    print(u)
                    plt.imshow(a)
                    plt.axis("off")
                    plt.show()

                # Save
                if np.mean(g_score) >= 0.9:
                    save_path = saver.save(sess, s_path)
                    print("SAVE - no.%s" % i)
                    print("===================")
                    print("===================")
                    break

                if u == 1499:
                    save_path = saver.save(sess, s_path)
                    print("SAVE - no.%s" % i)
                    print("===================")
                    print("===================")
                    Generate_img(num, z_batch, "emt ", s_path, True)
                    sys.exit()
    # plt.show
    Generate_img(num, z_batch, "G_img ", s_path, True)