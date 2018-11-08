import numpy as np
import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
import scipy.ndimage
import json
import random
import shutil

# # Add additional import paths.
# sys.path.append(code_path)
# sys.path.append(os.path.join(code_path, 'model_train'))
# sys.path.append(os.path.join(code_path, 'Visualisation VTK'))

from tools.model_util import tf_transform_voxel_to_match_image, tf_random_crop_voxel_image
from tools.layer_util import keep_prob, conv3d, prelu, res_block_2d, res_block_3d, projection_unit
from tools.data_util import data_loader
from tools.resampling_voxel_grid import tf_rotation_resampling


with open(sys.argv[1], 'r') as fh:
    cfg = json.load(fh)

IMAGE_PATH       = cfg['image_path']
IMAGE_PATH_VALID = cfg['image_path_valid']
MODEL_PATH       = cfg['model_path']
SAMPLE_SAVE = cfg['sample_save']
MODEL_SAVE  = os.path.join(SAMPLE_SAVE, cfg['trained_model_name'])
LOGDIR = os.path.join(SAMPLE_SAVE, "log")

os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(cfg['gpu'])


def RenderNet(models_in, is_training, prob = 0.75, reuse =False):
    with tf.variable_scope("encoder"):
        batch_size = tf.shape(models_in)[0]

        with tf.variable_scope('e_conv1'):
            enc1 = prelu(conv3d(models_in, 8, kernel_size=[5, 5, 5], stride=[2, 2, 2],
                                              reuse=reuse, pad="SAME", scope='e_conv1', weight_initializer_type=tf.contrib.layers.xavier_initializer()))
            enc1 = tf.nn.dropout(enc1, keep_prob(prob, is_training))
        with tf.variable_scope('e_conv2'):
            enc2 = prelu(conv3d(enc1, 16, kernel_size=[3, 3, 3], stride=[1, 1, 2],
                                              reuse=reuse, pad="SAME", scope='e_conv2', weight_initializer_type=tf.contrib.layers.xavier_initializer()))
            enc2 = tf.nn.dropout(enc2, keep_prob(prob, is_training))
        with tf.variable_scope('e_conv3'):
            enc3 = prelu(conv3d(enc2, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1],
                                              reuse=reuse, pad="SAME", scope='e_conv3', weight_initializer_type=tf.contrib.layers.xavier_initializer()))
            enc3 = tf.nn.dropout(enc3, keep_prob(prob, is_training))

        shortcut = enc3

        res1_1 = res_block_3d(enc3,   32, scope='res1_1')
        res1_2 = res_block_3d(res1_1, 32, scope='res1_2')
        res1_3 = res_block_3d(res1_2, 32, scope='res1_3')
        res1_4 = res_block_3d(res1_3, 32, scope='res1_4')
        res1_5 = res_block_3d(res1_4, 32, scope='res1_5')
        res1_6 = res_block_3d(res1_5, 32, scope='res1_6')
        res1_7 = res_block_3d(res1_6, 32, scope='res1_7')
        res1_8 = res_block_3d(res1_7, 32, scope='res1_8')
        res1_9 = res_block_3d(res1_8, 32, scope='res1_9')
        res1_10 = res_block_3d(res1_9,32, scope='res1_10')

        with tf.variable_scope('res1_skip'):
            enc3_skip = conv3d(res1_10, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], pad="SAME", scope="con1_3X3", weight_initializer_type=tf.contrib.layers.xavier_initializer())
            enc3_skip = tf.add(tf.cast(enc3_skip, tf.float32), tf.cast(shortcut, tf.float32))

        # height = tf.shape(enc3_skip)[1]
        # width = tf.shape(enc3_skip)[2]
        # #Collapsing Z dimension
        # enc3_2d = tf.reshape(enc3_skip, [batch_size, height, width, 32 * 32])
        #
        # with tf.variable_scope('e_conv4'):
        #     enc4 = prelu(slim.conv2d(inputs=enc3_2d, num_outputs = 32 * 32, kernel_size=1, activation_fn=None, scope='e_conv4'))
        #     enc4 = tf.nn.dropout(enc4, keep_prob(prob, is_training))
        enc4 = projection_unit(enc3_skip)

        shortcut = enc4

        res2_1 = res_block_2d(enc4, 32 * 32, scope='res2_1')
        res2_2 = res_block_2d(res2_1, 32 * 32, scope='res2_2')
        res2_3 = res_block_2d(res2_2, 32 * 32, scope='res2_3')
        res2_4 = res_block_2d(res2_3, 32 * 32, scope='res2_4')
        res2_5 = res_block_2d(res2_4, 32 * 32, scope='res2_5')
        res2_6 = res_block_2d(res2_5, 32 * 32, scope='res2_6')
        res2_7 = res_block_2d(res2_6, 32 * 32, scope='res2_7')
        res2_8 = res_block_2d(res2_7, 32 * 32, scope='res2_8')
        res2_9 = res_block_2d(res2_8, 32 * 32, scope='res2_9')
        res2_10 = res_block_2d(res2_9, 32 * 32, scope='res2_10')

        with tf.variable_scope('res2_skip'):
            enc4_skip = slim.conv2d(res2_10, 32 * 32, kernel_size=3, stride=1, activation_fn=None,  scope="con1_3X3")
            enc4_skip = tf.add(tf.cast(enc4_skip, tf.float32), tf.cast(shortcut, tf.float32))

        with tf.variable_scope('e_conv5'):
            enc5 = prelu(slim.conv2d(inputs=enc4_skip, num_outputs = 32 * 16, kernel_size=(4,4), stride = 1, activation_fn=None, scope='e_conv5'))
            enc5 = tf.nn.dropout(enc5, keep_prob(prob, is_training))
        shortcut = enc5

        res3_1 = res_block_2d(enc5, 32 * 16, scope='res3_1')
        res3_2 = res_block_2d(res3_1, 32 * 16, scope='res3_2')
        res3_3 = res_block_2d(res3_2, 32 * 16, scope='res3_3')
        res3_4 = res_block_2d(res3_3, 32 * 16, scope='res3_4')
        res3_5 = res_block_2d(res3_4, 32 * 16, scope='res3_5')

        with tf.variable_scope('res3_skip'):
            enc5_skip = slim.conv2d(res3_5, 32 * 16, kernel_size=3, stride=1, activation_fn=None,  scope="con1_3X3")
            enc5_skip = tf.add(tf.cast(enc5_skip, tf.float32), tf.cast(shortcut, tf.float32))

        with tf.variable_scope('e_conv6'):
            enc6 = prelu(slim.conv2d(inputs=enc5_skip, num_outputs = 32 * 8, kernel_size=(4,4), stride = 1, activation_fn=None, scope='e_conv6'))
            enc6 = tf.nn.dropout(enc6, keep_prob(prob, is_training))

        with tf.variable_scope('e_conv7'):
            enc7 = prelu(slim.conv2d_transpose(enc6, 32 * 4, (4, 4), stride = 2, activation_fn=None, scope='e_conv7'))
            enc7 = tf.nn.dropout(enc7, keep_prob(prob, is_training))

        with tf.variable_scope('e_conv7_1'):
            enc7_1 = prelu(slim.conv2d_transpose(enc7, 32 * 4, (4, 4), stride = 1, activation_fn=None, scope='e_conv7_1'))
            enc7_1 = tf.nn.dropout(enc7_1, keep_prob(prob, is_training))

        with tf.variable_scope('e_conv8'):
            enc8 = prelu(slim.conv2d_transpose(enc7_1, 32 * 2, (4, 4), stride = 2, activation_fn=None, scope='e_conv8'))
            enc8 = tf.nn.dropout(enc8, keep_prob(prob, is_training))

        with tf.variable_scope('e_conv9'):
            enc9 = prelu(slim.conv2d_transpose(enc8, 32, (4, 4), stride = 2, activation_fn=None, scope='e_conv9'))
            enc9 = tf.nn.dropout(enc9, keep_prob(prob, is_training))

        with tf.variable_scope('e_conv10'):
            enc10 = prelu(slim.conv2d_transpose(enc9, 16, (4, 4), stride = 1, activation_fn=None, scope='e_conv10'))
            enc10 = tf.nn.dropout(enc10, keep_prob(prob, is_training))

        if cfg['is_greyscale'].lower() == "true":
            enc11 = slim.conv2d_transpose(enc10, 1, (4, 4), stride = 1, activation_fn=None, scope='e_conv11')
            output = tf.nn.sigmoid(enc11, name = "output")
        else:
            enc11 = slim.conv2d_transpose(enc10, 3, (4, 4), stride = 1, activation_fn=None, scope='e_conv11')
            output = tf.nn.sigmoid(enc11, name = "output")
        return output

#============================================================================================================================
#============================================================================================================================
graph=tf.Graph()
new_res = 128 #Size of the new voxel grid to embed the rotated input grid to make sure the it is not cut off after rotation.


with graph.as_default():
    model_in = tf.placeholder(shape=[None, 64, 64, 64, 1], dtype=tf.float32, name = "real_model_in")
    if cfg['is_greyscale'].lower() == "true":
        image_in = tf.placeholder(shape=[None, 512, 512, 1], dtype=tf.float32, name = "real_image_in")
    else:
        image_in = tf.placeholder(shape=[None, 512, 512, 3], dtype=tf.float32, name = "real_image_in")
    param_in = tf.placeholder(shape=[None, 3], dtype=tf.float32, name = "view_name")
    patch_size_in = tf.placeholder(tf.int32, [], name="patch_size")
    is_training = tf.placeholder(tf.bool, [], name="is_training")

    #Rotate input to camera coordinate
    rotated_models = tf_rotation_resampling(model_in, param_in, new_size = new_res)
    rotated_models = tf_transform_voxel_to_match_image(rotated_models) #Transform voxel array to match image array (xyz --> ijk)

    #Randomly crop the voxel grid input and the target image
    cropped_vox, cropped_img = tf.cond(is_training, lambda: tf_random_crop_voxel_image(rotated_models, image_in, patch_size=patch_size_in),
                                                    lambda: (tf.identity(rotated_models), tf.identity(image_in)))
    images_pred = RenderNet(models_in = cropped_vox, is_training=is_training, prob=cfg['keep_prob'])

    #Binary cross entropy if grey scale
    if cfg['is_greyscale'].lower() == "true":
        recon_loss = tf.reduce_mean(-tf.reduce_sum(cropped_img * tf.log(1e-6 + images_pred)
                                    + (1 - cropped_img) * tf.log(1e-6 + 1 - images_pred), [1, 2, 3]))
    else:
        recon_loss = tf.reduce_mean(tf.losses.mean_squared_error(cropped_img, images_pred))

    global_step = tf.Variable(0, name='global_step',trainable=False)
    learning_rate = tf.train.exponential_decay(cfg['e_eta'], global_step, cfg['decay_steps'], 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(recon_loss, global_step=global_step)

    tf.summary.scalar('Recon loss train', recon_loss)
    merged_summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    sess_saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOGDIR, graph=graph)


#======================================================================================================================
#======================================================================================================================

sv = tf.train.Supervisor(logdir=LOGDIR,
                         graph=graph,
                         is_chief=True,
                         saver=sess_saver,
                         summary_op=None,
                         summary_writer=summary_writer,
                         save_summaries_secs = 300,
                         save_model_secs=cfg['checkpoint_secs'],
                         global_step=global_step)

sess_config = tf.ConfigProto(allow_soft_placement=True)

L1_ALL = list()
train = True
with sv.managed_session(config=sess_config) as sess:
    with tf.device("/gpu:0"):
        while not sv.should_stop():
            if not os.path.exists(SAMPLE_SAVE):
                os.makedirs(SAMPLE_SAVE)

            shutil.copyfile(sys.argv[1], os.path.join(SAMPLE_SAVE, 'config.json'))
            train_writer = tf.summary.FileWriter(os.path.join(SAMPLE_SAVE, 'train'), graph=graph)
            if train:
                for epoch in range(cfg['max_epochs']):
                    print ("Epoch starting")
                    if epoch < 5:
                        batch_patch_size = new_res // 4
                    else:
                        batch_patch_size = new_res // 2

                    if cfg['is_greyscale'].lower() == "true":
                        train_loader = data_loader(cfg, img_path=IMAGE_PATH,
                                                        model_path=MODEL_PATH, flatten=True,
                                                        validation_mode=False,
                                                        img_res=512)
                    else:
                        train_loader = data_loader(cfg, img_path=IMAGE_PATH,
                                                        model_path=MODEL_PATH, flatten=False,
                                                        validation_mode=False,
                                                        img_res=512)
                    counter=0
                    print ("Loading data")
                    for real_images, real_models, real_params, real_names in train_loader:# Loop across chunks, READER object will grab the next chunk after every time a chunk is completed
                        print ("Data loaded")
                        chunk_start = time.time()
                        # Check input
                        real_images /= 255.0
                        num_batches = len(real_images) // cfg['batch_size']
                        for idx in range(num_batches):
                            batch_models = real_models[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            batch_images = real_images[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            batch_names  = real_names[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            batch_params = real_params[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]

                            feed_dict_train = {model_in: batch_models,
                                             image_in: batch_images,
                                             param_in:batch_params,
                                             patch_size_in:batch_patch_size,
                                             is_training: True}

                            train, summary, step, totalLoss = sess.run([optimizer, merged_summary_op, global_step, recon_loss],
                                                feed_dict = feed_dict_train)
                            train_writer.add_summary(summary, global_step=step)
                            print("Step {0} Loss {1}".format(step, totalLoss))

                            if step % 600 == 0:
                                rendered_samples, out_cropped_im = sess.run([images_pred, cropped_img],
                                                         feed_dict = feed_dict_train)
                                out_cropped_im = np.clip(255 * out_cropped_im, 0, 255).astype(np.uint8)
                                rendered_samples = np.clip(255 * rendered_samples, 0, 255).astype(np.uint8)
                                print( "Epoch {2} {0} BCE Loss: {1}".format(step, totalLoss, epoch))
                                for i in range(1):
                                    idx=random.randint(0, cfg['batch_size'] - 1)
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "{0}_train_target_{1}_patch.png".format(batch_names[idx], step)), np.squeeze(out_cropped_im[idx]))
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "{0}_train_{1}_patch.png".format(batch_names[idx], step)), np.squeeze(rendered_samples[idx]))


                    # # #
                    save_path = sess_saver.save(sess, MODEL_SAVE)
                    if cfg['is_greyscale'].lower() == "true":
                        valid_loader = data_loader(cfg, img_path=IMAGE_PATH,
                                                        model_path=MODEL_PATH, flatten=True,
                                                        validation_mode=True,
                                                        img_res=512)
                    else:
                        valid_loader = data_loader(cfg, img_path=IMAGE_PATH,
                                                        model_path=MODEL_PATH, flatten=False,
                                                        validation_mode=True,
                                                        img_res=512)
                    counter = 0
                    L1_valid = 0.
                    for real_images, real_models, real_params, real_names in valid_loader:
                        real_images /= 255.0
                        num_batches = len(real_images) // cfg['batch_size']
                        for idx in range(num_batches):
                            valid_batch_models = real_models[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            valid_batch_images = real_images[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            valid_batch_names  = real_names[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            valid_batch_params = real_params[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            feed_dict_valid = {model_in:valid_batch_models,
                                             image_in: valid_batch_images,
                                             param_in: valid_batch_params,
                                             patch_size_in: batch_patch_size,
                                             is_training:False}
                            rendered_samples, out_cropped_im = sess.run([images_pred, cropped_img], feed_dict=feed_dict_valid)
                            out_cropped_im = np.clip(255 * out_cropped_im, 0, 255).astype(np.uint8)
                            rendered_samples = np.clip(255 * rendered_samples, 0, 255).astype(np.uint8)

                            if counter % 600  == 0:
                                for i in range(1):
                                    index = random.randint(0, cfg['batch_size'] - 1)
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "VALID_{0}_target_{1}.png".format(valid_batch_names[index], epoch)), np.squeeze(out_cropped_im[index]))
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "VALID_{0}_pred_{1}.png".format(valid_batch_names[index], epoch)), np.squeeze(rendered_samples[index]))
                                    print("Screenshot is saved")
                            validation_accuracy = np.mean(np.absolute(valid_batch_images - rendered_samples))
                            print("Validation accuracy {0}".format(validation_accuracy))
                            L1_valid += validation_accuracy
                            counter +=1

                    # Across test set (multi mini-batches) (after 1 epoch)
                    L1_valid = L1_valid / counter
                    L1_ALL.append(L1_valid)
                    np.savez(os.path.join(SAMPLE_SAVE, "L1 All.txt"), L1_ALL)
                    print("Validation done")

                sv.request_stop()