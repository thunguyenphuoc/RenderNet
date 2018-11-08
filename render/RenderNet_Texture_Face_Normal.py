import numpy as np
import os
import tensorflow as tf
import sys
import time
import scipy.ndimage
import json
import random
import shutil

from tools.model_util import tf_transform_voxel_to_match_image, tf_random_crop_voxel_texture_image_normal, load_weights
from tools.data_util import data_loader_image_texture_normal_face
from tools.layer_util import conv3d_transpose, conv3d, prelu, fully_connected, conv2d, conv2d_transpose, res_block_3d, res_block_2d, keep_prob, projection_unit
from tools.resampling_voxel_grid import tf_rotation_resampling


#=======================================================================================================================
with open(sys.argv[1], 'r') as fh:
    cfg = json.load(fh)

IMAGE_PATH       = cfg['image_path']
IMAGE_PATH_VALID = cfg['image_path_valid']
NORMAL_PATH      = cfg['normal_path']
TEXTURE_PATH     = cfg['texture_path']
MODEL_PATH       = cfg['model_path']
SAMPLE_SAVE      = cfg['sample_save']
MODEL_SAVE       = os.path.join(SAMPLE_SAVE, cfg['trained_model_name'])
LOGDIR           = os.path.join(SAMPLE_SAVE, "log")


os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(cfg['gpu'])
#=======================================================================================================================

def decoder_texture(z_in):
    with tf.variable_scope("texture_encoder"):
        batch_size = tf.shape(z_in)[0]
        with tf.variable_scope('e_tex_fc1'):
            zP = prelu((fully_connected(z_in, 4 * 4 * 4 * 512)))
            z_resize = tf.reshape(zP, [batch_size, 32, 32, 32, 4])
        with tf.variable_scope('e_tex_conv0'):
            conv0 = prelu(conv3d_transpose(z_resize, 4, kernel_size=[4, 4, 4], stride=[1, 1, 1]))
        with tf.variable_scope('e_tex_conv1'):
            conv1 = prelu(conv3d_transpose(conv0, 8, kernel_size=[4, 4, 4], stride=[2, 2, 2]))
        with tf.variable_scope('e_tex_conv2'):
            conv2 = prelu(conv3d(conv1, 4, kernel_size=[4, 4, 4], stride=[1, 1, 1]))
        return conv2

def RenderNet(models_in, prob = 0.75, reuse=False):
    batch_size = tf.shape(models_in)[0]
    with tf.variable_scope("encoder"):
        with tf.variable_scope('e_conv1'):
            enc1 = prelu(conv3d(models_in, 16, kernel_size=[5, 5, 5], stride=[2, 2, 2],
                                 reuse=reuse, pad="SAME", scope='e_conv1',
                                 weight_initializer_type=tf.contrib.layers.xavier_initializer()))
            enc1 = tf.nn.dropout(enc1, keep_prob(prob, is_training))
        with tf.variable_scope('e_conv2'):
            enc2 = prelu(conv3d(enc1, 32, kernel_size=[3, 3, 3], stride=[1, 1, 2],
                                reuse=reuse, pad="SAME", scope='e_conv2',
                                weight_initializer_type=tf.contrib.layers.xavier_initializer()))
            enc2 = tf.nn.dropout(enc2, keep_prob(prob, is_training))
        with tf.variable_scope('e_conv3'):
            enc3 = prelu(conv3d(enc2, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1],
                                reuse=reuse, pad="SAME", scope='e_conv3',
                                weight_initializer_type=tf.contrib.layers.xavier_initializer()))
            enc3 = tf.nn.dropout(enc3, keep_prob(prob, is_training))

        shortcut = enc3
        res1_1 = res_block_3d(enc3, 16, scope='res1_1')
        res1_2 = res_block_3d(res1_1, 16, scope='res1_2')
        res1_3 = res_block_3d(res1_2, 16, scope='res1_3')
        res1_4 = res_block_3d(res1_3, 16, scope='res1_4')
        res1_5 = res_block_3d(res1_4, 16, scope='res1_5')
        res1_6 = res_block_3d(res1_5, 16, scope='res1_6')
        res1_7 = res_block_3d(res1_6, 16, scope='res1_7')
        res1_8 = res_block_3d(res1_7, 16, scope='res1_8')
        res1_9 = res_block_3d(res1_8, 16, scope='res1_9')
        res1_10 = res_block_3d(res1_9, 16, scope='res1_10')
        with tf.variable_scope('res1_skip'):
            enc3_skip = conv3d(res1_10, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], pad="SAME", scope="con1_3X3", weight_initializer_type=tf.contrib.layers.xavier_initializer())
            enc3_skip = tf.add(tf.cast(enc3_skip, tf.float32), tf.cast(shortcut, tf.float32))

        # #===============================================================================================================
        # #PROJECTION UNIT
        # #===============================================================================================================
        # #Collapsing Z dimension
        # height = tf.shape(enc3_skip)[1]
        # width = tf.shape(enc3_skip)[2]
        # enc3_2d = tf.reshape(enc3_skip, [batch_size, height, width, 32 * 16])
        # #Followed by 1x1 convolution
        # with tf.variable_scope('e_conv4'):
        #     enc4 = prelu(conv2d(enc3_2d, 32 * 16, kernel_size=[1, 1], stride=[1, 1], scope='e_conv4', weight_initializer_type=tf.contrib.layers.xavier_initializer()))
        #     enc4 = tf.nn.dropout(enc4, keep_prob(prob, is_training))
        # #===============================================================================================================

        enc4 = projection_unit(enc3_skip)
        shortcut = enc4
        res2_1 = res_block_2d(enc4, 32 * 16, scope='res2_1')
        res2_2 = res_block_2d(res2_1, 32 * 16, scope='res2_2')
        res2_3 = res_block_2d(res2_2, 32 * 16, scope='res2_3')
        res2_4 = res_block_2d(res2_3, 32 * 16, scope='res2_4')
        res2_5 = res_block_2d(res2_4, 32 * 16, scope='res2_5')
        res2_6 = res_block_2d(res2_5, 32 * 16, scope='res2_6')
        res2_7 = res_block_2d(res2_6, 32 * 16, scope='res2_7')
        res2_8 = res_block_2d(res2_7, 32 * 16, scope='res2_8')
        res2_9 = res_block_2d(res2_8, 32 * 16, scope='res2_9')
        res2_10 = res_block_2d(res2_9, 32 * 16, scope='res2_10')
        with tf.variable_scope('res2_skip'):
            enc4_skip = conv2d(res2_10, 32 * 16, kernel_size=[3, 3], stride=[1, 1], scope="con1_3X3", weight_initializer_type=tf.contrib.layers.xavier_initializer())
            enc4_skip = tf.add(tf.cast(enc4_skip, tf.float32), tf.cast(shortcut, tf.float32))

        with tf.variable_scope('e_conv5'):
            enc5 = prelu(conv2d(enc4_skip, 32 * 8, kernel_size=[4,4], stride = [1, 1], scope='e_conv5', weight_initializer_type=tf.contrib.layers.xavier_initializer()))
            enc5 = tf.nn.dropout(enc5, keep_prob(prob, is_training))

        shortcut = enc5
        res3_1 = res_block_2d(enc5, 32 * 8, scope='res3_1')
        res3_2 = res_block_2d(res3_1, 32 * 8, scope='res3_2')
        res3_3 = res_block_2d(res3_2, 32 * 8, scope='res3_3')
        res3_4 = res_block_2d(res3_3, 32 * 8, scope='res3_4')
        res3_5 = res_block_2d(res3_4, 32 * 8, scope='res3_5')
        with tf.variable_scope('res3_skip'):
            enc5_skip = conv2d(res3_5, 32 * 8, kernel_size=[3, 3], stride=[1, 1],  scope="con1_3X3", weight_initializer_type=tf.contrib.layers.xavier_initializer())
            enc5_skip = tf.add(tf.cast(enc5_skip, tf.float32), tf.cast(shortcut, tf.float32))

        with tf.variable_scope("Image"):
            with tf.variable_scope('e_conv6_1'):
                enc6_1 = prelu(conv2d(enc5_skip, 32 * 4, kernel_size=[4, 4], stride=[1, 1], scope='e_conv6_1', weight_initializer_type=tf.contrib.layers.xavier_initializer()))
                enc6_1 = tf.nn.dropout(enc6_1, keep_prob(prob, is_training))
            with tf.variable_scope('e_conv7_1'):
                enc7_1 = prelu(conv2d_transpose(enc6_1, 32 * 2, [4, 4], stride=[2, 2], scope='e_conv7_2', weight_initializer_type=tf.contrib.layers.xavier_initializer()))
                enc7_1 = tf.nn.dropout(enc7_1, keep_prob(prob, is_training))
            with tf.variable_scope('e_conv8_1'):
                enc8_1 = prelu(conv2d_transpose(enc7_1, 32, [4, 4], stride=[2, 2], weight_initializer_type=tf.contrib.layers.xavier_initializer()))
                enc8_1 = tf.nn.dropout(enc8_1, keep_prob(prob, is_training))
            with tf.variable_scope('e_conv9_1'):
                enc9_1 = prelu(conv2d_transpose(enc8_1, 16, [4, 4], stride=[2, 2], weight_initializer_type=tf.contrib.layers.xavier_initializer()))
                enc9_1 = tf.nn.dropout(enc9_1, keep_prob(prob, is_training))
            with tf.variable_scope('e_conv10_1'):
                enc10_1 = conv2d_transpose(enc9_1, 3, [4, 4], stride=[1, 1], weight_initializer_type=tf.contrib.layers.xavier_initializer())
                enc10_1 = tf.nn.sigmoid(enc10_1, name="encoder_output")

        with tf.variable_scope("Normal"):
            with tf.variable_scope('e_conv6_2'):
                enc6_2 = prelu(conv2d(enc5_skip, 32 * 4, kernel_size=[4, 4], stride=[1,1], scope='e_conv6_2', weight_initializer_type=tf.contrib.layers.xavier_initializer()))
                enc6_2 = tf.nn.dropout(enc6_2, keep_prob(prob, is_training))
            with tf.variable_scope('e_conv7_2'):
                enc7_2 = prelu(conv2d_transpose(enc6_2, 32 * 2, [4, 4], stride=[2, 2], scope='e_conv7_2', weight_initializer_type=tf.contrib.layers.xavier_initializer()))
                enc7_2 = tf.nn.dropout(enc7_2, keep_prob(prob, is_training))
            with tf.variable_scope('e_conv8_2'):
                enc8_2 = prelu(conv2d_transpose(enc7_2, 32, [4, 4], stride=[2, 2], scope='e_conv8_2', weight_initializer_type=tf.contrib.layers.xavier_initializer()))
                enc8_2 = tf.nn.dropout(enc8_2, keep_prob(prob, is_training))
            with tf.variable_scope('e_conv9_2'):
                enc9_2 = prelu(conv2d_transpose(enc8_2, 16, [4, 4], stride=[2, 2], scope='e_conv9_2', weight_initializer_type=tf.contrib.layers.xavier_initializer()))
                enc9_2 = tf.nn.dropout(enc9_2, keep_prob(prob, is_training))
            with tf.variable_scope('e_conv10_2'):
                enc10_2 = conv2d_transpose(enc9_2, 3, [4, 4], stride=[1, 1], scope='e_conv10_2', weight_initializer_type=tf.contrib.layers.xavier_initializer())
                enc10_2 = tf.nn.sigmoid(enc10_2, name="encoder_output")

        return enc10_1, enc10_2


#============================================================================================================================
#============================================================================================================================
graph=tf.Graph()
new_res = 128

with graph.as_default():
    model_in = tf.placeholder(shape=[None, 64, 64, 64, 1], dtype=tf.float32, name = "real_model_in") #Real 3D voxel objects
    normal_in = tf.placeholder(shape=[None, 512, 512, 3], dtype=tf.float32, name = "real_normal_in") #2D render of voxel objects
    image_in = tf.placeholder(shape=[None, 512, 512, 3], dtype=tf.float32, name = "real_image_in") #2D render of voxel objects
    texture_in = tf.placeholder(shape=[None, 199], dtype=tf.float32, name = "real_texture_in") #2D render of voxel objects
    param_in = tf.placeholder(shape=[None, 3], dtype=tf.float32, name = "view_name")
    is_training = tf.placeholder(tf.bool, [], name="is_training")
    patch_size_in = tf.placeholder(tf.int32, [], name="patch_size")

    #Rotate and resample 3D input
    rotated_models = tf_rotation_resampling(model_in, param_in, new_size = new_res, shapenet_viewer=cfg['shapenet_viewer'])
    rotated_models = tf_transform_voxel_to_match_image(rotated_models) #Transform voxel array to match image array (ijk -> xyz)

    #Create 3D representation of the texture of the vector input
    texture_decoded = decoder_texture (z_in=texture_in)
    #Rotate and resample 3D input
    texture_rotated = tf_rotation_resampling(texture_decoded, param_in, new_size = new_res, shapenet_viewer=cfg['shapenet_viewer'])
    texture_rotated = tf_transform_voxel_to_match_image(texture_rotated)

    #Crop voxel gris and images for patch training
    crop_vox, crop_tex, crop_img, crop_normal = tf_random_crop_voxel_texture_image_normal(rotated_models, texture_rotated, image_in, normal_in, patch_size_in)

    #Concateante voxel geometry and texture for rendering
    model_texture_concat = tf.concat([crop_vox, crop_tex], 4)
    images_pred, normal_pred = RenderNet(model_texture_concat, prob=cfg['keep_prob'])

    #Comput MSE loss on the rendered image and rendered normal
    recon_loss = tf.reduce_mean(tf.losses.mean_squared_error(crop_img, images_pred)) + \
                 tf.reduce_mean(tf.losses.mean_squared_error(crop_normal, normal_pred))

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(cfg['e_eta'], global_step, cfg['decay_steps'], 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5).minimize(recon_loss, global_step=global_step)

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
                    if epoch < 4:
                        batch_patch_size = new_res // 4
                    else:
                        batch_patch_size = new_res // 2
                    start_time=time.time()

                    train_loader = data_loader_image_texture_normal_face(cfg, img_path=IMAGE_PATH,
                                               model_path=MODEL_PATH, normal_path = NORMAL_PATH,
                                               texture_path=TEXTURE_PATH,
                                               validation_mode=False,
                                               img_res=512)
                    counter=0
                    print ("Loading data")
                    for real_images, real_normals, real_models, real_texture, real_params, real_names in train_loader:# Loop across chunks, READER object will grab the next chunk after every time a chunk is completed
                        print ("Data chunk loaded")
                        chunk_start = time.time()
                        bceLoss_chunk = 0.0
                        # Check input
                        real_images /=  255.0
                        real_normals /= 255.0
                        num_batches = len(real_images) // cfg['batch_size']
                        print("Start stepping" + str(time.time() - chunk_start))
                        for idx in range(num_batches):
                            feed_train = {model_in: batch_models,
                                    image_in: batch_images,
                                    normal_in: batch_normals,
                                    texture_in: batch_texture,
                                    param_in:batch_params,
                                    patch_size_in: batch_patch_size,
                                    is_training: True}

                            step_start = time.time()
                            batch_models = real_models[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            batch_images = real_images[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            batch_normals = real_normals[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            batch_texture = real_texture[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            batch_names  = real_names[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            batch_params = real_params[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            train, summary, step = sess.run([optimizer, merged_summary_op, global_step], feed_dict = feed_train)
                            train_writer.add_summary(summary, global_step=step)
                            print("Time/step" + str(time.time() - step_start))

                            if step % 600 == 0:
                                print( "Epoch {2} {0} BCE Loss: {1}".format(step, bceLoss_chunk, epoch))
                                rendered_samples, rendered_normals, out_cropped_im, out_cropped_normal = sess.run([images_pred, normal_pred, crop_img, crop_normal],
                                                                                                                    feed_dict = feed_train)
                                out_cropped_im = np.clip(255 * out_cropped_im, 0, 255).astype(np.uint8)
                                out_cropped_normal = np.clip(255 * out_cropped_normal, 0, 255).astype(np.uint8)
                                rendered_samples = np.clip(255 * rendered_samples, 0, 255).astype(np.uint8)
                                rendered_normals = np.clip(255 * rendered_normals, 0, 255).astype(np.uint8)
                                for i in range(1):
                                    idx=random.randint(0, cfg['batch_size'] - 1)
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "{0}_train_target_{1}_patch.png".format(batch_names[idx], step)), out_cropped_im[idx])
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "{0}_train_target_normal_{1}_patch.png".format(batch_names[idx], step)), out_cropped_normal[idx])
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "{0}_train_{1}_patch.png".format(batch_names[idx], step)), rendered_samples[idx])
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "{0}_train_{1}_patch_normal.png".format(batch_names[idx],step)), rendered_normals[idx])
                        print("Time/chunk " + str(time.time() - chunk_start))
                    # # #
                    save_path = sess_saver.save(sess, MODEL_SAVE)
                    # # if epoch % 200 == 0:
                    valid_loader = data_loader_image_texture_normal_face(cfg, img_path=IMAGE_PATH_VALID, normal_path=NORMAL_PATH,
                                               model_path=MODEL_PATH, texture_path=TEXTURE_PATH,
                                               validation_mode=True,
                                               img_res=512)

                    counter = 0
                    L1_valid = 0.
                    for real_images, real_normals, real_models, real_texture,  real_params, real_names in valid_loader:
                        real_images /= 255.0
                        real_normals /= 255.0
                        num_batches = len(real_images) // cfg['batch_size']
                        for idx in range(num_batches):
                            valid_batch_models  = real_models[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            valid_batch_images  = real_images[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            valid_batch_normals  = real_normals[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            valid_batch_texture = real_texture[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            valid_batch_names   = real_names[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            valid_batch_params  = real_params[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                            feed_valid = {model_in: valid_batch_models,
                                          image_in: valid_batch_images,
                                          normal_in: valid_batch_normals,
                                          texture_in: valid_batch_texture,
                                          param_in: valid_batch_params,
                                          patch_size_in: new_res,
                                          is_training:False}
                            rendered_samples, rendered_normals    = sess.run([images_pred, normal_pred],
                                                                             feed_dict = feed_valid)

                            rendered_samples = np.clip(255 * rendered_samples, 0, 255).astype(np.uint8)
                            rendered_normals = np.clip(255 * rendered_normals, 0, 255).astype(np.uint8)

                            if counter % 600 == 0:
                                for i in range(1):
                                    index = random.randint(0, cfg['batch_size'] - 1)
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "VALID_{0}_target_{1}.png".format(valid_batch_names[index], epoch)), np.squeeze(valid_batch_images[index]))
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "VALID_{0}_target_normal_{1}.png".format(valid_batch_names[index], epoch)), np.squeeze(valid_batch_normals[index]))
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "VALID_{0}_pred_{1}.png".format(valid_batch_names[index], epoch)), np.squeeze(rendered_samples[index]))
                                    scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "VALID_{0}_pred_normal_{1}.png".format(valid_batch_names[index], epoch)), np.squeeze(rendered_normals[index]))
                            validation_accuracy = np.mean(np.absolute(valid_batch_images - rendered_samples))
                            L1_valid += validation_accuracy
                            counter +=1

                    # Across test set (multi mini-batches) (after 1 epoch)
                    L1_valid = L1_valid / counter
                    L1_ALL.append(L1_valid)
                    np.savez(os.path.join(SAMPLE_SAVE, "L1 All.txt"), L1_ALL)
                    print("Time elapsed: " + str(time.time() - start_time))
                sv.request_stop()