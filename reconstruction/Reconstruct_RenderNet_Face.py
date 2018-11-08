import numpy as np
import os
import time
import tensorflow as tf
import sys
import math
import scipy.ndimage
import shutil
import json
import glob
# code_path    = os.environ['SYMMETRYVAE_CODE']
# data_path    = os.environ['SYMMETRYVAE_DATA']
# results_path = os.environ['SYMMETRYVAE_RESULTS']
#
# # Add additional import paths.
# sys.path.append(code_path)
# sys.path.append(os.path.join(code_path, 'model_train'))
# sys.path.append(os.path.join(code_path, 'Visualisation VTK'))
# sys.path.append(os.path.join(code_path, 'Symmetry'))

from tools.model_util import tf_transform_voxel_to_match_image, load_weights
from tools.layer_util import conv3d_transpose, conv3d, prelu, conv2d, conv2d_transpose, fully_connected, res_block_3d, res_block_2d
from tools import binvox_rw
from tools import Phong_shading
from tools.resampling_voxel_grid import tf_rotation_resampling


with open(sys.argv[1], 'r') as fh:
    cfg = json.load(fh)


code_path = os.environ['SYMMETRYVAE_CODE']
data_path = os.environ['SYMMETRYVAE_DATA']
results_path = os.environ['SYMMETRYVAE_RESULTS']

IMAGE_PATH = cfg['image_path']
MODEL_PATH = cfg['model_path']
SAMPLE_SAVE = cfg['sample_save']
MODEL_SAVE = os.path.join(SAMPLE_SAVE, cfg['trained_model_name'])
WEIGHT_DIR = os.path.join(data_path, cfg['weight_dir'])
WEIGHT_DIR_DECODER = os.path.join(data_path, cfg['weight_dir_decoder'])
WEIGHT_DIR_TEXTURE = os.path.join(data_path, cfg['weight_dir_texture'])
LOGDIR = os.path.join(SAMPLE_SAVE, "log")

os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(cfg['gpu'])



def decoder_3d_pretrained_small(z_in, weight_dict, is_training=False, prob=1.0, trainable=False):
    """
    Create the Generator Network
    :param:
        z: Latent vector
        reuse: True for reusing scope name
    :return:
        gen4: 3D voxel models
    """
    batch_size = tf.shape(z_in)[0]
    with tf.variable_scope('g_zP'):
        zP = (fully_connected(z_in, 4 * 4 * 4 * 256, scope='g_gc1',
                              trainable=trainable,
                              weight_initializer=weight_dict["g_zP_g_gc1_weights"],
                              bias_initializer=weight_dict["g_zP_g_gc1_biases"]))
        zCon = tf.reshape(zP, [batch_size, 4, 4, 4, 256])

    with tf.variable_scope('g_conv1'):
        gen1 = tf.nn.elu(conv3d_transpose(zCon, 128, kernel_size=[4, 4, 4], stride=[2, 2, 2], pad="SAME", scope='g_conv1',
                                          trainable=trainable,
                                          weight_initializer=weight_dict["g_conv1_g_conv1_weights"],
                                          bias_initializer=weight_dict["g_conv1_g_conv1_biases"]))
        # gen1 = tf.nn.dropout(gen1, keep_prob(is_training, prob))
    with tf.variable_scope('g_conv2'):
        gen2 = tf.nn.elu(conv3d_transpose(gen1, 64, kernel_size=[4, 4, 4], stride=[2, 2, 2], pad="SAME", scope='g_conv2',
                                          trainable=trainable,
                                          weight_initializer=weight_dict["g_conv2_g_conv2_weights"],
                                          bias_initializer=weight_dict["g_conv2_g_conv2_biases"]))
        # gen2 = tf.nn.dropout(gen2, keep_prob(is_training, prob))
    with tf.variable_scope('g_conv3'):
        gen3 = tf.nn.elu(conv3d_transpose(gen2, 32, kernel_size=[4, 4, 4], stride=[2, 2, 2], pad="SAME", scope='g_conv3',
                                          trainable=trainable,
                                          weight_initializer=weight_dict["g_conv3_g_conv3_weights"],
                                          bias_initializer=weight_dict["g_conv3_g_conv3_biases"]))
        # gen3 = tf.nn.dropout(gen3, keep_prob(is_training, prob))
    with tf.variable_scope('g_conv4'):
        gen4 = tf.nn.elu(conv3d_transpose(gen3, 16, kernel_size=[4, 4, 4], stride=[2, 2, 2], pad="SAME", scope='g_conv4',
                                          trainable=trainable,
                                          weight_initializer=weight_dict["g_conv4_g_conv4_weights"],
                                          bias_initializer=weight_dict["g_conv4_g_conv4_biases"]))
        # gen4 = tf.nn.dropout(gen4, keep_prob(is_training, prob))
    gen5 = tf.nn.sigmoid(conv3d_transpose(gen4, 1, kernel_size=[4, 4, 4], stride=[1, 1, 1], pad="SAME", scope='g_conv5',
                                          trainable=trainable,
                                          weight_initializer=weight_dict["g_conv5_weights"],
                                          bias_initializer=weight_dict["g_conv5_biases"]), name="output")
    return gen5

def encoder_texture_pretrained(z_in, weight_dict, is_training=False, prob=1.0, trainable=False):
    with tf.variable_scope("texture_encoder"):
        batch_size = tf.shape(z_in)[0]
        with tf.variable_scope('e_tex_dc1'):
            zP = prelu((fully_connected(z_in, 4 * 4 * 4 * 512, scope='g_gc1',
                              trainable=trainable,
                              weight_initializer=weight_dict["e_tex_dc1_e_tex_dc1_weights"],
                              bias_initializer=weight_dict["e_tex_dc1_e_tex_dc1_biases"])), alpha=weight_dict["e_tex_dc1_alpha"], trainable=trainable)
            z_resize = tf.reshape(zP, [batch_size, 32, 32, 32, 4])
        with tf.variable_scope('e_tex_conv0'):
            conv0 = prelu(conv3d_transpose(z_resize, 4, kernel_size=[4, 4, 4], stride=[1, 1, 1],
                                          trainable=trainable,
                                          weight_initializer=weight_dict["e_tex_conv0_e_tex_conv0_weights"],
                                          bias_initializer=weight_dict["e_tex_conv0_e_tex_conv0_biases"]), alpha =weight_dict["e_tex_conv0_alpha"], trainable=trainable)
        with tf.variable_scope('e_tex_conv1'):
            conv1 = prelu(conv3d_transpose(conv0, 8, kernel_size=[4, 4, 4], stride=[2, 2, 2],
                                           trainable=trainable,
                                           weight_initializer=weight_dict["e_tex_conv1_e_tex_conv1_weights"],
                                           bias_initializer=weight_dict["e_tex_conv1_e_tex_conv1_biases"]),
                                           alpha=weight_dict["e_tex_conv1_alpha"], trainable=trainable)
        with tf.variable_scope('e_tex_conv2'):
            conv2 = prelu(conv3d(conv1, 4, kernel_size=[4, 4, 4], stride=[1, 1, 1],
                                  trainable=trainable,
                                  weight_initializer=weight_dict["e_tex_conv2_e_tex_conv2_weights"],
                                  bias_initializer=weight_dict["e_tex_conv2_e_tex_conv2_biases"]),
                                  alpha=weight_dict["e_tex_conv2_alpha"], trainable=trainable)
        return conv2

def encoder_MLP_pretrained(models_in, weight_dict, prob = 1.0, n_blocks = 5, trainable = False):
    batch_size = tf.shape(models_in)[0]
    with tf.variable_scope("encoder"):
        with tf.variable_scope('e_conv1'):
            enc1 = conv3d(models_in, 8, kernel_size=[5, 5, 5], stride=[2, 2, 2], pad="SAME", scope='e_conv1',
                                trainable=trainable,
                                weight_initializer=weight_dict["e_conv1_e_conv1_weights"],
                                bias_initializer=weight_dict["e_conv1_e_conv1_biases"])
            enc1 = prelu(enc1, alpha=weight_dict["e_conv1_alpha"], trainable=trainable)
            enc1 = tf.nn.dropout(enc1, prob)
        with tf.variable_scope('e_conv2'):
            enc2 = prelu(conv3d(enc1, 16, kernel_size=[3, 3, 3], stride=[1, 1, 2], pad="SAME", scope='e_conv2',
                                trainable=trainable,
                                weight_initializer=weight_dict["e_conv2_e_conv2_weights"],
                                bias_initializer=weight_dict["e_conv2_e_conv2_biases"]),
                                alpha=weight_dict["e_conv2_alpha"], trainable=trainable)
            enc2 = tf.nn.dropout(enc2, prob)
        with tf.variable_scope('e_conv3'):
            enc3 = prelu(conv3d(enc2, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], pad="SAME", scope='e_conv3',
                                trainable=trainable,
                                weight_initializer=weight_dict["e_conv3_e_conv3_weights"],
                                bias_initializer=weight_dict["e_conv3_e_conv3_biases"]),
                                alpha=weight_dict["e_conv3_alpha"], trainable=trainable)
            enc3 = tf.nn.dropout(enc3, prob)

        shortcut = enc3

        res1_1 = res_block_3d(enc3,    16, scope='res1_1',  weight_dict=weight_dict, trainable=trainable)
        res1_2 = res_block_3d(res1_1,  16, scope='res1_2',  weight_dict=weight_dict, trainable=trainable)
        res1_3 = res_block_3d(res1_2,  16, scope='res1_3',  weight_dict=weight_dict, trainable=trainable)
        res1_4 = res_block_3d(res1_3,  16, scope='res1_4',  weight_dict=weight_dict, trainable=trainable)
        res1_5 = res_block_3d(res1_4,  16, scope='res1_5',  weight_dict=weight_dict, trainable=trainable)
        res1_6 = res_block_3d(res1_5,  16, scope='res1_6',  weight_dict=weight_dict, trainable=trainable)
        res1_7 = res_block_3d(res1_6,  16, scope='res1_7',  weight_dict=weight_dict, trainable=trainable)
        res1_8 = res_block_3d(res1_7,  16, scope='res1_8',  weight_dict=weight_dict, trainable=trainable)
        res1_9 = res_block_3d(res1_8,  16, scope='res1_9',  weight_dict=weight_dict, trainable=trainable)
        res1_10 = res_block_3d(res1_9, 16, scope='res1_10', weight_dict=weight_dict, trainable=trainable)

        with tf.variable_scope('res1_skip'):
            enc3_skip = conv3d(res1_10, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], pad="SAME", scope="con1_3X3",
                               trainable = trainable,
                               weight_initializer=weight_dict["res1_skip_con1_3X3_weights"],
                               bias_initializer=weight_dict["res1_skip_con1_3X3_biases"])
            # enc3_skip = tf.nn.dropout(enc3_skip, keep_prob(prob, is_training))
            enc3_skip = tf.add(tf.cast(enc3_skip, tf.float32), tf.cast(shortcut, tf.float32))

        height = tf.shape(enc3_skip)[1]
        width = tf.shape(enc3_skip)[2]
        #Collapsing Z dimension
        enc3_2d = tf.reshape(enc3_skip, [batch_size, height, width, 32 * 16])

        with tf.variable_scope('e_conv4'):
            enc4 = prelu(conv2d(enc3_2d, num_outputs = 32 * 16, kernel_size=[1, 1], scope='e_conv4',
                                trainable=trainable,
                                weight_initializer=weight_dict["e_conv4_e_conv4_weights"],
                                bias_initializer=weight_dict["e_conv4_e_conv4_biases"]),
                         alpha=weight_dict["e_conv4_alpha"], trainable=trainable)
            enc4 = tf.nn.dropout(enc4, prob)

        shortcut = enc4

        res2_1 = res_block_2d(enc4,   32 * 16, scope='res2_1', weight_dict=weight_dict, trainable=trainable)
        res2_2 = res_block_2d(res2_1, 32 * 16, scope='res2_2', weight_dict=weight_dict, trainable=trainable)
        res2_3 = res_block_2d(res2_2, 32 * 16, scope='res2_3', weight_dict=weight_dict, trainable=trainable)
        res2_4 = res_block_2d(res2_3, 32 * 16, scope='res2_4', weight_dict=weight_dict, trainable=trainable)
        res2_5 = res_block_2d(res2_4, 32 * 16, scope='res2_5', weight_dict=weight_dict, trainable=trainable)
        res2_6 = res_block_2d(res2_5, 32 * 16, scope='res2_6', weight_dict=weight_dict, trainable=trainable)
        res2_7 = res_block_2d(res2_6, 32 * 16, scope='res2_7', weight_dict=weight_dict, trainable=trainable)
        res2_8 = res_block_2d(res2_7, 32 * 16, scope='res2_8', weight_dict=weight_dict, trainable=trainable)
        res2_9 = res_block_2d(res2_8, 32 * 16, scope='res2_9', weight_dict=weight_dict, trainable=trainable)
        res2_10 = res_block_2d(res2_9, 32 * 16, scope='res2_10', weight_dict=weight_dict, trainable=trainable)

        with tf.variable_scope('res2_skip'):
            enc4_skip = conv2d(res2_10, 32 * 16, kernel_size=[3, 3], scope="con1_3X3",
                               trainable=trainable,
                               weight_initializer=weight_dict["res2_skip_con1_3X3_weights"],
                               bias_initializer=weight_dict["res2_skip_con1_3X3_biases"])
            # enc4_skip = tf.nn.dropout(enc4_skip, keep_prob(prob, is_training))
            enc4_skip = tf.add(tf.cast(enc4_skip, tf.float32), tf.cast(shortcut, tf.float32))

        with tf.variable_scope('e_conv5'):
            enc5 = prelu(conv2d(enc4_skip, 32 * 8, kernel_size=[4, 4],  scope='e_conv5',
                                trainable=trainable,
                                weight_initializer=weight_dict["e_conv5_e_conv5_weights"],
                                bias_initializer=weight_dict["e_conv5_e_conv5_biases"]),
                                alpha = weight_dict["e_conv5_alpha"], trainable=trainable)
            enc5 = tf.nn.dropout(enc5, prob)

        shortcut = enc5

        res3_1 = res_block_2d(enc5,   32 * 8, scope='res3_1', weight_dict=weight_dict, trainable=trainable)
        res3_2 = res_block_2d(res3_1, 32 * 8, scope='res3_2', weight_dict=weight_dict, trainable=trainable)
        res3_3 = res_block_2d(res3_2, 32 * 8, scope='res3_3', weight_dict=weight_dict, trainable=trainable)
        res3_4 = res_block_2d(res3_3, 32 * 8, scope='res3_4', weight_dict=weight_dict, trainable=trainable)
        res3_5 = res_block_2d(res3_4, 32 * 8, scope='res3_5', weight_dict=weight_dict, trainable=trainable)

        with tf.variable_scope('res3_skip'):
            enc5_skip = conv2d(res3_5, 32 * 8, kernel_size=[3, 3], scope="con1_3X3",
                               trainable=trainable,
                               weight_initializer=weight_dict["res3_skip_con1_3X3_weights"],
                               bias_initializer=weight_dict["res3_skip_con1_3X3_biases"])

            # enc5_skip = tf.nn.dropout(enc5_skip, keep_prob(prob, is_training))
            enc5_skip = tf.add(tf.cast(enc5_skip, tf.float32), tf.cast(shortcut, tf.float32))

        with tf.variable_scope('e_conv6'):
            enc6 = prelu(conv2d(enc5_skip, 32 * 4, kernel_size=[4,4], scope='e_conv6',
                                trainable=trainable,
                                weight_initializer=weight_dict["e_conv6_e_conv6_weights"],
                                bias_initializer=weight_dict["e_conv6_e_conv6_biases"]),
                                alpha=weight_dict["e_conv6_alpha"], trainable=trainable)
            enc6 = tf.nn.dropout(enc6, prob)
        with tf.variable_scope('e_conv7'):
            enc7 = prelu(conv2d_transpose(enc6, 32 * 2, [4, 4], stride = [2, 2], scope='e_conv7',
                                          trainable=trainable,
                                          weight_initializer=weight_dict["e_conv7_e_conv7_weights"],
                                          bias_initializer=weight_dict["e_conv7_e_conv7_biases"]),
                                          alpha=weight_dict["e_conv7_alpha"], trainable=trainable)
            enc7 = tf.nn.dropout(enc7, prob)
        with tf.variable_scope('e_conv8'):
            enc8 = prelu(conv2d_transpose(enc7, 32, [4, 4], stride = [2, 2], scope='e_conv8',
                                          trainable=trainable,
                                          weight_initializer=weight_dict["e_conv8_e_conv8_weights"],
                                          bias_initializer=weight_dict["e_conv8_e_conv8_biases"]),
                                          alpha=weight_dict["e_conv8_alpha"], trainable=trainable)
            enc8 = tf.nn.dropout(enc8, prob)
        with tf.variable_scope('e_conv9'):
            enc9 = prelu(conv2d_transpose(enc8, 16, [4, 4], stride = [2, 2], scope='e_conv9',
                                            trainable=trainable,
                                            weight_initializer = weight_dict["e_conv9_e_conv9_weights"],
                                            bias_initializer = weight_dict["e_conv9_e_conv9_biases"]),
                                            alpha=weight_dict["e_conv9_alpha"], trainable=trainable)
            enc9 = tf.identity(tf.nn.dropout(enc9, prob), name = "encoder_feature_enc9")

        #output of the network for MSE-for debugging
        with tf.variable_scope('e_conv11'):
            enc11 = conv2d_transpose(enc9, 3, [4, 4], stride=[1, 1], scope='e_conv11',
                                          trainable=trainable,
                                          weight_initializer=weight_dict["e_conv11_weights"],
                                          bias_initializer=weight_dict["e_conv11_biases"])

            enc11 = tf.nn.sigmoid(enc11, name="encoder_output")


        return enc11


def extract_param_from_names(image_dir):
    content = os.path.basename(image_dir).split('_')
    if len(content) == 4:
        azimuth = float(content[1][1:]) * math.pi / 180.0
        scale = 3.3 / float(content[3][1:1 + 3])
        elevation = 90. - float(content[2][1:])  # Map from range [10;170] (Up-Z axis =0) to [80;-80] ((horizontal X axis = 0)
    else:
        azimuth = float(content[4][1:]) * math.pi / 180.0
        scale = 3.3 / float(content[6][1:1 + 3])
        elevation = 90. - float(content[5][1:])  # Map from range [10;170] (Up-Z axis =0) to [80;-80] ((horizontal X axis = 0)
    elevation = elevation * math.pi / 180.0
    param = np.expand_dims(np.array([azimuth, elevation, scale]), axis=0)
    return param

def create_param_center(phi_mid=90, phi_range = 240, theta_mid=90, theta_range=120):
    phi_min = ((phi_mid - phi_range * 0.5) % 360) * math.pi / 180.0
    phi_max = ((phi_mid + phi_range * 0.5) % 360)* math.pi / 180.0
    theta_min = (90 - (theta_mid - theta_range * 0.5)) * math.pi / 180.0
    theta_max = (90 - (theta_mid + theta_range * 0.5)) * math.pi / 180.0
    phi_mid =    phi_mid * math.pi / 180.0
    theta_mid =  (90 - theta_mid) * math.pi / 180.0

    params = np.zeros(shape=[cfg['batch_size'], 3], dtype = np.float32)
    params[0] = np.array([phi_min, theta_min, 1.0], dtype=np.float32)
    params[1] = np.array([phi_min, theta_max, 1.0], dtype=np.float32)
    params[2] = np.array([phi_mid, theta_mid, 1.0], dtype=np.float32)
    params[3] = np.array([phi_max, theta_min, 1.0], dtype=np.float32)
    params[4] = np.array([phi_max, theta_max, 1.0], dtype=np.float32)

    return params
#=======================================================================================================================
#Global parameters
#=======================================================================================================================
graph=tf.Graph()
new_res = 128
ambient_in = (0.)
k_diffuse = 1.0
light_col = np.array([[1.0, 1.0, 1.0]])
elevation_GT = (90 - 80) * math.pi / 180.0
azimuth_GT = 230 * math.pi / 180.0

#=======================================================================================================================
#Start building the graph
#=======================================================================================================================
with graph.as_default():
    weight_dict_MLP = load_weights(WEIGHT_DIR)
    weight_dict_decoder = load_weights(WEIGHT_DIR_DECODER)
    weight_dict_texture = load_weights(WEIGHT_DIR_TEXTURE)

    target_img = tf.placeholder(shape=[None, 512, 512, 3], dtype=tf.float32, name="target_img")
    param_in = tf.placeholder(shape=[cfg['batch_size'], 3], dtype=tf.float32, name="view_in")# Has to be in radian
    texture_in = tf.placeholder(shape=[cfg['batch_size'], 199], dtype=tf.float32, name="texture_in")
    vector_in = tf.placeholder(shape = [cfg['batch_size'], cfg['z_dim']], dtype=tf.float32, name="vector_in")
    light_in = tf.placeholder(shape = [cfg['batch_size'], 1], dtype=tf.float32, name="light_in") # Has to be in radian

    initial_vector = tf.get_variable (name="initial_vector",  shape = [cfg['batch_size'], cfg['z_dim']], trainable = True)
    initial_param = tf.get_variable  (name="initial_param",   shape = [cfg['batch_size'], 3], trainable=True)
    initial_texture = tf.get_variable(name="initial_texture", shape = [cfg['batch_size'], 199], trainable=True)
    initial_light = tf.get_variable(name="initial_light", shape=[cfg['batch_size'], 1], trainable=True)

    #Assigning these vairables with the current best after a fixed number of steps
    assign_op_vec = initial_vector.assign(vector_in)
    assign_op_par = initial_param.assign(param_in)
    assign_op_tex = initial_texture.assign(texture_in)
    assign_op_light = initial_light.assign(light_in)

    recon_shape = decoder_3d_pretrained_small(initial_vector, weight_dict_decoder)
    recon_texture = encoder_texture_pretrained(initial_texture, weight_dict_texture)
    light_dir_batch = Phong_shading.tf_generate_light_pos(initial_light, numpy_theta=elevation_GT)

    rotated_model = tf_rotation_resampling(recon_shape, initial_param, new_size=new_res)
    rotated_model = tf_transform_voxel_to_match_image(rotated_model)  # Transform voxel array to match image array (ijk -> xyz)

    rotated_texture = tf_rotation_resampling(recon_texture, initial_param, new_size=new_res)
    rotated_texture = tf_transform_voxel_to_match_image(rotated_texture)  # Transform voxel array to match image array (ijk -> xyz)

    model_texture_concat = tf.concat([rotated_model, rotated_texture], 4)
    img_pred, normal_pred= encoder_MLP_pretrained(models_in=model_texture_concat, weight_dict=weight_dict_MLP, prob=1.0)


    #===================================================================================================================
    # Compute Phong shading
    #===================================================================================================================

    batch_light_intensity = np.tile(light_col, (cfg['batch_size'], 1))
    tf_light_col_in = tf.constant(batch_light_intensity, tf.float32)
    tf_ambient_in = tf.constant(ambient_in, dtype=tf.float32)
    tf_k_diffuse = tf.constant(k_diffuse, dtype=tf.float32)

    # Combine albedo and shading
    shading, mask = Phong_shading.tf_phong_composite(normal_pred, light_dir_batch, tf_ambient_in, tf_k_diffuse,
                                                     tf_light_col_in, with_mask=True)
    compos_pred = tf.multiply(img_pred, shading)


    #===================================================================================================================
    # Compute loss and build optimisers
    #===================================================================================================================

    MSE = tf.reduce_mean(tf.squared_difference(img_pred, compos_pred), axis =(1, 2, 3))
    recon_loss =  cfg["MSE_weight"] * MSE

    global_step = tf.Variable(0, name='global_step', trainable=False)
    t_vars = tf.trainable_variables()

    var_list1 = [var for var in t_vars if 'initial_vector' in var.name]
    var_list2 = [var for var in t_vars if 'initial_param' in var.name]
    var_list3 = [var for var in t_vars if 'initial_texture' in var.name]
    var_list4 = [var for var in t_vars if 'initial_light' in var.name]

    opt1 = tf.train.GradientDescentOptimizer(cfg['shape_eta']) #Update shape
    opt2 = tf.train.GradientDescentOptimizer(cfg['pose_eta'])#Update pose
    opt3 = tf.train.GradientDescentOptimizer(cfg['tex_eta'])  # Update texture
    opt4 = tf.train.GradientDescentOptimizer(cfg['light_eta'])  # Update light

    grads = tf.gradients(recon_loss, var_list1 + var_list2 + var_list3 + var_list4)
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1) : len(var_list1) + len(var_list2)]
    grads3 = grads[len(var_list1) + len(var_list2) : len(var_list1) + len(var_list2) + len(var_list3)]
    grads4 = grads[len(var_list1) + len(var_list2) + len(var_list3) :]

    train_op1 = opt1.apply_gradients(zip(grads1, var_list1), global_step = global_step)
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
    train_op3 = opt3.apply_gradients(zip(grads3, var_list3))
    train_op4 = opt4.apply_gradients(zip(grads4, var_list4))
    train_op = tf.group(train_op1, train_op2, train_op3, train_op4)



    #===================================================================================================================
    # Start reconstruction
    #===================================================================================================================
    start = time.time()
    for i in range(cfg['batch_size']):
        tf.summary.scalar('Recon loss train {0}'.format(i), recon_loss[i])

    merged_summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    sess_saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOGDIR, graph=graph)

    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            sess.run(init)
            if not os.path.exists(SAMPLE_SAVE):
                os.makedirs(SAMPLE_SAVE)

            shutil.copyfile(sys.argv[1], os.path.join(SAMPLE_SAVE, 'config.json'))
            train_writer = tf.summary.FileWriter(os.path.join(SAMPLE_SAVE, 'train'), graph=graph)

            target1_path = glob.glob(os.path.join(data_path, "BaselFace/baselFace_render_test/ply{0}*".format(image_idx)))
            target1_normal_path = glob.glob(os.path.join(data_path, "BaselFace/Face_Normal_test/ply{0}*".format(image_idx)))
            target_voxel_path = glob.glob(os.path.join(data_path, "BaselFace/baselFace_binvox_all/ply{0}*".format(image_idx)))

            # ==========================================================================================================
            # CREATING SHADED TARGET
            target = scipy.misc.imread(target1_path)[:, :, :3].reshape((1, 512, 512, 3)) / 255.
            target_normal = scipy.misc.imread(target1_normal_path)[:, :, :3].reshape((1, 512, 512, 3)) / 255.
            with open(target_voxel_path, 'rb') as f:
                target_voxel = np.reshape(binvox_rw.read_as_3d_array(f).data.astype(np.float32), (1, 64, 64, 64))

            light_dir = np.array([[np.multiply(np.sin(elevation_GT), np.cos(azimuth_GT)),
                                   np.multiply(np.sin(elevation_GT), np.sin(azimuth_GT)),
                                   np.cos(elevation_GT)]])

            target_shading = Phong_shading.np_phong_composite(target_normal, light_dir, light_col, ambient_in, k_diffuse,
                                                with_mask=True)
            target_compos = np.multiply(target, target_shading)
            scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "shaded_target.png"),
                              np.clip(target_compos[0] * 255., 0, 255).astype(np.uint8))
            scipy.misc.imsave(os.path.join(SAMPLE_SAVE, "shading.png"),
                              np.clip(target_shading[0] * 255., 0, 255).astype(np.uint8))

            target_batch = np.tile(target_compos, (cfg['batch_size'], 1, 1, 1))
            # ==========================================================================================================

            a = extract_param_from_names(target1_path)
            best_param = np.zeros(shape = (3))
            best_vector = np.zeros(shape = (200))
            best_tex = np.zeros(shape=(199))
            best_light = None
            phi_range = 60 #Initial range, will get halved for every outerloop iteration
            theta_range = 30 #Initial range, will get halved for every outerloop iteration
            inner_step = 200
            for i in range(cfg['max_epochs']):
                best_recon_loss = 0
                if i == 0:
                    #FIRST STEP INITIALISATION
                    params_batch = create_param_center(phi_mid=270, phi_range =phi_range, theta_mid=90, theta_range=theta_range)
                    vector_batch = np.ones((cfg['batch_size'], cfg['z_dim'])) * 0.5
                    tex_batch = np.random.randn(cfg['batch_size'], 199)
                    light_batch = np.expand_dims((np.linspace(220, 320, num=5) * math.pi / 180.0), axis=0).T
                else:
                    phi_range /= 2
                    theta_range /= 2
                    params_batch = create_param_center(phi_mid=best_param[0], phi_range=phi_range, theta_mid=best_param[1], theta_range=theta_range)
                    vector_batch = np.tile(best_vector[np.newaxis, :], (cfg['batch_size'], 1))
                    tex_batch = np.tile(best_tex[np.newaxis, :], (cfg['batch_size'], 1))
                    light_batch = np.tile(best_light[np.newaxis, :], (cfg['batch_size'], 1))

                for idx in range(inner_step):
                    feed_recon = {target_img: target_batch,
                                   param_in: params_batch,
                                   vector_in: vector_batch,
                                   texture_in: tex_batch,
                                   light_in: light_batch}
                    if idx == 0:
                        print("ASSIGN")
                        _, __, ___, ____ = sess.run([assign_op_vec, assign_op_par, assign_op_tex, assign_op_light],
                                                    feed_dict=feed_recon)

                    summary, train, step, recon_loss_out = sess.run([merged_summary_op, train_op, global_step, recon_loss],
                                                                    feed_dict=feed_recon)
                    train_writer.add_summary(summary, global_step=step)
                    print(str(step) + str(recon_loss_out))

                    if step % 100 == 0:
                        vox, shading_out, normal_out, image_out, params_out, tex_out, light_out, MSE_out = \
                            sess.run([recon_shape, shading, normal_pred, compos_pred, initial_param, initial_texture,
                                      initial_light, MSE], feed_dict=feed_recon)

                        shading_out = np.clip(255. * shading_out, 0, 255).astype(np.uint8)
                        image_out = np.clip(255. * image_out, 0, 255).astype(np.uint8)
                        normal_out = np.clip(255. * normal_out, 0, 255).astype(np.uint8)

                        for recon_idx in range (cfg['batch_size']):
                            save_name = "{0}_{1}_p{2:.1f}_t_{3:.1f}_los_{4:.5f}".format(recon_idx, step, int(params_out[recon_idx][0] * 180 / math.pi),
                                                                                int(90 - params_out[recon_idx][1] * 180 / math.pi),
                                                                                recon_loss_out[recon_idx])

                            scipy.misc.toimage(image_out[i], cmin=0.0, cmax=255.0).save(os.path.join(SAMPLE_SAVE,  "{0}.jpg".format(save_name)))
                            binvox_rw.save_binvox(vox[i].reshape(64, 64, 64) > 0.1,  os.path.join(SAMPLE_SAVE, "{0}.binvox".format(save_name)))
                            np.savez(os.path.join(SAMPLE_SAVE,  "{0}_Param.txt".format(save_name), vox[i].reshape(64, 64, 64)))
                            np.savez(os.path.join(SAMPLE_SAVE,  "{0}_TEX.txt".format(save_name), tex_out[i]))

                        print("Voxels saved")
                    if idx == (inner_step - 1):
                        recon_loss_out, z_out, param_out, tex_out, light_out = sess.run(
                            [recon_loss, initial_vector, initial_param, initial_texture, initial_light],
                                                                    feed_dict=feed_recon)

                        best_vector = z_out[np.argmin(recon_loss_out)]
                        best_tex = tex_out[np.argmin(recon_loss_out)]
                        best_light = light_out[np.argmin(recon_loss_out)]  # No need to convert back to degree
                        best_param = param_out[np.argmin(recon_loss_out)] * 180. / math.pi  # Convert radian to degree
                        best_param = np.array([best_param[0], 90 - best_param[1], 1])  # convert theta from tange [90, -90] to [10,170]

                        np.savez(os.path.join(SAMPLE_SAVE, "{0}_loss_.txt".format(step)), recon_loss_out)
                        print("BEST LOSS " + str(np.argmin(recon_loss_out)))
                        print("BEST PARAM " + str(best_param))