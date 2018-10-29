import numpy as np
import tensorflow as tf
import scipy.misc
import math
import os
import scipy.ndimage
import glob
import scipy.io

def load_mat(mat_dir):
    return scipy.io.loadmat(mat_dir)

def get_weight(weight_name, weight_dict):
    """
    Helper function to retrieve weight using layer name from a dictionary
    :param weight_name: layer name
    :param weight_dict: dictionary of weight values
    :return A numpy array of weight
    """
    if weight_dict is None:
        print("Can't find weight")
        return None
    else:
        return weight_dict.get(weight_name)  # returns None if name is not found in dictionary

def load_weights(weight_dir):
    """
    Helper function to load weight from a pretrained model saved as a .npz file
    :param weight_dir Path to the weight fle
    :return A dictionarty of layer name and values
    """
    weight_path_all = glob.glob(os.path.join(weight_dir, "*.txt.npz"))
    pretrained_weight_dict = {}
    print(len(weight_path_all))
    for path in weight_path_all:
        with np.load(path) as data:
            layer_name = os.path.basename(path).split('.')[0]
            pretrained_weight_dict[layer_name] = data['arr_0']
    return pretrained_weight_dict

def tf_transform_voxel_to_match_image(tensor_voxel):
    """
    Function to swap the
    :param tensor_voxel: Tensor to swap axis. Shape [batch_size, height, width, depth, channel]
    :return a tensor with X axis aligned with image rows, and Y-axis aligned with image columns
    """
    tensor = tf.transpose(tensor_voxel, [0, 2, 1, 3, 4])
    tensor = tensor[:, ::-1, :, :, :]
    return tensor

def np_transform_tensor_to_image (tensor):
    """
    Function to swap
    :param tensor_voxel: Tensor to swap axis. Shape [batch_size, height, width, depth, channel]
    :return a tensor with X axis aligned with image rows, and Y-axis aligned with image columns
    """
    t = np.transpose(tensor, [0, 2, 1, 3])
    return t

def name_to_param(model_names):
    """
    Helper function to extract view parameters from file names (Azimuth, elevation, scale)
    Provided that the name follows convention: model_normalized_{model_index}_clean_p{azimuth}_t{elevation}_r{scale}.png
    :param model_name: name of the image files
    :return numpy array of shape 3 for azimuth, elevation and scale
    """
    params = np.zeros([len(model_names), 2])
    for i, name in enumerate(model_names):
        content = name.split('_')
        azimuth = float(content[4]) * 15.0 * math.pi / 180.0
        elevation = float(content[5]) * 15.0 * math.pi / 180.0
        params[i] = np.array([azimuth, elevation])

    return params


def tf_random_crop_voxel_image(voxels, images, patch_size):
    """
    Function to randomly and spatially crop voxel grids and corresponding images for training wih patches
    :param voxels: voxels to crop. Shape [batch_size, height, width, depth, channel]
    :param images: images to crop. Shape [batch_size, height, width, channel]
    :param patch_size: size to crop voxel patches. The corresponding size to crop the images will be automatically calculated.
    :return a tuple of cropped (voxel grids, images)
    """
    with tf.variable_scope("Random_Cropping"):
        batch_size = tf.shape(voxels)[0]
        voxel_dim = tf.shape(voxels)[1]
        image_dim = tf.shape(images)[1]
        image_voxel_factor = tf.to_int32(image_dim / voxel_dim)
        if patch_size == voxel_dim:
            return tf.identity(voxels, "voxel_patch" ), tf.identity(images, "image_patch")
        voxel_start_point = tf.random_uniform([2], minval=0, maxval=voxel_dim - patch_size + 1, dtype=tf.int32, seed=None, name=None)
        image_start_point = voxel_start_point * image_voxel_factor

        voxel_patch = voxels[:, voxel_start_point[0] : voxel_start_point[0] + patch_size,
                                voxel_start_point[1] : voxel_start_point[1] + patch_size, : , :]
        image_patch = images[:, image_start_point[0] : image_start_point[0] + image_voxel_factor * patch_size,
                                image_start_point[1] : image_start_point[1] + image_voxel_factor * patch_size, :]

        return tf.identity(voxel_patch, "voxel_patch" ), tf.identity(image_patch, "image_patch")

def tf_random_crop_voxel_texture_image(voxels, texture, images, patch_size):
    """
    Function to randomly and spatially crop voxel grids, texture grids and corresponding images for training wih patches
    :param voxels: voxels to crop. Shape [batch_size, height, width, depth, channel]
    :param texture: 3D texture grids to crop. Have the same shape with voxel grids
    :param images: images to crop. Shape [batch_size, height, width, channel]
    :param patch_size: size to crop voxel patches. The corresponding size to crop the images will be automatically calculated.
    :return a tuple of cropped (voxel grids, texture_grids, images)
    """
    with tf.variable_scope("Random_Cropping"):
        batch_size = tf.shape(voxels)[0]
        voxel_dim = tf.shape(voxels)[1]
        image_dim = tf.shape(images)[1]
        image_voxel_factor = tf.to_int32(image_dim / voxel_dim)
        if patch_size == voxel_dim:
            return tf.identity(voxels, "voxel_patch" ), tf.identity(texture, "texture_patch" ), tf.identity(images, "image_patch")

        voxel_start_point = tf.random_uniform([2], minval=0, maxval=voxel_dim - patch_size + 1, dtype=tf.int32, seed=None, name=None)
        image_start_point = voxel_start_point * image_voxel_factor

        voxel_patch = voxels[:, voxel_start_point[0] : voxel_start_point[0] + patch_size,
                                voxel_start_point[1] : voxel_start_point[1] + patch_size, : , :]
        texture_patch = texture[:, voxel_start_point[0] : voxel_start_point[0] + patch_size,
                                voxel_start_point[1] : voxel_start_point[1] + patch_size, : , :]
        image_patch = images[:, image_start_point[0] : image_start_point[0] + image_voxel_factor * patch_size,
                                image_start_point[1] : image_start_point[1] + image_voxel_factor * patch_size, :]

        return tf.identity(voxel_patch, "voxel_patch"), tf.identity(texture_patch, "texture_patch"), tf.identity(image_patch,"image_patch")

def tf_random_crop_voxel_texture_image_normal(voxels, texture, images, normals, patch_size):
    """
    Function to randomly and spatially crop voxel grids, texture grids and corresponding images, normal maps for training wih patches
    :param voxels: voxels to crop. Shape [batch_size, height, width, depth, channel]
    :param texture: 3D texture grids to crop. Have the same shape with voxel grids
    :param images: images to crop. Shape [batch_size, height, width, channel]
    :param normals: normal maps to crop. Have the same shape with images
    :param patch_size: size to crop voxel patches. The corresponding size to crop the images and normal maps will be automatically calculated.
    :return a tuple of cropped (voxel grids, texture_grids, images, normals)
    """
    with tf.variable_scope("Random_Cropping"):
        batch_size = tf.shape(voxels)[0]
        voxel_dim = tf.shape(voxels)[1]
        image_dim = tf.shape(images)[1]
        image_voxel_factor = tf.to_int32(image_dim / voxel_dim)
        if patch_size == voxel_dim:
            return tf.identity(voxels, "voxel_patch" ), tf.identity(texture, "texture_patch" ), tf.identity(images, "image_patch")

        voxel_start_point = tf.random_uniform([2], minval=0, maxval=voxel_dim - patch_size + 1, dtype=tf.int32, seed=None, name=None)
        image_start_point = voxel_start_point * image_voxel_factor

        voxel_patch = voxels[:, voxel_start_point[0] : voxel_start_point[0] + patch_size,
                                voxel_start_point[1] : voxel_start_point[1] + patch_size, : , :]
        texture_patch = texture[:, voxel_start_point[0] : voxel_start_point[0] + patch_size,
                                voxel_start_point[1] : voxel_start_point[1] + patch_size, : , :]
        image_patch = images[:, image_start_point[0] : image_start_point[0] + image_voxel_factor * patch_size,
                                image_start_point[1] : image_start_point[1] + image_voxel_factor * patch_size, :]
        normal_patch =normals[:, image_start_point[0] : image_start_point[0] + image_voxel_factor * patch_size,
                                image_start_point[1] : image_start_point[1] + image_voxel_factor * patch_size, :]

        return tf.identity(voxel_patch, "voxel_patch"), tf.identity(texture_patch, "texture_patch"), tf.identity(image_patch,"image_patch"), tf.identity(normal_patch, "normal_patch")

def tf_random_crop_image(images, patch_size, crop_ratio):
    """
    Function to randomly and spatially crop images for training wih patches
    :param images: images to crop. Shape [batch_size, height, width, channel]
    :param patch_size: size of cropped images
    :param crop_ratio: the portion of the area of the original image where the random patch is cropped from
    :return a tensor of cropped images
    """
    with tf.variable_scope("Random_Image_Cropping"):
        image_dim = tf.shape(images)[1]
        if patch_size == image_dim:
            return tf.identity(images, "image_patch")


        image_start_point = tf.random_uniform([2], minval=image_dim // crop_ratio, maxval=image_dim - image_dim // crop_ratio - patch_size + 1, dtype=tf.int32, seed=None, name=None)


        image_patch = images[:, image_start_point[0] : image_start_point[0] + patch_size,
                                image_start_point[1] : image_start_point[1] + patch_size, :]

        return tf.identity(image_patch, "image_patch")

def tf_center_crop_voxel_image(voxels, images, patch_size):
    """
    Function to spatially crop voxel grids and images around the centre for training wih patches
    :param voxels: voxels to crop. Shape [batch_size, height, width, depth, channel]
    :param images: images to crop. Shape [batch_size, height, width, channel]
    :param patch_size: size to crop voxel patches. The corresponding size to crop the images and normal maps will be automatically calculated.
    :return a tupple of cropped (voxel grid, images
    """
    with tf.variable_scope("Center_Cropping"):
        batch_size = tf.shape(voxels)[0]
        voxel_dim = tf.shape(voxels)[1]
        image_dim = tf.shape(images)[1]
        image_voxel_factor = tf.to_int32(image_dim / voxel_dim)


        center = tf.shape(voxels)[1:3] // 2

        voxel_start_point = center - patch_size//2

        image_start_point = voxel_start_point * image_voxel_factor

        voxel_patch = voxels[:, voxel_start_point[0] : voxel_start_point[0] + patch_size,
                                voxel_start_point[1] : voxel_start_point[1] + patch_size, : , :]
        image_patch = images[:, image_start_point[0] : image_start_point[0] + image_voxel_factor * patch_size,
                                image_start_point[1] : image_start_point[1] + image_voxel_factor * patch_size, :]

        return tf.identity(voxel_patch, "voxel_patch" ), tf.identity(image_patch, "image_patch")

def tf_center_crop_voxel(voxels, patch_size):
    """
    Function to crop voxels from the center for training wih patches
    :param voxels: voxels to crop. Shape [batch_size, height, width, depth, channel]
    :param patch_size: size of cropped voxels
    :return a tensor of cropped voxels
    """
    with tf.variable_scope("Center_Cropping"):
        center = tf.shape(voxels)[1:3] // 2

        voxel_start_point = center - patch_size//2

        voxel_patch = voxels[:, voxel_start_point[0] : voxel_start_point[0] + patch_size,
                                voxel_start_point[1] : voxel_start_point[1] + patch_size, : , :]

        return tf.identity(voxel_patch, "voxel_patch" )

def tf_center_crop_image(images, patch_size):
    """
    Function to crop images from the center for training wih patches
    :param images: images to crop. Shape [batch_size, height, width, channel]
    :param patch_size: size of cropped images
    :return a tensor of cropped images
    """
    with tf.variable_scope("Center_Image_Cropping"):

        center = tf.shape(images)[1:3] // 2
        image_start_point = center - patch_size//2

        image_patch = images[:, image_start_point[0] : image_start_point[0] + patch_size,
                                image_start_point[1] : image_start_point[1] + patch_size, :]

        return tf.identity(image_patch, "image_patch")

def np_random_crop_voxel_image(voxels, images, patch_size, crop_ratio):
    """
    Function to randomly and spatially crop voxel grids and corresponding images for training wih patches
    :param voxels: voxels to crop. Shape [batch_size, height, width, depth, channel]
    :param images: images to crop. Shape [batch_size, height, width, channel]
    :param patch_size: size to crop voxel patches. The corresponding size to crop the images will be automatically calculated.
    :param crop_ratio: the portion of the area of the original voxel grid where the random patch is cropped from
    :return a tuple of cropped (voxel grids, images)
    """
    batch_size = voxels.shape[0]
    voxel_dim = voxels.shape[1]
    image_dim = images.shape[1]
    image_voxel_factor = int(image_dim / voxel_dim)

    voxel_start_point = np.random.randint(voxel_dim//crop_ratio, voxel_dim - voxel_dim // crop_ratio- patch_size + 1, size = batch_size)
    image_start_point = voxel_start_point * image_voxel_factor

    voxel_patch = voxels[:, voxel_start_point[0]: voxel_start_point[0] + patch_size,
                  voxel_start_point[1]: voxel_start_point[1] + patch_size, :, :]
    image_patch = images[:, image_start_point[0]: image_start_point[0] + image_voxel_factor * patch_size,
                  image_start_point[1]: image_start_point[1] + image_voxel_factor * patch_size, :]


    return voxel_patch, image_patch

def np_center_crop_voxel_image(voxels, images, patch_size):
    """
    Function to spatially crop voxel grids and corresponding images around the cente for training wih patches
    :param voxels: voxels to crop. Shape [batch_size, height, width, depth, channel]
    :param images: images to crop. Shape [batch_size, height, width, channel]
    :param patch_size: size to crop voxel patches. The corresponding size to crop the images will be automatically calculated.
    :return a tuple of cropped (voxel grids, images)
    """
    with tf.variable_scope("Center_Cropping"):
        batch_size = voxels.shape[0]
        voxel_dim = voxels.shape[1] * 2 #Since in tensorflow function, the input voxel is rotated and embedded in 128^3 array before getting cropped
        image_dim = images.shape[1]
        image_voxel_factor = int(image_dim / voxel_dim)
        print(image_voxel_factor)
        center = np.array([voxels.shape[1] // 2, voxels.shape[2] // 2, voxels.shape[3] // 2])

        voxel_start_point = center - patch_size//2

        image_start_point = voxel_start_point * image_voxel_factor

        voxel_patch = voxels[:, voxel_start_point[0] : voxel_start_point[0] + patch_size,
                                voxel_start_point[1] : voxel_start_point[1] + patch_size, : , :]
        image_patch = images[:, image_start_point[0] : image_start_point[0] + image_voxel_factor * patch_size,
                                image_start_point[1] : image_start_point[1] + image_voxel_factor * patch_size, :]

        print(voxel_patch.shape)
        print(image_patch.shape)

        return voxel_patch, image_patch

def np_center_image_crop(images, patch_size):
    """
    Function to crop images from the center for training wih patches
    :param images: images to crop. Shape [batch_size, height, width, channel]
    :param patch_size: size of cropped images
    :return a tensor of cropped images
    """
    with tf.variable_scope("Center_Cropping"):
        center = np.array([images.shape[1] // 2, images.shape[2] // 2, images.shape[3] // 2])
        start_point = center - patch_size//2

        image_patch = images[:, start_point[0] : start_point[0] + patch_size,
                                start_point[1] : start_point[1] + patch_size, :]

        return image_patch

#=====================================================================================================================================================

def center_pad_binvox_cube(binvox):
    """
    Pads the binvox with zeros to fit into a cube.
    :param binvox: binvox to pad. Shape [height, width, depth, channel]
    :return Padded cubic numpy array
    """

    cube_length = max(binvox.shape)
    pad_before = [(cube_length - e) // 2 for e in binvox.shape]
    pad_after = [(cube_length - a - b) for (a, b) in zip(binvox.shape, pad_before)]
    crop = np.pad(binvox, list(zip(pad_before, pad_after)), 'constant')
    return crop




if __name__ == "__main__":
    #Testing random crop for voxels and corresponding images
    # path = r"D:\Data_sets\BaselFaceModel\BaselFaceModel\Generated\ply0_3.binvox"
    # with open(path, "rb") as fp:
    #     binvox = binvox_rw.read_as_3d_array(fp).data.astype(np.float32).reshape([64, 64, 64])
    #     binvox[32:, :, :] = 0
    #     utils.save_binvox(binvox, r"D:\Data_sets\BaselFaceModel\BaselFaceModel\Generated\ply0_3_half.binvox")
    # # # binvox = np.concatenate([binvox, binvox, binvox, binvox, binvox, binvox], axis = 0)
    # #
    # #
    # images = scipy.misc.imread(r"D:\Data_sets\ShapeNetCore_v2\ShapeNet64_Chair\SS\ShapeNet64_Chair_images_Contours\model_normalized_0_clean_23_3.png").astype(float).reshape((1, 512, 512, 3))
    # # images = np.concatenate([images, images, images, images, images, images], axis = 0)
    # #
    # crop_img
    # #
    # for i in range (crop_img.shape[0]):
    #     # utils.save_binvox(crop_vox[i].reshape((16, 16, 64)), "D:/BINVOX_cropped_{0}.binvox".format(i))
    #     scipy.misc.imsave("D:/cropped_{0}.png".format(i), crop_img[i])
    #

    # binvox = np.zeros((64, 64, 64))
    # binvox[32:, :32, 32:] = 1
    # binvox[:50, :20, :20] = 1
    # utils.save_binvox(binvox > 0, "D:/BINVOX_cropped.binvox")
    # import glob
    # images = np.zeros([4, 512, 512, 3])
    # path = glob.glob(os.path.join(os.path.join("D:\Data_sets\ShapeNetCore_v2\ShapeNet64_Chair\debug_pixelCNN_Normals_512", "*.png")))
    # print(len(path))
    # for i in range(len(path)):
    #     images[i] = scipy.misc.imread(path[i]).astype(np.float32)
    # crop_imgs = np_center_image_crop(images, 128)
    # for i in range(len(path)):
    #     name = os.path.basename(path[i]).split(".png")[0]
    #     scipy.misc.imsave("D:/crop_center_{0}.png".format(name), crop_imgs[i])
    # dict = scipy.io.loadmat(r"D:\Data_sets\BaselFaceModel\BaselFaceModel\Generated\alpha0.mat")
    # param = dict['alpha']
    # print(dict.keys())
    # print(param.shape)
    #===================================================================================
    #Test Phong shading
    #===================================================================================
    #===================================================================================
    #Test Phong shading
    #===================================================================================
    image_in = tf.placeholder(shape=[None, 512, 512, 3], dtype=tf.float32, name = "real_image_in") #2D render of voxel objects
    tf_ambient_in = tf.constant(0., dtype = tf.float32)
    tf_k_diffuse = tf.constant(0.7, dtype=tf.float32)
    # light_pos = tf.constant(np.array([[3., 2.5, 3], [1, 0, 0], [0, 1, 0], [-1, 0, 0]]), dtype=tf.float32)

    light_col = tf.constant(np.array([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0, 0, 1]]), dtype=tf.float32)
    batch_light_azimuth = tf.constant(np.tile(np.array([230 * math.pi / 180.0]), (4, 1)), dtype=tf.float32)
    light_pos = tf_generate_light_pos(batch_light_azimuth, (90-80) * math.pi / 180)
    phong = tf_phong_composite(image_in, light_dir=light_pos, ambient_in=tf_ambient_in, k_diffuse=tf_k_diffuse, k_intensity=light_col, with_mask=True)
    # img_np = np.expand_dims(scipy.misc.imread(
    #     r"D:\Projects\symmetryvae\Results\180514_FACE_NORMAL_gridParam_gradParamZTex_rdnInitTex_debugging\3_100_p299.0_t_75.0_los_0.04667_li_293.0_normal.jpg"), axis=0)/255.
    img_np = np.expand_dims(scipy.misc.imread(
        r"D:\Projects\symmetryvae\Results\180515_new_FACE_NORMAL_gridParam_gradParamZTex_rdnInitTex_Compare_DG_ICN_v2_debugging\4_300_p315.0_t_111.0_los_0.05654_li_398.0_normal.jpg"), axis=0)/255.

    img_np = np.tile(img_np[:, :, :, :3], (4, 1, 1, 1))

    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            # image, mask_out = sess.run([phong, mask], feed_dict={image_in: img_np})
            image = sess.run(phong, feed_dict={image_in: img_np})
            print(np.amax(image))
            image = np.clip(255 * image, 0, 255).astype(np.uint8)
            # mask_out = np.clip(255 * mask_out, 0, 255).astype(np.uint8)
            scipy.misc.imsave(os.path.join("D:/phong_tf_GT_v3.png"), image[0])
            # scipy.misc.imsave(os.path.join("D:/View for Chuan/phong_mask_tf.png"), np.squeeze(mask_out[0]))

    # with np.load(r"D:\Projects\SymmetryVAE\Results\00_CSPC_60\180328_3DAE_v4\z_all.txt.npz") as data:
    #     z_all = data['arr_0']
    #
    # z_mean = np.mean(z_all, axis = 0)
    # print(z_mean)
    # print(z_mean.shape)
    # z_cov = np.cov(z_all.T)
    # print(z_cov.shape)
    #
    # print(z_all.shape)
    #
    # np.savez(r"D:\Projects\SymmetryVAE\Results\00_CSPC_60\180328_3DAE_v4\z_all_mean.txt", z_mean)
    # np.savez(r"D:\Projects\SymmetryVAE\Results\00_CSPC_60\180328_3DAE_v4\z_all_cov.txt", z_cov)