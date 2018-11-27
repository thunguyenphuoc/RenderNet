import tensorflow as tf
import numpy as np
import math

#=======================================================================================================
#Compute Phong shading in tensorflow - Work for batches
#Images must be in [0, 1]
#=======================================================================================================
def tf_repeat(input, repeats):
    #Taken from https://github.com/tensorflow/tensorflow/issues/8246
    """
    Tensorflow implementation of numpy.repeat
    :param input: A Tensor. 1-D or higher.
    :param repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimension of input
    :returns A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(input, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(input) * repeats)
    return repeated_tensor

def tf_mask(images_in):
    """
    Create a soft binary mask when the input image has a black background
    :param images_in: input image, must be in range [0, 1]
    :return a binary mask in range [0, 1]
    """
    mask = tf.norm(images_in, axis=3, keep_dims=True)
    mask = tf.sigmoid(255. * mask - 80) #Mask has to be in [0,255] to work

    return mask

def tf_mask_white(images_in):
    """
    Create a soft binary mask when the input image has a white background
    :param images_in: input image, must be in range [0, 1]
    :return a binary mask in range [0, 1]
    """
    mask = tf.ones_like(images_in, dtype=tf.float32) * math.sqrt(3) - tf.norm((images_in), axis=3, keep_dims=True)
    mask = tf.sigmoid(255. * mask - 80) #Mask has to be in [0,255] to work

    return mask

def tf_phong_shading(images_in, light_dir, light_col, k_diffuse):
    """
    Creating Phong shaded images. Work with batches
    :param images_in. Shape [batch_size, height, hannelwidth, channel]. Must be in range [0, 1]
    :param light_dir. Shape [batch_size, 3]
    :param light_col. Shape [batch_size, 3].
    :param k_diffuse. Diffuse constant
    :return batch of Phong shaded images clipped to range [0,1]
    """

    normals_ish = images_in - 0.5

    # reshape into N-by-3 matrix, Works for batches as well
    normals_ish_vec = tf.reshape(normals_ish, [-1, 3])

    # normalise each row => unit normals in each row
    normals_vec = normals_ish_vec / tf.expand_dims(tf.norm(normals_ish_vec, axis=1), axis = 1)

    #normalise light dir
    light_dir /= tf.reshape(tf.norm(light_dir, axis=1), [-1, 1])

    # Repeat each light for all pixels in an image.
    light_dir = tf_repeat(light_dir, [tf.reduce_prod(tf.shape(images_in)[1:3]), 1])
    light_col = tf_repeat(light_col, [tf.reduce_prod(tf.shape(images_in)[1:3]), 1])

    #Doing the Dot Poduct for batches
    diffuse_vec = tf.reduce_sum(tf.multiply(normals_vec, light_dir), axis=1, keep_dims=True)

    # ... clamping negative values ...
    diffuse_vec = tf.maximum(diffuse_vec, 0.)

    # ... repeated for each colour channel ...
    diffuse_vec = tf_repeat(diffuse_vec, [1, 3])

    # ... multiplied by the light colour and diffuse reflectivity (k_diffuse) ...
    diffuse_col_vec = k_diffuse * tf.multiply(diffuse_vec, light_col)

    # ... and reshaped back into the batch shape.
    diffuse = tf.reshape(diffuse_col_vec, tf.shape(images_in))

    return tf.clip_by_value(diffuse, 0., 1.)

def tf_phong_composite(images_in, light_dir, light_col, ambient_in, k_diffuse, with_black_background=False, with_mask=True):
    """
    Creating Phong shaded images with binary mask. Work with batches
    :param images_in. Shape [batch_size, height, hannelwidth, channel]. Must be in range [0, 1]
    :param light_dir. Shape [batch_size, 3]
    Lparam light_col. Shape [batch_size, 3]
    :param ambient. Ambient constant
    :param k_diffuse. Diffuse constant
    :param with_mask. Using binary mask for the background
    :return batch of Phong shaded images clipped to range [0,1]
    """

    #Create diffuse
    diffuse = tf_phong_shading(images_in, light_dir, light_col, k_diffuse)

    if with_mask:
        # Create binary mask
        if with_black_background:
            mask = tf_mask(images_in)
        else:
            mask = tf_mask_white(images_in)
        compos = mask * (ambient_in + diffuse) + (1 - mask)
    else:
        compos = ambient_in + diffuse

    return tf.clip_by_value(compos, 0., 1.)

def tf_generate_light_pos(batch_light_azimuth, light_elevation, batch_size):
    """
    Generate position of light source using 3D spherical coordinates
    :param batch_light_azimuth
    :param light_elevation: GT light elevation
    :param batch_size
    :return tensor of batches of light position
    """
    theta_array = np.tile(np.array([[light_elevation]]), (batch_size, 1))
    batch_light_elevation = tf.constant(theta_array, dtype=tf.float32)

    x = tf.multiply(tf.sin(batch_light_elevation), tf.cos(batch_light_azimuth))
    y = tf.multiply(tf.sin(batch_light_elevation), tf.sin(batch_light_azimuth))
    z = tf.cos(batch_light_elevation)

    return tf.concat((x, y, z), axis = 1)


#=======================================================================================================
#Compute Phong shading in Numpy - Only works for one image
#Input image must be in range [0, 255]
#=======================================================================================================

def np_mask(images_in):
    """
    Create a soft binary mask when the input image has a black background
    :param images_in: input image, must be in range [0, 1]. Shape [batch_size, height, width, 3]
    :return a binary mask in range [0, 1]
    """
    mask = np.linalg.norm(images_in, axis=3, keepdims=True)
    sigmoid = lambda x: 1. / (1. + np.exp(-x))
    mask = sigmoid(255. * mask - 150)

    return mask

def np_mask_white(images_in):
    """
    Create a soft binary mask when the input image has a white background
    :param image_in: input images, must be in range [0, 1]. Shape [batch_size, height, width, 3]
    :return a binary mask in range [0, 1]
    """
    mask = np.linalg.norm(1. - images_in, axis=3, keepdims=True)
    sigmoid = lambda x: 1. / (1. + np.exp(-x))
    mask = sigmoid(255. * mask - 80)

    return mask

def np_phong_shading(img_batch, light_dir, light_col, k_diffuse):
    """
    Creating Phong shaded images. Work with batches
    :param image_batch Shape [batch_size, height, width, channel]. Must be in range [0, 1]
    :param light_dir. Must be numpy array. Shape [batch_size, 3]
    :param k_diffuse. Diffuse constant
    :param light_col. Must be numpy array. Light intensity, a constant.
    :return batch of Phong shaded images clipped to range [0,1]
    """
    # Convert normal map image to normals, and reshape into N-by-3 matrix.
    normals_ish = img_batch - 0.5
    normals_ish_vec = normals_ish.reshape([-1, 3])

    # Normalise each row => unit normals in each row
    normals_vec = normals_ish_vec / np.linalg.norm(normals_ish_vec, axis=1)[:, np.newaxis]

    # Normalise light directions.
    light_dir /= np.linalg.norm(light_dir, axis=1).reshape([-1, 1])

    # Repeat each light for all pixels in an image.
    light_dir = np.repeat(light_dir, np.prod(img_batch.shape[1:3]), 0)
    light_col = np.repeat(light_col, np.prod(img_batch.shape[1:3]), 0)

    # Diffuse shading is basically N.L ...
    diffuse_vec = np.sum(np.multiply(normals_vec, light_dir), axis=1, keepdims=True)

    # ... clamping negative values ...
    diffuse_vec = np.maximum(diffuse_vec, 0.0)

    # ... repeated for each colour channel ...
    diffuse_vec = np.repeat(diffuse_vec, 3, 1)

    # ... multiplied by the light colour and diffuse reflectivity (k_diffuse) ...
    diffuse_col_vec = k_diffuse * np.multiply(diffuse_vec, light_col)

    # ... and reshaped back into the batch shape.
    diffuse = np.reshape(np.array(diffuse_col_vec), img_batch.shape)

    return np.clip(diffuse, 0, 1)

def np_phong_composite(images_in, light_dir, light_col, ambient_in, k_diffuse, background_col="Black", with_mask=True):
    """
    Creating Phong shaded images with binary mask. Work with batches. Images must be in range [0, 1]
    :param image_in. Shape [batch_size, height, width, channel]
    :param light_dir. Must be numpy array. Shape [batch_size, 3]
    :param light_col. Must be numpy array. Shape [batch_size, 3]
    :param ambient_in. Ambient constant
    :param k_diffuse. Diffure constant
    :param background_col. Choose background color of the input images: black and white
    :param with_mask. Using binary mask for the background
    :return batch of Phong shaded images clipped to range [0,1]
    """

    #Create diffuse
    diffuse = np_phong_shading(images_in, light_dir, light_col, k_diffuse)

    if with_mask:
        # Create binary mask
        if background_col == "Black" or background_col == "black" or background_col == "BLACK":
            mask = np_mask(images_in)
        else:
            mask = np_mask_white(images_in)
        compos = mask * (ambient_in + diffuse) + (1 - mask)
    else:
        compos = ambient_in + diffuse

    return np.clip(compos, 0, 1)

def generate_random_light_pos(batch_size, elevation_low=0, elevation_high=90, azimuth_low =0, azimuth_high=360):
    """
    Generate position of light source using 3D spherical coordinates within a pre-defined range
    :param batch_size
    :param elevation_low
    :param elevation_high
    :param azimuth_low
    :param azimuth_high
    :return numpy array of batches of random light position. Shape [batch_size, 3]
    """
    elevation = (np.random.randint(elevation_low, elevation_high, size = (batch_size, 1))) * math.pi / 180.0
    azimuth = np.random.randint(azimuth_low, azimuth_high, 90, size = (batch_size, 1)) * math.pi / 180.0
    x = np.multiply(-np.sin(elevation), np.cos(azimuth))
    y = np.cos(elevation)
    z = np.multiply(-np.sin(elevation), np.sin(azimuth))
    return np.hstack((x, y, z))

def generate_light_pos(elevation=90, azimuth=90):
  elevation = (np.array([[elevation]])) * math.pi / 180.0
  azimuth = (np.array([[azimuth]])) * math.pi / 180.0
  x = np.multiply(-np.sin(elevation), np.cos(azimuth))
  y = np.cos(elevation)
  z = np.multiply(-np.sin(elevation), np.sin(azimuth))
  return np.hstack((x, y, z))



if __name__ == "__main__":
    import scipy.misc
    from PIL import Image
    normal = np.expand_dims(scipy.misc.imread(r"D:\Projects\RenderNet\data\ply80024_p294_t105_r3.3_normal.png"), 0)


    light_dir = np.array([[3., 2.5, 3]])
    light_col = np.array([[1., 1., 0.]])
    shaded_im = np_phong_composite(normal[:, :, :, :3]/255.0, light_dir, light_col, 0.1, 1.0, background_col="White")


    scipy.misc.imsave(r"D:/numpy_phong.png", shaded_im[0])



    #=========================================================================================================================
    batch_size = 5
    images_in = tf.placeholder(shape=[batch_size, 512, 512, 3], dtype=tf.float32)
    light_dir = tf.constant(np.tile(np.array([[3., 2.5, 3]]), (batch_size,1)), dtype=tf.float32)
    light_col = tf.constant(np.tile(np.array([[1., 1., 0.]]), (batch_size,1)), dtype=tf.float32)
    shaded = tf_phong_composite(images_in / 255.0, light_dir, light_col, 0.1, 1.0)

    with tf.Session() as sess:
        result = sess.run(shaded, feed_dict={images_in: np.tile(normal[:, :, :, :3], (batch_size, 1, 1, 1)).astype(np.float32)})
    for i in range(batch_size):
        scipy.misc.imsave(r"D:/tf_phong_{0}.png".format(i), result[i])