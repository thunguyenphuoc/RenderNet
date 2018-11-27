import numpy as np
import math
import os
import time
import scipy.misc
import scipy.io
import random

import tools.utils as utils
import tools.binvox_rw as binvox_rw


def extract_param_from_names(image_path):
    """
    Extract pose parameters from an image name.
    The name should contain pattern: "_p[azimuth]_t[elevation]_r[scale]"
    :param image_path: name to extract parameters from.
    :return: A tuple of pose parameter (Azimuth, elevation, scale). Shape [1, 3]
    """
    azimuth_index = image_path.find('_p')
    elevation_index = image_path.find('_t')
    scale_index = image_path.find('_r')
    azimuth = float(image_path[azimuth_index + 2: elevation_index]) * math.pi / 180.0
    scale = 3.3 / float(image_path[scale_index + 2: scale_index + 5])
    elevation = 90. - float(image_path[elevation_index + 2: scale_index])  # Map from range [10;170] (Up-Z axis =0) to [80;-80] ((horizontal X axis = 0)
    elevation = elevation * math.pi / 180.0

    param = np.expand_dims(np.array([azimuth, elevation, scale]), axis=0)
    return param

def model_loader(cfg, model_path):
    """
    A data loader built upon python generator to load binvox models and their names simultaneously from a TAR file
    :param cfg: config dictionary
    :param model_path: path to TAR file containing binvox models
    :return batch containing tuple of (binvox, model name)
    """
    chunk_size = cfg['batch_size']*cfg['batches_chunk']
    mod_batch = np.zeros(shape=(chunk_size, 64, 64, 64,1), dtype=np.float32)
    mod_name_batch = []
    reader = utils.NpyTarReader(model_path) #Generator, reads images and object this until having gone through all of the images data #Create a new one for every epoch
    counter = 0
    for ix, (mod, mod_name) in enumerate(reader):
        idx = ix % chunk_size
        mod_batch[idx] = np.reshape(mod.astype(np.float32), (64,64,64,1)) #Append one image from the generator Reader in to the placeholder list
        mod_name_batch.append(mod_name)
        counter += 1
        if counter == chunk_size: #for every batch
            yield ( mod_batch, mod_name_batch) #Only use dilation for now since erosion might cause some missing parts in thin elements
            counter = 0
            mod_batch.fill(0)
            mod_name_batch = []
    if counter > 0:
        # pad to nearest multiple of batch_size
        if counter % cfg['batch_size'] != 0:
            repetitions = int(np.ceil(float(cfg['batch_size']) / counter))
            mod_batch = np.repeat(mod_batch[:counter, ...], repetitions, axis=0)
            mod_batch = mod_batch[:cfg['batch_size']]
            mod_name_batch = np.repeat(mod_name_batch[:counter], repetitions, axis=0)
            mod_name_batch = mod_name_batch[:cfg['batch_size']]
            reader.close()
            yield (mod_batch, mod_name_batch)

def data_loader(cfg, img_path, model_path, validation_mode = False, flatten = False, img_res = 256, add_noise = False):
    """
    A data loader built upon python generator to load target images, binvox models, view parameters
    and their names simultaneously from a TAR file
    :param cfg: config dictionary
    :param img_path: path to TAR file containing target images
    :param model_path: path to folder containing binvox models
    :param validation_mode
    :param flatten: make channel = 1
    :param img_res: size of label images
    :param add_noise: add small random noise to images
    :return batch containing tuple of (images, binvox, view parameters, model name)
    """

    if validation_mode:
        chunk_size = cfg['batch_size']
    else:
        chunk_size = cfg['batch_size'] * cfg['batches_chunk']

    if flatten:
        im_batch = np.zeros(shape=(chunk_size, img_res, img_res, 1), dtype=np.float32)
    else:
        im_batch = np.zeros(shape=(chunk_size, img_res, img_res, 3), dtype=np.float32)
    mod_batch = np.zeros(shape=(chunk_size, 64, 64, 64,1), dtype=np.float32)
    param_batch = np.zeros(shape=(chunk_size, 3), dtype=np.float32)

    mod_name_batch = []
    reader = utils.NpyTarReader(img_path) #Generator, reads images and object this until having gone through all of the images data #Create a new one for every epoch
    start = time.time()
    counter = 0

    for ix, (img, img_name) in enumerate(reader):
        if img is None or img_name is None:
            continue
        idx = ix % chunk_size
        img = img.astype(np.float32)
        if flatten:
            if len(img.shape) == 3:
                im_batch[idx] = np.reshape(np.mean(img, axis = 2), (img_res, img_res, 1))  # Append one image from the generator Reader in to the placeholder list
            else:
                im_batch[idx] = np.reshape(img, (img_res, img_res, 1))
        else:
            im_batch[idx] = np.reshape(img[:, :, :3], (img_res, img_res, 3))  # ignore alpha
        if add_noise:
            im_batch[idx] += np.random.uniform(low = 0.0, high = 1.0, size = im_batch[idx].shape)

        #Extract view parameteres and convert to radian
        azimuth_index = img_name.find('_p')
        elevation_index = img_name.find('_t')
        scale_index = img_name.find('_r')
        azimuth   = float(img_name[azimuth_index + 2 : elevation_index]) * math.pi /180.0
        scale = 3.3 / float(img_name[scale_index + 2: scale_index+5])
        elevation = 90. - float(img_name[elevation_index + 2 : scale_index])  # Map from range [10;170] (Up-Z axis =0) to [80;-80] ((horizontal X axis = 0)
        elevation = elevation * math.pi / 180.0
        param_batch[idx] = np.array([azimuth, elevation, scale])

        mod_name_batch.append(img_name)
        content = img_name.split("_")
        if 'ply' in content[0]:
            mod_name = content[0] + ".binvox"
            model_dir = os.path.join(model_path, mod_name)
        else:
            mod_name = "model_chair_{1}_clean.binvox".format(content[1], content[2])
            model_dir = os.path.join(model_path, mod_name)
            if os.path.exists(model_dir) == False:
                mod_name = "model_normalized_{0}_clean.binvox".format(content[2])
                model_dir = os.path.join(model_path, mod_name)

        with open(model_dir, 'rb') as f:
            mod_batch[idx] = np.reshape(binvox_rw.read_as_3d_array(f).data.astype(np.float32), (64, 64, 64, 1))  # Append one 3d model based on model_name from the generator Reader in to the placeholder list
        counter += 1
        if counter == chunk_size: #for every batch
            yield (im_batch, mod_batch, param_batch, mod_name_batch)
            counter = 0
            im_batch.fill(0)
            mod_name_batch = []
            param_batch.fill(0)
            mod_batch.fill(0)

    if counter > 0:
        # pad to nearest multiple of batch_size
        if counter % cfg['batch_size'] != 0:
            repetitions = int(np.ceil(float(cfg['batch_size']) / counter))
            im_batch = np.repeat(im_batch[:counter, ...], repetitions, axis=0)
            im_batch = im_batch[:cfg['batch_size']]
            mod_batch = np.repeat(mod_batch[:counter, ...], repetitions, axis=0)
            mod_batch = mod_batch[:cfg['batch_size']]
            param_batch = np.repeat(param_batch[:counter, ...], repetitions, axis=0)
            param_batch = param_batch[:cfg['batch_size']]
            mod_name_batch = np.repeat(mod_name_batch[:counter], repetitions, axis=0)
            mod_name_batch = mod_name_batch[:cfg['batch_size']]

            reader.close()
            yield (im_batch, mod_batch, param_batch, mod_name_batch)

def data_loader_image_texture_normal_face(cfg, img_path, model_path, texture_path, normal_path, validation_mode = False, img_res = 256, add_noise = True):
    if validation_mode:
        chunk_size = cfg['batch_size']
    else:
        chunk_size = cfg['batch_size'] * cfg['batches_chunk']

    im_batch = np.zeros(shape=(chunk_size, img_res, img_res, 3), dtype=np.float32)
    normal_batch = np.zeros(shape=(chunk_size, img_res, img_res, 3), dtype=np.float32)
    mod_batch = np.zeros(shape=(chunk_size, 64, 64, 64,1), dtype=np.float32)
    texture_batch = np.zeros(shape=(chunk_size, 199), dtype=np.float32)
    param_batch = np.zeros(shape=(chunk_size, 3), dtype=np.float32)
    mod_name_batch = []
    reader = utils.NpyTarReader(img_path) #Generator, reads images and object this until having gone through all of the images data #Create a new one for every epoch
    start = time.time()
    counter = 0

    for ix, (img, img_name) in enumerate(reader):
        idx = ix % chunk_size
        img = img.astype(np.float32)
        im_batch[idx] = np.reshape(img[:, :, :3], (img_res, img_res, 3))  # Append one image from the generator Reader in to the placeholder list
        if add_noise:
            im_batch[idx] += np.random.uniform(low = 0.0, high = 1.0, size = im_batch[idx].shape)

        content  = img_name.split('_')
        mod_name = "{0}.binvox".format(content[0])
        texture_name = "beta{0}.mat".format(content[0].split('ly')[1])
        texture_param = scipy.io.loadmat(os.path.join(texture_path, texture_name))
        texture_batch[idx] = np.reshape(texture_param['beta'].astype(np.float32), 199)
        normal_batch[idx] = scipy.misc.imread(os.path.join(normal_path, img_name + ".png")).astype(np.float32)[:, :, :3]

        #Extract view parameteres and convert to radian
        azimuth_index = img_name.find('_p')
        elevation_index = img_name.find('_t')
        scale_index = img_name.find('_r')
        azimuth   = float(img_name[azimuth_index + 2 : elevation_index]) * math.pi /180.0
        scale = 3.3 / float(img_name[scale_index + 2: scale_index+5])
        elevation = 90. - float(img_name[elevation_index + 2 : scale_index])  # Map from range [10;170] (Up-Z axis =0) to [80;-80] ((horizontal X axis = 0)
        elevation = elevation * math.pi / 180.0
        param_batch[idx] = np.array([azimuth, elevation, scale])

        mod_name_batch.append(img_name)
        model_dir = os.path.join(model_path, mod_name)
        with open(model_dir, 'rb') as f:
            mod_batch[idx] = np.reshape(binvox_rw.read_as_3d_array(f).data.astype(np.float32), (64, 64, 64, 1))#Append one 3d model based on model_name from the generator Reader in to the placeholder list

        counter += 1
        if counter == chunk_size: #for every batch
            print("Data loading done " + str(time.time() - start))
            yield (im_batch, normal_batch, mod_batch, texture_batch, param_batch, mod_name_batch)
            start = time.time()
            counter = 0
            im_batch.fill(0)
            normal_batch.fill(0)
            mod_name_batch = []
            param_batch.fill(0)
            mod_batch.fill(0)
            texture_batch.fill(0)
    if counter > 0:
        # pad to nearest multiple of batch_size
        if counter % cfg['batch_size'] != 0:
            repetitions = int(np.ceil(float(cfg['batch_size']) / counter))
            im_batch = np.repeat(im_batch[:counter, ...], repetitions, axis=0)
            im_batch = im_batch[:cfg['batch_size']]
            normal_batch = np.repeat(normal_batch[:counter, ...], repetitions, axis=0)
            normal_batch = normal_batch[:cfg['batch_size']]
            mod_batch = np.repeat(mod_batch[:counter, ...], repetitions, axis=0)
            mod_batch = mod_batch[:cfg['batch_size']]
            texture_batch = np.repeat(texture_batch[:counter, ...], repetitions, axis=0)
            texture_batch = texture_batch[:cfg['batch_size']]
            param_batch = np.repeat(param_batch[:counter, ...], repetitions, axis=0)
            param_batch = param_batch[:cfg['batch_size']]
            mod_name_batch = np.repeat(mod_name_batch[:counter], repetitions, axis=0)
            mod_name_batch = mod_name_batch[:cfg['batch_size']]
            reader.close()
            yield (im_batch, normal_batch, mod_batch, texture_batch, param_batch, mod_name_batch)
