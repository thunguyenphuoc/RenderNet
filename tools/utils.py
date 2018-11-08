"""
H
"""

import io
import tarfile
import time
import zlib
import scipy


import numpy as np
import tools.binvox_rw as binvox_rw

"""
Adjusted from Andrew Brock (https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)
Function to write and read Tar files (using a python generator)
Tar files are used to store training data (images, voxels)
"""

PREFIX = 'data/'
SUFFIX = '.npy.z'

class NpyTarWriter(object):
    def __init__(self, fname):
        self.tfile = tarfile.open(fname, 'w|')

    def add(self, arr, name):

        sio = io.BytesIO()
        np.save(sio, arr)
        zbuf = zlib.compress(sio.getvalue())
        sio.close()

        zsio = io.BytesIO(zbuf)
        tinfo = tarfile.TarInfo('{}{}{}'.format(PREFIX, name, SUFFIX))
        tinfo.size = len(zbuf)
        tinfo.mtime = time.time()
        zsio.seek(0)
        self.tfile.addfile(tinfo, zsio)
        zsio.close()

    def close(self):
        self.tfile.close()


class NpyTarReader(object):
    def __init__(self, fname):
        self.tfile = tarfile.open(fname, 'r|')

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        entry = self.tfile.next()
        if entry is None:
            self.close()
            raise StopIteration()

        fileobj = self.tfile.extractfile(entry)
        contents = fileobj.read()

        name_components = entry.name.split('.')

        # handle .z
        if name_components[-1].lower() == "z":
            contents = zlib.decompress(contents)
            name_components.pop()  # remove 'z' from extensions

        # handle .npy
        if name_components[-1].lower() == "npy":
            array = np.load(io.BytesIO(contents))
            return array

        # handle .binvox
        if name_components[-1].lower() == "binvox":
            voxels = binvox_rw.read_as_3d_array(io.BytesIO(contents))
            model_name = name_components[0].split('_')
            if 'ply' in model_name[0]:
                return voxels.data, model_name[0]
            else:
                model_name = model_name[0] + '_' + model_name[1] + '_' + model_name[2]+'_clean'
                return voxels.data, model_name

        # handle .jpg or .png
        if name_components[-1].lower() == "jpg" or name_components[-1].lower() == "png":
            try:
                image = scipy.misc.imread(io.BytesIO(contents)).astype(np.float32)
                name_components = entry.name[:-4]
                model_name = name_components.split('_')
                model_name = model_name[0] + '_' + model_name[1] + '_' + model_name[
                    2] + '_clean.binvox'  # Model name to query the corresponding 3d voxels models
                return image, name_components
            except RuntimeError:
                return None, None
            except OSError:
                return None, None
            except TypeError:
                return None, None

        return None

    def close(self):
        self.tfile.close()


