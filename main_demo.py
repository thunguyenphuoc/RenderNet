import os
import argparse
import math
import numpy as np
from scipy import misc

import tensorflow as tf

from tools import binvox_rw, Phong_shading

# =======================================================================================================
# RenderNet Phong shading demo
# The network output a normal map from a 3D voxel.
# The normal map is then used to create phong shading with lighting control.
# ======================================================================================================

# Phong shading paramters
AMBIENT_IN = (0.1)
K_DIFFUSE = .9
LIGHT_COL = np.array([[1., 1., 1.]])


def load_graph(frozen_graph_filename):
  with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def, name="")
  return graph


def compute_pose_param(azimuth, elevation, radius):
  phi = azimuth * math.pi / 180.0
  theta = (90 - elevation) * math.pi / 180
  param = np.array([phi, theta, 3.3 / radius])
  param = np.expand_dims(param, axis=0)
  return param


def render(azimuth, elevation, radius, sess, voxel, light_dir,
           render_dir, count, light_azimuth, light_elevation,
           model_name):
  param = compute_pose_param(azimuth, elevation, radius)

  # Creating normal map
  rendered_samples = sess.run("encoder/output:0",
                              feed_dict={"real_model_in:0": voxel,
                                         "view_name:0": param,
                                         "patch_size:0": 128,
                                         "is_training:0": False})
  # Create phong shaded images
  img_phong = Phong_shading.np_phong_composite(rendered_samples,
                                 light_dir, LIGHT_COL,
                                 AMBIENT_IN, K_DIFFUSE)

  image_out = np.clip(255. * img_phong[0], 0, 255).astype(np.uint8)

  save_path = os.path.join(render_dir,
                           str(count).zfill(3) + "_" + model_name +
                           "_pose_%f_%f_%f_light_%f_%f.png" %
                           (azimuth, elevation, radius,
                            light_azimuth, light_elevation))
  print(save_path)
  misc.imsave(save_path, image_out)

def generate_light_pos(elevation=90, azimuth=90):
  elevation = (90 - np.array([[elevation]])) * math.pi / 180.0
  azimuth = (np.array([[azimuth]])) * math.pi / 180.0
  x = np.multiply(np.sin(elevation), np.cos(azimuth))
  y = np.multiply(np.sin(elevation), np.sin(azimuth))
  z = np.cos(elevation)
  return np.hstack((x, y, z))


def main():
  fmt_cls = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=fmt_cls)

  parser.add_argument('--voxel_path',
                      type=str,
                      default="./voxel/Misc/bunny.binvox",
                      help="Path to the input voxel.")
  parser.add_argument('--azimuth',
                      type=float,
                      default=250,
                      help="Value of azimuth, between (0,360)")
  parser.add_argument('--elevation',
                      type=float,
                      default=60,
                      help="Value of elevation, between (0,360)")
  parser.add_argument('--light_azimuth',
                      type=float,
                      default=250,
                      help="Value of azimuth for light, between (0,360)")
  parser.add_argument('--light_elevation',
                      type=float,
                      default=60,
                      help="Value of elevation for light, between (0,360)")
  parser.add_argument('--radius',
                      type=float,
                      default=3.3,
                      help="Value of radius, between (2.5, 4.5)")
  parser.add_argument('--render_dir',
                      type=str,
                      default='./render',
                      help='Path to the rendered images.')
  parser.add_argument('--rotate',
                      type=bool,
                      default=False,
                      help=('Flag rotate and render an object by 360 degree in azimuth. \
                            Overwrites early settings in azimuth.'))
  args = parser.parse_args()

  # We use our "load_graph" function
  graph = load_graph("./model/3d2d_renderer.pb")

  with tf.Session(graph=graph) as sess:
    if not os.path.exists(args.render_dir):
        os.makedirs(args.render_dir)

    azimuth = args.azimuth
    elevation = args.elevation
    radius = args.radius
    light_elevation = args.light_elevation
    light_azimuth = args.light_azimuth
    light_dir = generate_light_pos(light_elevation, light_azimuth)

    with open(args.voxel_path, 'rb') as f:
        voxel = np.reshape(
          binvox_rw.read_as_3d_array(f).data.astype(np.float32),
          (1, 64, 64, 64, 1))
        model_name = os.path.basename(args.voxel_path).split('.binvox')[0]

    if args.rotate:
      # Automatically rorate the object by 360 degree in azimuth
      count = 0
      for azimuth in np.arange(0.0, 360.0, 5.0):
        render(azimuth, elevation, radius, sess, voxel, light_dir,
               args.render_dir, count, light_azimuth, light_elevation,
               model_name)
        count = count + 1
    else:
      # Manually set up pose and light
      render(azimuth, elevation, radius, sess, voxel, light_dir,
             args.render_dir, 0, light_azimuth, light_elevation,
             model_name)


if __name__ == "__main__":
  main()
