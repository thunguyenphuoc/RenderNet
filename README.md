# RenderNet
Code release for RenderNet: A deep convolutional network for differentiable rendering from 3D shapes

__All these objects are rendered with the same network__
<p><table>
  <tr valign="top">
    <td width="23%"><img src="images/chair.gif" alt="chair" /></td>
    <td width="23%"><img src="images/table.gif" alt="table" /></td>
    <td width="23%"><img src="images/bunny.gif" alt="bunny" /></td>
    <td width="23%"><img src="images/tyra.gif" alt="tyra" /></td>
  </tr>
</table></p>

## Installation

Tested with Ubuntu 16.04, Tensorflow 1.8, CDUA 9.0, cuDNN 7. 

The following steps set up a python virtual environment and install the necessary dependencies to run the demo.



__Install Python, pip and virtualenv__

On Ubuntu, Python is automatically installed and pip is usually installed. Confirm the python and pip versions:

```
  python -V # Should be 2.7.x
  pip -V # Should be 10.x.x
```

Install these packages on Ubuntu:
```
sudo apt-get install python-pip python-dev python-virtualenv
```

__Create a virtual environment and install all dependencies__
```
cd the_folder_contains_this_READEME
virtualenv rendernetenv
source rendernetenv/bin/activate
pip install -r requirement.txt
```

__Download pre-trained model__

https://drive.google.com/open?id=1TwtJ6FXNCCm0H40nDQtZ_FIqGsgR97z3

Download the pb file and move it into the "model" folder.

__Example: rotate bunny by 360 degrees__
```
python rendernet_demo.py --voxel_path ./voxel/Misc/bunny.binvox --rotate

convert -delay 10 -loop 0 ./render/*.png animation.gif
```

## Usage

__help__
```
usage: rendernet_demo.py [-h] [--voxel_path VOXEL_PATH] [--azimuth AZIMUTH]
                         [--elevation ELEVATION]
                         [--light_azimuth LIGHT_AZIMUTH]
                         [--light_elevation LIGHT_ELEVATION] [--radius RADIUS]
                         [--render_dir RENDER_DIR] [--rotate ROTATE]

optional arguments:
  -h, --help            show this help message and exit
  --voxel_path VOXEL_PATH
                        Path to the input voxel. (default:
                        ./voxel/Misc/bunny.binvox)
  --azimuth AZIMUTH     Value of azimuth, between (0,360) (default: 250)
  --elevation ELEVATION
                        Value of elevation, between (0,360) (default: 60)
  --light_azimuth LIGHT_AZIMUTH
                        Value of azimuth for light, between (0,360) (default:
                        250)
  --light_elevation LIGHT_ELEVATION
                        Value of elevation for light, between (0,360)
                        (default: 60)
  --radius RADIUS       Value of radius, between (2.5, 4.5) (default: 3.3)
  --render_dir RENDER_DIR
                        Path to the rendered images. (default: ./render)
  --rotate ROTATE       Flag rotate and render an object by 360 degree in
                        azimuth. Overwrites early settings in azimuth.
                        (default: False)
```

__Example: chair__
```
python rendernet_demo.py --voxel_path ./voxel/Chair/64.binvox \
                         --azimuth 250 \
                         --elevation 60 \
                         --light_azimuth 90 \
                         --light_elevation 90 \
                         --radius 3.3 \
                         --render_dir ./render
```

__Example: rotate an object by 360 degrees__
```
python rendernet_demo.py --voxel_path ./voxel/Chair/64.binvox --rotate

python rendernet_demo.py --voxel_path ./voxel/Table/0.binvox --rotate

python rendernet_demo.py --voxel_path ./voxel/Misc/tyra.binvox --rotate
```

## Uninstall
```
rm -rf the_folder_contains_this_READEME # This will remove both the code and the virtual environment
```

