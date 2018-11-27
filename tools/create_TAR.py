import glob
import tarfile
import argparse
import os

#=======================================================================================================================

fmt_cls = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=fmt_cls)

parser.add_argument('--images_path',
                    type=str,
                    default="./voxel/Misc/bunny.binvox",
                    help="Path to the image directory.")
parser.add_argument('--save_path',
                    type=str,
                    default=250,
                    help="Path to save TAR file")
parser.add_argument('--file_format',
                    type=str,
                    default='*.png',
                    help="Image format")
parser.add_argument('--to_compress',
                    type=bool,
                    default=False,
                    help="Compress .tar file or not, useful to move files around")

args = parser.parse_args()

#=======================================================================================================================

if args.to_compress:
    tar = tarfile.open(args.save_path, "w:gz")
else:
    tar = tarfile.open(args.save_path, "w")

all_images = glob.glob(os.path.join(args.imgages_path, args.file_format))
print("Found {0} images".format(len(all_images)))

for item in all_images:
    print("  Adding %s..." % item)
    tar.add(item, arcname=os.path.basename(item), recursive=False)
tar.close()