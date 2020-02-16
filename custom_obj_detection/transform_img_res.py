''' This is a custom script for image resolution transformation '''
# works only witn jpg and not with jpeg

from PIL import Image
import os
import argparse


def rescale_imgs(directory, size):
    for img in os.listdir(directory):
        im = Image.open(directory + img)
        img_resized = im.resize(size, Image.ANTIALIAS)
        img_resized.save(directory + img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rescale images")
    parser.add_argument('-d', '--directory', type=str, required=True,
    help='Directory containing the images')
    parser.add_argument('-s', '--size', type=int, required=True,
    nargs=2, metavar=('width', 'hight'), help='Image size')
    args = parser.parse_args()
    rescale_imgs(args.directory, args.size)
