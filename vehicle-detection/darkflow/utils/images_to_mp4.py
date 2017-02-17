import cv2
import argparse
import numpy as np
import glob, os


def main(args):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    writer = cv2.VideoWriter('/tmp/outpu.avi', fourcc, 10, (1920, 1200))
    os.chdir(args.image_folder)
    file_list = glob.glob("*.jpg")
    sorted_files = sorted(file_list, key=lambda x: int(x.split('.')[0]))
    for file in sorted_files:
        print(file)
        img = cv2.imread(file)
        writer.write(img)

    writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts images in a directory to a mp4 for visualization')
    parser.add_argument('-i', '--image_folder', help='Input image folder')
    parser.add_argument('-d', '--decompress', default=False, type=bool, help='Decompress the images')
    parser.add_argument('-o', '--output', default='out', help='Output mp4 file name without extension')
    args = parser.parse_args()
    main(args)

