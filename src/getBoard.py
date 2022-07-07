import argparse
import cv2
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="This script creates a checkerboard image for calibration")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--margin_size", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=50)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    w = args.width + 1
    h = args.height + 1
    margin = args.margin_size
    block_size = args.block_size
    chessboard = np.ones((block_size * h + margin * 2, block_size * w + margin * 2,3), dtype=np.uint8) * 0
    chessboard[:,:,1] = 255

    for y in range(h):
        for x in range(w):
            if (x + y) % 2 == 0:
                sx = x * block_size + margin
                sy = y * block_size + margin
                chessboard[sy:sy + block_size, sx:sx + block_size,1] = 0

    cv2.imwrite("chessboard{}x{}.png".format(args.width, args.height), chessboard)
    cv2.imshow("chessboard", chessboard)
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()