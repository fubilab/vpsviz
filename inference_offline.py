import cv2

import torch
import torch.nn.functional as F
import numpy as np
import dsacstar

import argparse
import os
import re

from network import Network
from torchvision import transforms

import atexit
import socket
import threading
from datetime import datetime

host, port = "127.0.0.1", 25001

def parse_args():
    p = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--hypotheses', '-hyps', type=int, default=64,
                   help='number of hypotheses, i.e. number of RANSAC iterations')

    p.add_argument('--fps', type=int, default=1,
                   help='fps')

    p.add_argument('--threshold', '-t', type=float, default=10,
                   help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    p.add_argument('--focal', type=float, default=565,
                   help='focal length')

    p.add_argument('--inlieralpha', '-ia', type=float, default=100,
                   help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

    p.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                   help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

    p.add_argument('--session', '-sid', default='',
                   help='custom session name appended to output files, useful to separate different runs of a script')

    p.add_argument('--tiny', '-tiny', action='store_true',
                   help='Load a model with massively reduced capacity for a low memory footprint.')

    return p.parse_args()


def define_mask(img_height, img_width):
    # Mask dimensions as percentages of the image dimensions
    mask_width_percentage = 0.48
    mask_height_percentage = 0.50
    # Offset as percentages of the image dimensions
    offset_right_percentage = 0.271
    offset_down_percentage = 0.313
    # Calculate actual dimensions and offset
    mask_width = int(img_width * mask_width_percentage)
    mask_height = int(img_height * mask_height_percentage)
    offset_right = int(img_width * offset_right_percentage)
    offset_down = int(img_height * offset_down_percentage)
    # Create a mask filled with zeros
    mask = torch.ones((img_height, img_width))
    # Define the region to fill with ones
    top_left_x = offset_right
    top_left_y = offset_down
    bottom_right_x = top_left_x + mask_width
    bottom_right_y = top_left_y + mask_height
    # Fill the region with ones
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
    # mask = mask.cuda()
    return mask


def inference(image, transform, network):
    image = transform(image)
    image = image.unsqueeze(0)

    # mask = define_mask(480, 853)
    # image = image * mask.unsqueeze(0)
    # print(image.shape)

    scene_coordinates = network(image)
    scene_coordinates = scene_coordinates.cpu()

    out_pose = torch.zeros((4, 4))

    # pose from RGB
    dsacstar.forward_rgb(
        scene_coordinates,
        out_pose,
        args.hypotheses,
        args.threshold,
        args.focal,
        float(image.size(3) / 2),  # principal point assumed in image center
        float(image.size(2) / 2),
        args.inlieralpha,
        args.maxpixelerror,
        network.OUTPUT_SUBSAMPLE)

    # avg_time += time.time()-start_time
    # cv2.imwrite('./images/', image)
    # np.savetxt("./poses/" + file + ".txt", out_pose, fmt='%.16f')  # Save to a text file
    return out_pose


torch_busy = False
img_queue = None
transform = None
network = None
lock = threading.Lock()
frame_number = 0

def convert_right_to_left_handed(matrix):
    # Transformation matrix to flip the y-axis
    flip_y = np.array([
        [-1,  0,  0, 0],
        [0, 1,  0, 0],
        [0,  0,  1, 0],
        [0,  0,  0, 1]
    ])

    # Perform the matrix multiplication
    converted_matrix = np.dot(flip_y, matrix)
    return converted_matrix

def process_frame():
    global lock
    global frame_number

    filename = file_list[frame_number]
    file_path = os.path.join(directory, filename)
    rgb_frame = cv2.imread(file_path)

    pose = inference(rgb_frame, transform, network)
    pose = convert_right_to_left_handed(pose)

    # send pose over socket
    frame_number += 1
    msg = ''
    for t in pose:
        for v in t:
            msg += ',' + str(round(float(v), 4))
    msg = msg[1:]
   
    # send pose header
    poseHeader = 'pose:' + str(len(msg))
    print("⏩ %s" % poseHeader)
    sock.send(poseHeader.encode("utf-8"))
    sizeConfirmation = sock.recv(1024)
    print("✅ %s" % sizeConfirmation.decode('utf-8'))
   
    # send pose message
    print("⏩ %s" % msg)
    sock.sendall(msg.encode("utf-8"))
    response = sock.recv(1024).decode("utf-8")
    print("✅ %s" % response)

    # image path message
    msg = file_path

    # send image path header
    imgHeader = 'imgpath:' + str(len(msg))
    print("⏩ %s" % imgHeader)
    sock.send(imgHeader.encode("utf-8"))
    sizeConfirmation = sock.recv(1024)
    print("✅ %s" % sizeConfirmation.decode('utf-8'))

    # send image path message
    print("⏩ %s" % msg)
    sock.sendall(msg.encode("utf-8"))
    response = sock.recv(1024).decode("utf-8")
    print("✅ %s" % response)

def main_loop(args):
    global img_queue
    global transform
    global network
    global lock
    global file_list
    global directory

    directory = "/media/fubintlab/nvmedrive/dsac/dsacstar/rgb/"
    file_list = [f for f in os.listdir(directory) if f.endswith('.png')]
    file_list = sorted(file_list, key=lambda s: [int(text) if text.isdigit() else text.lower() for text in
                                                 re.split('([0-9]+)', s)])

    frames = {}
    #frame_number = 0

    # network_name = "model_e2e_550_final_epoch_237.net"
    network_name = "model_e2e_150_epoch_150.net"
    tiny = True

    # load network
    network = Network(torch.zeros((3)), args.tiny)
    network.load_state_dict(torch.load(network_name, map_location=torch.device('cpu')))
    network.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(480),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.44],
            std=[0.25]
        )
    ])
    with torch.no_grad():
        while True:
            process_frame()


def exit_handler():
    if "sock" in globals():
        sock.close()


atexit.register(exit_handler)


if __name__ == '__main__':
    args = parse_args()

    # SOCK_STREAM means TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("⏳ Connecting to %s:%s" % (host, port))
    sock.connect((host, port))
    print("✅ Connected to host")

    main_loop(args)


