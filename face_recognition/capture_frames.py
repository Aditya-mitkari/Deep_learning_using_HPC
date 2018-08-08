import cv2
import argparse
import os
import math

parser = argparse.ArgumentParser(add_help = True)
parser.add_argument('--video',type = str, action='store',dest='video_name')
parser.add_argument('--fps',type = int,action='store',dest = 'fps')
args = parser.parse_args()


video = cv2.VideoCapture(args.video_name)
video_dir_name = args.video_name.split('.')[0]
vid_fps = int(math.ceil(video.get(cv2.CAP_PROP_FPS)))
frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
framepath = 'data'
framepath = os.path.join(framepath,video_dir_name)
if not os.path.exists(framepath):
    os.mkdir(framepath)
curr_frame = 0
if args.fps is not 0:
    frame_jump = int(vid_fps/args.fps)
    print('video fps:' + str(vid_fps))
    print('frame jump:' + str(frame_jump))
else:
    frame_jump = 1

while video.isOpened():

    curr_frame = curr_frame + frame_jump
    video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
    ret, frame = video.read()
    if (ret != True):
        break
    else:
        filename = framepath +'/' + str(curr_frame) + ".jpg"
        
        cv2.imwrite(filename, frame)
        

video.release()
