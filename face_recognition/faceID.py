import os
import argparse
import shutil
import cv2
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser(add_help = True)
parser.add_argument('--video',type = str, action='store',dest='video')
parser.add_argument('--fps',type = int,action='store',dest = 'fps')
args = parser.parse_args()
folder_name = args.video.split('.')[0]

print('Extracting frames...\n')
os.system('python3 capture_frames.py --video '+args.video+' --fps '+ str(args.fps))
print('detecting faces...\n')
os.system('./run_preprocess.sh')
print('running classifier...\n')
os.system('./run_test.sh')
shutil.rmtree(os.path.join('data',folder_name))
shutil.rmtree(os.path.join('output','intermediate',folder_name))
os.remove(os.path.join('output','coords.txt'))

print('Writing tagged video...')
imglist = os.listdir(os.path.join('tagged',folder_name))
imglist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


img = cv2.imread(os.path.join('tagged',folder_name,imglist[0]))
height , width , layers =  img.shape
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work

if args.fps is not 0:
    framerate = args.fps
else:
    framerate = math.ceil(cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS))

vid = cv2.VideoWriter(os.path.join('output_video', folder_name + '.avi'), fourcc, framerate, (width, height))
for image in imglist:
    img = cv2.imread(os.path.join('tagged',folder_name,image))
    vid.write(img)


#cv2.destroyAllWindows()
vid.release()
shutil.rmtree(os.path.join('tagged',folder_name))







