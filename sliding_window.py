import os
import tensorflow as tf
import cv2
from tensorflow.python.platform import gfile
import numpy as np

WINDOW_SIZE=100
STRIDE=1
SIZE=68
MODEL_FILE='action_.pb'
#VID_FILE='/home/sayali/quark/kth/running/person03_running_d1_uncomp.avi'
#THRESHOLD=0.9



VID_FILE='person08_boxing_d3_uncomp.avi'
#VID_FILE='person01_walking_d3_uncomp.avi'
THRESHOLD=0.8


def getFrames(videofile):
    vidcap = cv2.VideoCapture(videofile);
    success,image = vidcap.read()
    print(image.dtype)
    count = 0
    success = True
    length=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    framestack=np.ndarray(shape=(SIZE,SIZE,length*3),dtype=np.uint8);
    success,image = vidcap.read()

    while success:
      image=cv2.resize(image,(SIZE,SIZE))
      temp_image=image
      framestack[:,:,3*count:3*count+3]=image
      count += 1
      success,image = vidcap.read()

    framestack=framestack[:,:,:count*3]
    return framestack

def slidingWindow(sess,framestack,stride):
    cnt = 1

    stack=np.ndarray(shape=[1,SIZE,SIZE,300])
    print("here\n",framestack)

    #cv2.imshow("Frame",cv2.resize(framestack[:,:,0],(300,300)))
    for i in range(0,framestack.shape[2],stride):
        try:
            cv2.imshow("Frame",cv2.resize(framestack[:,:,i:i+3],(300,300)))
            cv2.waitKey(1);
            stack[0,:,:,:]=framestack[:,:,i:i+300]
            if(cnt%5 == 0):
                classify(sess,stack)
        except:
            break;
        cnt +=1

def getclass(index):
    if index==0:
        return 'boxing'
    elif index==1:
        return 'handclapping'
    elif index==2:
        return 'handwaving'
    elif index==3:
        return 'walking'
    elif index==4:
        return 'running'


def classify(sess,test_dataset):
    x_input = sess.graph.get_tensor_by_name("input:0")
    output = sess.graph.get_tensor_by_name("output:0")
    result=sess.run(output,{x_input:test_dataset})

    if np.max(result)>=THRESHOLD:
        action=getclass(np.argmax(result))
        print(action)
    else:
        print('None')

with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(MODEL_FILE,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        framestack=getFrames(VID_FILE)
        slidingWindow(sess,framestack,STRIDE)
