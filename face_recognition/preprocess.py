import argparse
import glob
import logging
import multiprocessing as mp
from multiprocessing import Manager
import os
import time
import pickle
import cv2
import json

from align_dlib import AlignDlib

logger = logging.getLogger(__name__)

mgr=Manager()
box_locs = mgr.dict()
align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def main(input_dir, output_dir, crop_dim):
    start_time = time.time()
    pool = mp.Pool(processes=mp.cpu_count())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for image_dir in os.listdir(input_dir):
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.basename(image_dir)))
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

    image_paths = glob.glob(os.path.join(input_dir, '**/*.jpg'))
    for index, image_path in enumerate(image_paths):
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
        output_path = os.path.join(image_output_dir, os.path.basename(image_path))
        pool.apply_async(preprocess_image, (image_path, output_path, crop_dim))

    pool.close()
    pool.join()
    #print(box_locs)
    with open('./output/coords.txt','w') as file:
        json.dump(box_locs.copy(),file)
    logger.info('Completed in {} seconds'.format(time.time() - start_time))


def preprocess_image(input_path, output_path, crop_dim):
    """
    Detect face, align and crop :param input_path. Write output to :param output_path
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    """
    tag_path = 'tagged/'
    filename = input_path.split("/")[2]
    folder_name = input_path.split("/")[1]
    tag_path = tag_path + folder_name
    if not (os.path.isdir(tag_path)):
        os.mkdir(tag_path)
    tag_path = tag_path + "/" + filename
    bbs,image_list = _process_image(input_path, crop_dim)
    filename = filename.split('.')[0]
    facenum = 0
    tagged_img = _buffer_image(input_path)


    for image in image_list:
        if image is not None:

            x, y, w, l = rect_to_bb(bbs[facenum])
            cv2.rectangle(tagged_img, (x, y), (x + w, y + l), (0, 255, 0), 2)

            logger.debug('Writing processed file: {}'.format(output_path))
            filename_n = filename + '_' + str(facenum) + '.jpg'
            cv2.imwrite('output/intermediate/' +folder_name + '/' + filename_n, image)
            box_locs[filename_n] = (x,y,w,l)
            facenum = facenum +1


        else:
            logger.warning("Skipping filename: {}".format(input_path))

    if tagged_img is not None:
        cv2.imwrite(tag_path, cv2.cvtColor(tagged_img, cv2.COLOR_RGB2BGR))







def _process_image(filename, crop_dim):
    image = None
    aligned_image = None

    image = _buffer_image(filename)

    if image is not None:
        bbs,aligned_list = _align_image(filename, image, crop_dim)
    else:
        raise IOError('Error buffering image: {}'.format(filename))

    return bbs,aligned_list


def _buffer_image(filename):
    logger.debug('Reading image: {}'.format(filename))
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _align_image(filename, image, crop_dim):
    #bb = align_dlib.getLargestFaceBoundingBox(image)
    bbs = align_dlib.getAllFaceBoundingBoxes(image)
    ##tagged_img = image.copy()
    aligned_list = list()
    for bb in bbs:
        ##x, y, w, l = rect_to_bb(bb)
        ##cv2.rectangle(tagged_img, (x, y), (x + w, y + l), (0, 255, 0), 2)

        aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        if aligned is not None:
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            aligned_list.append(aligned);

    return bbs,aligned_list


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='data', dest='input_dir')
    parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
    parser.add_argument('--crop-dim', type=int, action='store', default=180, dest='crop_dim',
                        help='Size to crop images to')

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.crop_dim)

