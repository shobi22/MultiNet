#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Marvin Teichmann


"""
Detects Cars in an image using KittiSeg.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiSeg weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf

import time

from PIL import Image, ImageDraw, ImageFont

#Implementation of the flags interface.
#flags are used to parse command line arguments and hold input parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
#insert incl directory at the first position
sys.path.insert(1, 'incl')

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)

#define string flags
flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input', None,
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output', None,
                    'Image to apply KittiSeg.')

#submodules
default_run = 'MultiNet_ICCV'
#vgg16.py weights
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/MultiNet_ICCV.zip")

#if runs_dir or login_dir is not exit weights have to be downloaded
#then extract the weight zip files
def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
        # weights are downloaded. Nothing to do

    import zipfile
    download_name = tv_utils.download(weights_url, runs_dir)

    logging.info("Extracting MultiNet_pretrained.zip")
    #extract the zip file
    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return

#function to resize the image using scipy
def resize_label_image(image, gt_image, image_height, image_width):
    #using cubic interpolation
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    #using nearest neighbour interpolation
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image

#function to generate the output
def _output_generator(sess, tensor_list, image_pl, data_file,
                      process_image=lambda x: x):
    image_dir = os.path.dirname(data_file)
    with open(data_file) as file:
        for datum in file:
            datum = datum.rstrip() #strip
            image_file = datum.split(" ")[0] #get the name of the file by getting the first string before space
            #new  file name is combination of path of the image directory and name of image file
            image_file = os.path.join(image_dir, image_file)
            #read the image
            image = scp.misc.imread(image_file)
            #call the function process_image (resizing the image)
            image = process_image(image)

            feed_dict = {image_pl: image}
            start_time = time.time()
            output = sess.run(tensor_list, feed_dict=feed_dict)
            #return the result as objects
            yield image_file, output

#function to calculate the evaluation run time
def eval_runtime(sess, subhypes, image_pl, eval_list, data_file):
    logging.info(' ')
    logging.info('Evaluation complete. Measuring runtime.')
    #image file directory
    image_dir = os.path.dirname(data_file)
    #remove any unwanted spaces at the end of the filename
    with open(data_file) as file:
        for datum in file:
            datum = datum.rstrip()
    #get the name of image file - first part
    image_file = datum.split(" ")[0]
    #path name of the file by combining with image directory
    image_file = os.path.join(image_dir, image_file)
    #read the image
    image = scp.misc.imread(image_file)
    #resize the image by calling the function as specified height and width in hypes
    image = process_image(subhypes, image)
    #calculating the evaluation running time
    feed = {image_pl: image}
    for i in xrange(100):
        _ = sess.run(eval_list, feed_dict=feed)
    start_time = time.time()
    for i in xrange(100):
        _ = sess.run(eval_list, feed_dict=feed)
    dt = (time.time() - start_time)/100
    logging.info('Joined inference can be conducted at the following rates on'
                 ' your machine:')
    #evaluation time in milli seconds
    logging.info('Speed (msec): %f ', 1000*dt)
    #output in frames per seconds
    logging.info('Speed (fps): %f ', 1/dt)
    return dt

# function to test whether all the input images are same resolution
def test_constant_input(subhypes):
    #jitters in classification, segmentaion and detection
    road_input_conf = subhypes['road']['jitter'] 
    seg_input_conf = subhypes['segmentation']['jitter']
    car_input_conf = subhypes['detection']
    #check the height and width specified in all 3 parts are same
    gesund = True \
        and road_input_conf['image_width'] == seg_input_conf['image_width'] \
        and road_input_conf['image_height'] == seg_input_conf['image_height'] \
        and car_input_conf['image_width'] == seg_input_conf['image_width'] \
        and car_input_conf['image_height'] == seg_input_conf['image_height'] \
    # if resoltions are not same pass the error message
    if not gesund:
        logging.error("The different tasks are training"
                      "using different resolutions. Please retrain all tasks,"
                      "using the same resolution.")
        exit(1)
    return

# function to test the image file for the segmentation 
#if the images are trained with resize capability, then evaluation has to be done by resizing images
#otherwise, this method will pass the message to train the model alternatively without resizing.
def test_segmentation_input(subhypes):
    #if resize_image is false in hypes pass the error message
    if not subhypes['segmentation']['jitter']['reseize_image']:
        logging.error('')
        logging.error("Issue with Segmentation input handling.")
        logging.error("Segmentation input will be resized during this"
                      "evaluation, but was not resized during training.")
        logging.error("This will lead to bad results.")
        logging.error("To use this script please train segmentation using"
                      "the configuration:.")
        logging.error("""
{
    "jitter": {
    "reseize_image": true,
    "image_height" : 384,
    "image_width" : 1248,
    },
}""")
        logging.error("Alternatively implement evaluation using non-resized"
                      " input.")
        exit(1)
    return

# function to apply the text of classification result on top of the image
def road_draw(image, highway):
    #convert the image as an array
    im = Image.fromarray(image.astype('uint8'))
    #draw image
    draw = ImageDraw.Draw(im)
    #get the font type
    fnt = ImageFont.truetype('FreeMono/FreeMonoBold.ttf', 40)
    #shape of the image
    shape = image.shape
    #if the road is detected as highway on top of the image draw the text as highway
    if highway:
        draw.text((65, 10), "Highway",
                  font=fnt, fill=(255, 255, 0, 255))

        draw.ellipse([10, 10, 55, 55], fill=(255, 255, 0, 255),
                     outline=(255, 255, 0, 255))
    # if it is detected as small road, on top of the image draw the text as minor road
    else:
        draw.text((65, 10), "minor road",
                  font=fnt, fill=(255, 0, 0, 255))

        draw.ellipse([10, 10, 55, 55], fill=(255, 0, 0, 255),
                     outline=(255, 0, 0, 255))

    return np.array(im).astype('float32')

#function to process the image - resizing the image
def process_image(subhypes, image):
    hypes = subhypes['road']
    shape = image.shape
    # resize the input only if specified in hypes
    image_height = hypes['jitter']['image_height']
    image_width = hypes['jitter']['image_width']
    #assertion error exception will occur if image height or width specified is less than the original image's height and width
    assert(image_height >= shape[0])
    assert(image_width >= shape[1])
    #image resizing using cubic interpolation
    image = scp.misc.imresize(image, (image_height,
                                      image_width, 3),
                              interp='cubic')
    return image

# function to load the MultiNet model
def load_united_model(logdir):
    subhypes = {}
    subgraph = {}
    submodules = {}
    subqueues = {}

    first_iter = True
    #load the hypes from login directory
    meta_hypes = tv_utils.load_hypes_from_logdir(logdir, subdir="",
                                                 base_path='hypes')
    #for all the models in meta-hypes get the directory of output and input images
    for model in meta_hypes['models']:
        subhypes[model] = tv_utils.load_hypes_from_logdir(logdir, subdir=model)
        hypes = subhypes[model]
        hypes['dirs']['output_dir'] = meta_hypes['dirs']['output_dir']
        hypes['dirs']['image_dir'] = meta_hypes['dirs']['image_dir']
        submodules[model] = tv_utils.load_modules_from_logdir(logdir,
                                                              dirname=model,
                                                              postfix=model)

        modules = submodules[model]

    image_pl = tf.placeholder(tf.float32)
    #expand the shape of the array by inserting new axes in 0th positon
    image = tf.expand_dims(image_pl, 0)
    #set the shape of an array
    image.set_shape([1, 384, 1248, 3])
    decoded_logits = {}
    #for all the models in hypes
    for model in meta_hypes['models']:
        hypes = subhypes[model] #get the model
        modules = submodules[model]
        optimizer = modules['solver'] # solver- max steps of iteration and batch size and etc
        #This context manager validates that the given values are from the same graph, makes that graph the default graph, 
        #and pushes a name scope in that graph
        with tf.name_scope('Validation_%s' % model):
            reuse = {True: False, False: True}[first_iter]
            #Returns the current variable scope.
            scope = tf.get_variable_scope()
            #variable created here will be named as currentvariable and variables are not shared
            with tf.variable_scope(scope, reuse=reuse):
                logits = modules['arch'].inference(hypes, image, train=False)

            decoded_logits[model] = modules['objective'].decoder(hypes, logits,
                                                                 train=False)

        first_iter = False
    #using the context manager launch the graph in session
    sess = tf.Session()
    #saves and restores variables
    saver = tf.train.Saver()
    #loads the weights of the model from a HDF5 file
    cur_step = core.load_weights(logdir, sess, saver)

    return meta_hypes, subhypes, submodules, decoded_logits, sess, image_pl

#main function
def main(_):
    tv_utils.set_gpus_to_use()
    #if input is not given pass the error message
    if FLAGS.input is None:
        logging.error("No input was given.")
        logging.info(
            "Usage: python demo.py --input data/test.png "
            "[--output_image output_image] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] ")
        exit(1)
    
    #if log directory is empty
    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'MultiNet')
        else:
            runs_dir = 'RUNS'
        #call the function to extract and download the weights
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    logging.info("Loading model from: {}".format(logdir))

    # Loads the model from rundir
    load_out = load_united_model(logdir)

    # Create list of relevant tensors to evaluate
    meta_hypes, subhypes, submodules, decoded_logits, sess, image_pl = load_out

    seg_softmax = decoded_logits['segmentation']['softmax'] #softmax in segmentation
    pred_boxes_new = decoded_logits['detection']['pred_boxes_new'] #rough bounding boxes in detecgtion
    pred_confidences = decoded_logits['detection']['pred_confidences'] #confidence level in detection
    if len(meta_hypes['model_list']) == 3:
        road_softmax = decoded_logits['road']['softmax'][0] #softmax in classification
    else:
        road_softmax = None

    eval_list = [seg_softmax, pred_boxes_new, pred_confidences, road_softmax]

    # Run some tests on the hypes
    test_constant_input(subhypes)
    test_segmentation_input(subhypes)

    # Load and reseize Image
    image_file = FLAGS.input
    #read the image
    image = scp.misc.imread(image_file)
    #resizing the image in classification
    hypes_road = subhypes['road']
    shape = image.shape
    image_height = hypes_road['jitter']['image_height']
    image_width = hypes_road['jitter']['image_width']
    assert(image_height >= shape[0])
    assert(image_width >= shape[1])
    #resizing using cubic interpolation
    image = scp.misc.imresize(image, (image_height,
                                      image_width, 3),
                              interp='cubic')

    import utils.train_utils as dec_utils

    # Run KittiSeg model on image
    feed_dict = {image_pl: image}
    output = sess.run(eval_list, feed_dict=feed_dict)

    seg_softmax, pred_boxes_new, pred_confidences, road_softmax = output

    # Create Segmentation Overlay
    shape = image.shape
    seg_softmax = seg_softmax[:, 1].reshape(shape[0], shape[1])
    # if the segmentaion confidence more than 0.5 it is considered as hard softmax
    hard = seg_softmax > 0.5 
    overlay_image = tv_utils.fast_overlay(image, hard)

    # Draw Detection Boxes
    new_img, rects = dec_utils.add_rectangles(
        subhypes['detection'], [overlay_image], pred_confidences,
        pred_boxes_new, show_removed=False,
        use_stitching=True, rnn_len=subhypes['detection']['rnn_len'],
        min_conf=0.50, tau=subhypes['detection']['tau'])

    # Draw road classification
    highway = (np.argmax(road_softmax) == 1)
    new_img = road_draw(new_img, highway)

    logging.info("")

    # Printing some more output information
    threshold = 0.5
    accepted_predictions = []
    # removing predictions <= threshold
    for rect in rects:
        if rect.score >= threshold:
            accepted_predictions.append(rect)

    print('')
    logging.info("{} Cars detected".format(len(accepted_predictions)))

    # Printing coordinates of predicted rects.
    for i, rect in enumerate(accepted_predictions):
        logging.info("")
        logging.info("Coordinates of Box {}".format(i))
        logging.info("    x1: {}".format(rect.x1))
        logging.info("    x2: {}".format(rect.x2))
        logging.info("    y1: {}".format(rect.y1))
        logging.info("    y2: {}".format(rect.y2))
        logging.info("    Confidence: {}".format(rect.score))

    if len(meta_hypes['model_list']) == 3:
        logging.info("Raw Classification Softmax outputs are: {}"
                     .format(output[0][0]))

    # Save output image file
    if FLAGS.output is None:
        output_base_name = FLAGS.input
        out_image_name = output_base_name.split('.')[0] + '_out.png'
    else:
        out_image_name = FLAGS.output

    scp.misc.imsave(out_image_name, new_img)

    logging.info("")
    logging.info("Output image has been saved to: {}".format(
        os.path.realpath(out_image_name)))

    logging.info("")
    logging.warning("Do NOT use this Code to evaluate multiple images.")

    logging.warning("Demo.py is **very slow** and designed "
                    "to be a tutorial to show how the MultiNet works.")
    logging.warning("")
    logging.warning("Please see this comment, if you like to apply demo.py to"
                    " multiple images see:")
    logging.warning("https://github.com/MarvinTeichmann/KittiBox/"
                    "issues/15#issuecomment-301800058")

    exit(0)

if __name__ == '__main__':
    tf.app.run()
