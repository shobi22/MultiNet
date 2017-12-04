#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runs MultiNet on a whole bunch of input images.


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import scipy as scp
import scipy.misc
import numpy as np
import tensorflow as tf

import time

#Implementation of the flags interface.
#flags are used to parse command line arguments and hold input parameters
flags = tf.app.flags
FLAGS = flags.FLAGS

#insert incl directory at the first position
sys.path.insert(1, os.path.realpath('incl'))


import train as united_train

import tensorvision.train as train
import tensorvision.utils as utils
import tensorvision.core as core
from PIL import Image, ImageDraw, ImageFont


#define string flags
flags.DEFINE_string('data',
                    "data_road/testing.txt",
                    'Text file containing images.')

flags.DEFINE_bool('speed_test',
                  False,
                  'Only measure inference speed.')

res_folder = 'results'

#function to ouput generator
def _output_generator(sess, tensor_list, image_pl, data_file,
                      process_image=lambda x: x):
    image_dir = os.path.dirname(data_file)
    with open(data_file) as file:
        for datum in file:
            #strip
            datum = datum.rstrip()
            #get the name of the file by getting the first string before space
            image_file = datum.split(" ")[0]
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
    sess.run(eval_list, feed_dict=feed)
    sess.run(eval_list, feed_dict=feed)
    sess.run(eval_list, feed_dict=feed)
    for i in xrange(100):
        _ = sess.run(eval_list, feed_dict=feed)
    #calculating evaluation time
    start_time = time.time()
    for i in xrange(100):
        _ = sess.run(eval_list, feed_dict=feed)
    dt = (time.time() - start_time)/100
    logging.info('Joined inference can be conducted at the following rates on'
                 ' your machine:')
    logging.info('Speed (msec): %f ', 1000*dt)
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

#function to run the evaluation
def run_eval(load_out, output_folder, data_file):
    meta_hypes, subhypes, submodules, decoded_logits, sess, image_pl = load_out
    #if the model list is having 3 models; classification, detection and segentation then
    assert(len(meta_hypes['model_list']) == 3)
    # inf_out['pred_boxes_new'], inf_out['pred_confidences']
    seg_softmax = decoded_logits['segmentation']['softmax'] # softmax in segmenation
    pred_boxes_new = decoded_logits['detection']['pred_boxes_new'] #bounding boxes in detection
    pred_confidences = decoded_logits['detection']['pred_confidences'] #confidence level in detection
    road_softmax = decoded_logits['road']['softmax'][0] #softmax in classification
    eval_list = [seg_softmax, pred_boxes_new, pred_confidences, road_softmax]
    
    #process the image by resizing
    def my_process(image):
        return process_image(subhypes, image)
    #calculate the evaluation run time
    if FLAGS.speed_test:
        eval_runtime(sess, subhypes, image_pl, eval_list, data_file)
        exit(0)

    test_constant_input(subhypes)
    test_segmentation_input(subhypes)

    import utils.train_utils as dec_utils
    #generate the output
    gen = _output_generator(sess, eval_list, image_pl, data_file, my_process)
    for image_file, output in gen:
        #read the image
        image = scp.misc.imread(image_file)
        #process the image by resizing it
        image = process_image(subhypes, image)
        #shape of an image
        shape = image.shape
        seg_softmax, pred_boxes_new, pred_confidences, road_softmax = output

        # Create Segmentation Overlay
        shape = image.shape
        seg_softmax = seg_softmax[:, 1].reshape(shape[0], shape[1])
        #if threshold value is greater than 0.5, it is classified as hard segmentation
        hard = seg_softmax > 0.5 
        overlay_image = utils.fast_overlay(image, hard)

        # Draw Detection Boxes
        new_img, rects = dec_utils.add_rectangles(
            subhypes['detection'], [overlay_image], pred_confidences,
            pred_boxes_new, show_removed=False,
            use_stitching=True, rnn_len=subhypes['detection']['rnn_len'],
            min_conf=0.50, tau=subhypes['detection']['tau'])

        # Draw road classification
        highway = (np.argmax(road_softmax) == 1)
        new_img = road_draw(new_img, highway)

        # Save image file
        im_name = os.path.basename(image_file)
        new_im_file = os.path.join(output_folder, im_name)
        im_name = os.path.basename(image_file)
        new_im_file = os.path.join(output_folder, im_name)
        scp.misc.imsave(new_im_file, new_img)

        logging.info("Plotting file: {}".format(new_im_file))

    eval_runtime(sess, subhypes, image_pl, eval_list, data_file)
    exit(0)


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
    meta_hypes = utils.load_hypes_from_logdir(logdir, subdir="",
                                              base_path='hypes')
    #for all the models in meta-hypes get the directory of output and input images
    for model in meta_hypes['models']:
        subhypes[model] = utils.load_hypes_from_logdir(logdir, subdir=model)
        hypes = subhypes[model]
        hypes['dirs']['output_dir'] = meta_hypes['dirs']['output_dir']
        hypes['dirs']['image_dir'] = meta_hypes['dirs']['image_dir']
        submodules[model] = utils.load_modules_from_logdir(logdir,
                                                           dirname=model,
                                                           postfix=model)

        modules = submodules[model]

    image_pl = tf.placeholder(tf.float32)
    #expand the shape of the array by inserting new axes in 0th positon
    image = tf.expand_dims(image_pl, 0)
    #set the shape of an array
    image.set_shape([1, 384, 1248, 3])
    decoded_logits = {}

    hypes = subhypes['segmentation']
    modules = submodules['segmentation']
    logits = modules['arch'].inference(hypes, image, train=False)
    #for all the models in hypes
    for model in meta_hypes['models']:
        hypes = subhypes[model]#get the model
        modules = submodules[model]
        # solver- max steps of iteration and batch size and etc
        optimizer = modules['solver']
        #This context manager validates that the given values are from the same graph, makes that graph the default graph, 
        #and pushes a name scope in that graph
        with tf.name_scope('Validation_%s' % model):
            reuse = {True: False, False: True}[first_iter]
            #Returns the current variable scope.
            scope = tf.get_variable_scope()
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
    utils.set_gpus_to_use()

    logdir = FLAGS.logdir
    data_file = FLAGS.data
     #if input is not given pass the error message
    if logdir is None:
        logging.error('Usage python predict_joint --logdir /path/to/logdir'
                      '--data /path/to/data/txt')
        exit(1)

    output_folder = os.path.join(logdir, res_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    logdir = logdir
    utils.load_plugins()
    #if data directory exist, join the path
    if 'TV_DIR_DATA' in os.environ:
        data_file = os.path.join(os.environ['TV_DIR_DATA'], data_file)
    else:
        #else create a directory DATA    
        data_file = os.path.join('DATA', data_file)
    
    if not os.path.exists(data_file):
        logging.error('Please provide a valid data_file.')
        logging.error('Use --data_file')
        exit(1)
    
    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'],
                                                 'UnitedVision2')
    logging_file = os.path.join(output_folder, "analysis.log") #log file
    utils.create_filewrite_handler(logging_file, mode='a')
    load_out = load_united_model(logdir)

    run_eval(load_out, output_folder, data_file)

    # stopping input Threads


if __name__ == '__main__':
    tf.app.run()
