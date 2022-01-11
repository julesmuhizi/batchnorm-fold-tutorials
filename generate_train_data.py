"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################

import os
import glob
import sys
import time
########################################################################
import logging
########################################################################
# import additional python-library
########################################################################
import numpy
# from import
from tqdm import tqdm
# original lib
import common as com
########################################################################


########################################################################
# visualizer
########################################################################


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         downsample=False,
                         dims = 640):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    # dims = n_mels * frames
    dims = dims

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power,
                                                downsample=downsample,
                                                input_dim=dims)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    print("Shape of dataset: {}".format(dataset.shape))
    return dataset


def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files
    return :
        train_files : list [ str ]
            file list for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files
########################################################################
if __name__ == "__main__":                                        
                                        
    args = com.command_line_chk()

    # load parameter.yaml 
    param = com.yaml_load(args.config)
    param = param["train"]

    # load base_directory list
    dirs = com.select_dirs(param=param)
    print(dirs)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        # generate dataset
        train_data_save_load_directory = "./train_time_data/train_data_inputs_{}_frames_{}_hops_{}_fft_{}_mels_{}_power_{}_downsample_{}.npy".format(
        param["model"]["input_dim"],param["feature"]["frames"], param["feature"]["hop_length"], 
            param["feature"]["n_fft"], param["feature"]["n_mels"], param["feature"]["power"],param["feature"]["downsample"],)
        
        # if train_data available, load processed data in local directory without reprocessing wav files --saves time--
        if os.path.exists(train_data_save_load_directory):
            print("Loading train_data from {}".format(train_data_save_load_directory))
            
            train_data = numpy.load(train_data_save_load_directory)
        else:
            print("============== DATASET_GENERATOR ==============")
            files = file_list_generator(target_dir)
            train_data = list_to_vector_array(files,
                                              msg="generate train_dataset",
                                              n_mels=param["feature"]["n_mels"],
                                              frames=param["feature"]["frames"],
                                              n_fft=param["feature"]["n_fft"],
                                              hop_length=param["feature"]["hop_length"],
                                              power=param["feature"]["power"],
                                              downsample=param["feature"]["downsample"],
                                              dims = param["model"]["input_dim"])
            #save train_data
            if not os.path.exists('train_time_data'):
                os.makedirs('./train_time_data')
            numpy.save(train_data_save_load_directory, train_data)
            print("Train data saved to {}".format(train_data_save_load_directory))