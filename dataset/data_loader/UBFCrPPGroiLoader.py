"""The dataloader for UBFC-rPPG dataset.

Details for the UBFC-rPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import dlib
from concurrent.futures import ThreadPoolExecutor

class UBFCrPPGroiLoader(BaseLoader):
    """The data loader for the UBFC-rPPG dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an UBFC-rPPG dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- subject1/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |   |-- subject2/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |...
                     |   |-- subjectn/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)
    
    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(
                os.path.join(data_dirs[i]['path'],"vid.avi"))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(
                os.path.join(data_dirs[i]['path'],"ground_truth.txt"))
            
        dlibFacePredictor = 'E:/code/pure/shape_predictor_81_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()  #使用dlib自带的frontal_face_detector作为我们的人脸提取器
        predictor = dlib.shape_predictor(dlibFacePredictor) 

        frames_clips, bvps_clips = self.roi_preprocess(frames, bvps, config_preprocess,detector,predictor)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def chunk(frames, bvps, chunk_length):
        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)
    
    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)

    @staticmethod
    def roi_preprocess(frames,bvps,config_preprocess,detector,predictor):
        # Check data transformation type
        data = list()  # Video data
        rois = list()  # ROI data

        for frame in frames:
            roi=UBFCrPPGroiLoader.get_landmark(frame, config_preprocess,detector, predictor,need='forehead')
            rois.append(roi)
        rois=np.array(rois)
        for data_type in config_preprocess.DATA_TYPE:
            # print(rois.shape)
            f_c = rois.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)  # concatenate all channels
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:  # chunk data into snippets
            frames_clips, bvps_clips = UBFCrPPGroiLoader.chunk(
                frames=data, bvps=bvps, chunk_length=config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips
    
    @staticmethod
    def get_landmark(frame,config_preprocess,detector,predictor,need):

        dets = detector(frame, 0) #使用detector进行人脸检测 dets为返回的结果

        for d in dets:
            shape = predictor(frame, d) #使用predictor进行人脸关键点识别
            landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])

            # Define regions based on landmarks
            if need == 'forehead':
                forehead_pts = landmarks[[18,19, 20, 23, 24,25,79,80,70,75], :]  # Forehead region
                forehead_mask = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillConvexPoly(forehead_mask, forehead_pts, (255, 255, 255))
                forehead_region = cv2.bitwise_and(frame, forehead_mask)
                forehead_bbox = cv2.boundingRect(forehead_pts)
                forehead_crop = forehead_region[forehead_bbox[1]:forehead_bbox[1]+forehead_bbox[3],
                                                forehead_bbox[0]:forehead_bbox[0]+forehead_bbox[2]]
                forehead_crop = cv2.resize(forehead_crop, (config_preprocess.RESIZE.W, config_preprocess.RESIZE.H))
                return forehead_crop

            if need == 'left_cheek':
                left_cheek_pts = landmarks[[1, 2, 3, 4, 48, 31, 28], :]  # Left cheek region
                left_cheek_mask = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillConvexPoly(left_cheek_mask, left_cheek_pts, (255, 255, 255))
                left_cheek_region = cv2.bitwise_and(frame, left_cheek_mask)
                left_cheek_bbox = cv2.boundingRect(left_cheek_pts)
                left_cheek_crop = left_cheek_region[left_cheek_bbox[1]:left_cheek_bbox[1]+left_cheek_bbox[3],
                                                    left_cheek_bbox[0]:left_cheek_bbox[0]+left_cheek_bbox[2]]
                left_cheek_crop = cv2.resize(left_cheek_crop, (config_preprocess.RESIZE.W, config_preprocess.RESIZE.H))
                return left_cheek_crop

            if need == 'right_cheek':
                right_cheek_pts = landmarks[[15, 14, 13, 12, 54, 35, 28], :]  # Right cheek region         
                right_cheek_mask = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillConvexPoly(right_cheek_mask, right_cheek_pts, (255, 255, 255))
                right_cheek_region = cv2.bitwise_and(frame, right_cheek_mask)
                right_cheek_bbox = cv2.boundingRect(right_cheek_pts)
                right_cheek_crop = right_cheek_region[right_cheek_bbox[1]:right_cheek_bbox[1]+right_cheek_bbox[3],
                                                    right_cheek_bbox[0]:right_cheek_bbox[0]+right_cheek_bbox[2]]
                right_cheek_crop = cv2.resize(right_cheek_crop, (config_preprocess.RESIZE.W, config_preprocess.RESIZE.H))
                return right_cheek_crop
