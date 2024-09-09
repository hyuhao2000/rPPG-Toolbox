import glob
import json
import os
import re

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import dlib
import itertools

class PUREroiLoader(BaseLoader):
    """The data loader for the PURE dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an PURE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 01-01/
                     |      |-- 01-01/
                     |      |-- 01-01.json
                     |   |-- 01-02/
                     |      |-- 01-02/
                     |      |-- 01-02.json
                     |...
                     |   |-- ii-jj/
                     |      |-- ii-jj/
                     |      |-- ii-jj.json
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For PURE dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*-*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('-', '')
            index = int(subject_trail_val)
            subject = int(subject_trail_val[0:2])
            dirs.append({"index": index, "path": data_dir, "subject": subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = list(data_info.keys())  # all subjects by number ID (1-27)
        subj_list = sorted(subj_list)
        num_subjs = len(subj_list)  # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        # compile file list
        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files  # add file information to file_list (tuple of fname, subj ID, trial num,
            # chunk num)

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(
                os.path.join(data_dirs[i]['path'], filename, ""))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'], filename, '*.npz')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(
                os.path.join(data_dirs[i]['path'], "{0}.json".format(filename)))

        dlibFacePredictor = 'F:/code/rPPG-Toolbox/shape_predictor_81_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()  #使用dlib自带的frontal_face_detector作为我们的人脸提取器
        predictor = dlib.shape_predictor(dlibFacePredictor) 

        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        
        frames_clips, bvps_clips = self.roi_preprocess(frames, bvps, config_preprocess,detector,predictor)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def chunk(frames, bvps, chunk_length):
        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)
    
    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        all_png = sorted(glob.glob(video_file + '*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            # cv2.imwrite('E:/code/rPPG-Toolbox/test.jpg', img)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["waveform"]
                     for label in labels["/FullPackage"]]
        return np.asarray(waves)
    
    @staticmethod
    def roi_preprocess(frames,bvps,config_preprocess,detector,predictor):
        # Check data transformation type
        data = list()  # Video data
        mst_maps = list()  # ROI data

        for frame in frames:
            mst_map=PUREroiLoader.get_landmark(frame, config_preprocess,detector, predictor)
           
            if len(mst_map)==0:  
                continue
            
            mst_maps.append(mst_map)

        mst_maps=np.array(mst_maps)

        for data_type in config_preprocess.DATA_TYPE:
            f_c = mst_maps.copy()
            # print(f_c[0])
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
            frames_clips, bvps_clips = PUREroiLoader.chunk(
                frames=data, bvps=bvps, chunk_length=config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips
    
    @staticmethod
    def get_landmark(frame,config_preprocess,detector,predictor):

            # print(region)
        def create_mask_and_extract_region(frame, pts):
            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, (255, 255, 255))
            region = cv2.bitwise_and(frame, mask)
            bbox = cv2.boundingRect(pts)
            crop = region[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            return region, crop

        dets = detector(frame, 0) # 使用detector进行人脸检测 dets为返回的结果
        
        if len(dets) == 0:
            print('No face detected in')
            return []
        
        else:
            crops = []
            for d in dets:
                shape = predictor(frame, d) # 使用predictor进行人脸关键点识别
                landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])

                regions = {
                    "forehead": landmarks[[18, 19, 20, 23, 24, 25, 79, 80, 70, 75], :],
                    "left_up_cheek": landmarks[[1, 2, 3, 31, 28], :],
                    "left_down_cheek": landmarks[[3, 4, 5, 48, 31], :],
                    "right_up_cheek": landmarks[[13, 14, 15, 28, 35], :],
                    "right_down_cheek": landmarks[[11, 12, 13, 35, 54], :],
                    "jaw": landmarks[[5, 6, 7, 8, 9, 10, 11, 54, 55, 56, 57, 58, 59, 60, 48], :]
                }

                for region in ['forehead','left_up_cheek','left_down_cheek','right_up_cheek','right_down_cheek','jaw']:
                    pts = regions[region]
                    region_img, crop = create_mask_and_extract_region(frame, pts)
                    if crop.size == 0:
                        print('Error:', region)
                        continue
                    else:
                        b,g,r = cv2.split(crop)
                        #去除0值后取平均值
                        b = b[b>0].mean()
                        g = g[g>0].mean()
                        r = r[r>0].mean()
                        crops.append([b,g,r])
                        # Y,U,V = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2YUV))
                        # Y = Y[Y>0].mean()
                        # U = U[U>0].mean()
                        # V = V[V>0].mean()

                        # crops.append([b,g,r,Y,U,V])
            #if len(crops)<6 repeat the last roi for 6-n times
            while len(crops)<6:
                crops.append(crops[-1]) 
            mst_map = PUREroiLoader.generate_combinations(np.array(crops))

        return mst_map

    @staticmethod
    def generate_combinations(array):
        """
        对输入形状为 (n, C) 的数组的第二维度进行排列组合，生成所有非空子集的组合，
        并对每个组合的第三维的对应值取平均。
        
        参数:
        array (numpy.ndarray): 输入的三维数组，形状为 (n, C)
        
        返回:
        list: 包含所有组合平均后的数组列表，每个数组的形状为 (2^n-1, C)
        """
        n, C = array.shape

        # 生成所有非空子集的索引组合
        indices = range(n)
        all_combinations = []
        for r in range(1, n + 1):
            combinations = list(itertools.combinations(indices, r))
            all_combinations.extend(combinations)

        # print(all_combinations)
        # 计算每个组合的平均值
        all_combinations_avg = []
        for combination in all_combinations:
            combination_avg = array[combination, :]
            combination_avg = np.mean(combination_avg, axis=0)
            # print(combination_avg.shape)
            all_combinations_avg.append(combination_avg)

        all_combinations_avg = np.array(all_combinations_avg)
        #numpy add a dimension
        all_combinations_avg = np.expand_dims(all_combinations_avg, axis=0)
        # print(all_combinations_avg.shape)
        return all_combinations_avg