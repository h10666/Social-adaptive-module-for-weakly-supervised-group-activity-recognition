import glob
import os
import numpy as np
import random
import utils
import csv
import shutil


def video_statistics(dataset_root: str):
    count_array = np.zeros(120)
    sum_clips = 0
    for root, dirs, files in utils.lwalk(dataset_root, max_level=2):
        if root is not dataset_root:
            if 0 < len(dirs) < 10:
                print('error', root, len(dirs))
                print(('move', root, 'to', '../error/' + os.path.split(root)[-1]))
                shutil.move(root, '../error/' + os.path.split(root)[-1])
            else:
                count_array[len(dirs)] += 1
                sum_clips += len(dirs)
    print(count_array, sum(count_array), sum_clips)
    return sum_clips

def get_folders(root):
    dir_list = os.listdir(root)
    folder_list = []
    for dir in dir_list:
        if not os.path.isfile(os.path.join(root, dir)):
            folder_list.append(dir)
    return folder_list

def sample_statistics(dataset_root: str, video_ids: list=None):
    label_dict = {}
    for root, dirs, files in utils.lwalk(dataset_root, max_level=2):
        txt_file_list = glob.glob(root + '/annotation.txt')
        if txt_file_list:
            if video_ids and (os.path.split(root)[-1] not in video_ids):
                continue
            txt_file = txt_file_list[0]  # only one txt file in each video folder.
            with open(txt_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    clip_id, activity = line.split('\n')[0].split('\t')
                    if activity not in label_dict.keys():
                        label_dict[activity] = 1
                    else:
                        label_dict[activity] += 1

    print(label_dict)
    # print('keys', label_dict.keys())
    key_list = label_dict.keys()
    # print('sorted keys', sorted(key_list))
    for k in sorted(key_list):
        print(label_dict[k], sep='\t', end='\t')



def get_video_ids(dataset_root: str):
    videos_ids = []
    for root, dirs, files in utils.lwalk(dataset_root, max_level=2):
        if root is not dataset_root:
            if 0 < len(dirs) < 10:
                print('error', root, len(dirs))
            else:
                videos_ids.append(os.path.split(root)[-1])
    return videos_ids

def read_video_ids(txt_file: str):
    video_ids = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            video_ids.extend(list(line.split('\n')[0].split(',')))
    return video_ids

def list_formate(List, width=10):
    output_str = ''
    for idx, item in enumerate(List):
        output_str += str(item) + ','
    return output_str


def train_test_split(dataset_root: str):
    # get video ids.
    video_ids = get_video_ids(dataset_root)

    # get train-test splits of videos.
    total_samples = video_statistics(dataset_root)
    random.shuffle(video_ids)
    test_sample_count = 0
    for idx, video_id in enumerate(video_ids):
        if test_sample_count < (total_samples / 6):
            test_sample_count += len(get_folders(os.path.join(dataset_root, video_id)))
            print(test_sample_count)
        else:
            break
    test_split, train_split = list(map(int, video_ids[:idx])), list(map(int, video_ids[idx:]))

    # print('train_split: ', len(train_split), train_split, '\n test_split: ', len(test_split), test_split)
    utils.write_txt('train_videos', list_formate(train_split), 'w')
    utils.write_txt('test_videos', list_formate(test_split), 'w')
    # utils.write_txt('train_videos.txt', str(train_split), 'w')
    # utils.write_txt('test_videos.txt', str(train_split), 'w')
    # generate train-test splits of video clips.


def csv2txt(dataset_root: str):
    for root, dirs, files in utils.lwalk(dataset_root, max_level=2):
        csv_file_list = glob.glob(root + '/*.csv')
        if csv_file_list:
            print(csv_file_list)
            csv_file = csv_file_list[0]  # only one file in each video folder.
            txt_file = os.path.join(os.path.split(csv_file)[0], 'annotation.txt')
            print(txt_file)
            with open(txt_file, "w") as output_file:
                with open(csv_file, "r") as input_file:
                    [output_file.write("\t".join(row) + '\n') for row in csv.reader(input_file)]
                output_file.close()


if __name__ == '__main__':
    dataset_root = '../frames/v2_1'
    #video_statistics(dataset_root)
    #csv2txt(dataset_root)

#     train_test_split(dataset_root)



    #sample_statistics(dataset_root, read_video_ids('train_videos.txt'))
    #sample_statistics(dataset_root, read_video_ids('test_videos.txt'))
