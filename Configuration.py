import torch
import os
import numpy as np
import cv2
from random import shuffle
from tqdm import tqdm


class Config:

    def __init__(self):
        # model parameters
        self.device = "cuda:0"
        self.n_epochs = 10
        self.batch_size = 1
        self.learning_rate = 1e-3
        self.mean = [0.38310535, 0.36254338, 0.33678384]
        self.std =  [0.27785318, 0.27457725, 0.26008851]
        self.class_weight = [11, 115, 382]
        self.loss_gamma = 5.
        self.regularization_coef = 1e-6
        self.changing_coef_class_weight = 1e-4
        self.n_class = 3
        self.num_workers = 1
        self.pretrained_weights = 'my_weights_5\\epoch_02.pt'
        self.initial_epoch = 1
        # input parameters
        self.input_image_h = 256
        self.input_image_w = 512
        self.input_image_c = 3
        # text files
        if os.path.exists('./lists/train.txt') and os.path.exists('./lists/val.txt')\
          and os.path.exists('./lists/test.txt'):
            self.train_dir = os.path.abspath("./lists/train.txt")
            self.val_dir = os.path.abspath("./lists/val.txt")
            self.test_dir = os.path.abspath("./lists/test.txt")
            with open(self.train_dir, "r") as f:
                num_train_images = len(f.readlines())
            with open(self.val_dir, "r") as f:
                num_val_images = len(f.readlines())
            with open(self.test_dir, "r") as f:
                num_test_images = len(f.readlines())
            self.num_images = {
                'train': num_train_images,
                'val': num_val_images,
                'test': num_test_images
            }
        # cnn encoder parameters
        self.encoder_kernel_size = 3
        self.encoder_droprate = 0.3
        self.encoder_filters_num_1 = 64
        self.encoder_filters_num_2 = 128
        self.encoder_filters_num_3 = 256
        self.encoder_filters_num_4 = 512
        # pool parameters
        self.pool_kernel_size = 2
        # rnn parameters
        self.rnn_time_steps = 5
        self.rnn_num_layers = 5
        self.rnn_hiden_filters = [512, 512, 512, 512, 512]
        # self.rnn_num_filters = 128
        self.rnn_shape = (int(self.input_image_h / 2 ** 4),
                          int(self.input_image_w / 2 ** 4))
        self.rnn_kernel_size = (3, 3)
        self.rnn_dropout = 0.3
        # cnn decoder parameters
        self.decoder_kernel_size = 3
        self.decoder_droprate = 0.3
        self.decoder_filters_num_1 = self.encoder_filters_num_4
        self.decoder_filters_num_2 = self.encoder_filters_num_3
        self.decoder_filters_num_3 = self.encoder_filters_num_2
        self.decoder_filters_num_4 = self.encoder_filters_num_1

    def computing_mean_std(self):
        _mean_absolut = np.array([0.0, 0.0, 0.0])
        _std_absolut = np.array([0.0, 0.0, 0.0])
        _len_absolut = self.num_images['train'] * self.rnn_time_steps
        _file_path = self.train_dir
 
        print('start: {}'.format(_file_path))
        with open(_file_path, 'r') as f:
            img_list = f.readlines()

        for line in tqdm(img_list):
            line = line.split()

            for img in line[:-1]:
                x_data = cv2.imread(img)
                x_data = x_data.astype("float32")
                x_data /= 255
                means = x_data.mean(axis=(0, 1), dtype='float64')
                stds = x_data.std(axis=(0, 1), dtype='float64')

                _mean_absolut += means
                _std_absolut += stds
            # print(line)

        _mean_absolut /= _len_absolut
        _std_absolut /= _len_absolut

        with open('./lists/mean_std.txt', 'w') as f:
            f.write('mean: {}\nstd: {}'
                    .format(_mean_absolut, _std_absolut))

    def computing_weights_coef(self):
        _weight_coef = np.array([0 for coef in range(self.n_class)], dtype=np.float64)
        _absolut_pixel = self.num_images['train'] * self.input_image_h *\
            self.input_image_w
        _file_path = self.train_dir
 
        print('start: {}'.format(_file_path))
        with open(_file_path, 'r') as f:
            img_list = f.readlines()

        for line in tqdm(img_list):
            line = line.split()
            x_data = cv2.imread(line[-1])[:,:,0]   
            for class_ in range(self.n_class):
                _weight_coef[class_] += x_data[x_data == class_].size
            
            # print(line)

        _weight_coef /= _absolut_pixel

        with open('./lists/weights_coef.txt', 'w') as f:
            f.write('{}'.format(_weight_coef))

    def create_train_val_test_lists(self):

        merged_list_file_path = './lists/merged_file.txt'
        img_list = []
        img_list_ = []

        with open(merged_list_file_path, 'r') as f:
            while True:
                lines = f.readline()
                if not lines:
                    break
                item = lines.strip().split()
                img_list.append(item)
        img_list.sort()
        
        for i in tqdm(range(len(img_list)-5)):

            buf = [img_list[i+j][0] for j in range(self.rnn_time_steps)]
            buf.append(img_list[i+self.rnn_time_steps-1][1])
            check = [img_list[i][0].split('/')[5], img_list[i][0].split('/')[-3], img_list[i][0].split('/')[-2]]
            check_flag = 0

            for buf_item in buf:
                if buf_item.split('/')[5] != check[0] or buf_item.split('/')[-3] != check[1] or buf_item.split('/')[-2] != check[2]:
                    check_flag += 1
            if check_flag:
                continue

            img_list_.append(' '.join(buf))
        
        shuffle(img_list_)

        for i in range(len(img_list_)):

            if i <= int(len(img_list_)*0.7):
                with open('./lists/train.txt', 'a') as f:
                    f.write(img_list_[i] + '\n')
            elif i <= int(len(img_list_)*0.9):
                with open('./lists/val.txt', 'a') as f:
                    f.write(img_list_[i] + '\n')
            else:
                with open('./lists/test.txt', 'a') as f:
                    f.write(img_list_[i] + '\n')


if __name__ == '__main__':
    # cfg = Config()
    # cfg.create_train_val_test_lists()
    cfg = Config()
    # cfg.computing_mean_std()
    cfg.computing_weights_coef()
