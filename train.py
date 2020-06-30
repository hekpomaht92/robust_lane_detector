import cv2
from PIL import Image
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import os
import random
from multiprocessing import set_start_method

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn import metrics

from utils import FocalLoss
from model import generate_model
import Configuration

np.random.seed(42)
set_start_method('spawn', True)
cfg = Configuration.Config()


class ImageGenerator(Dataset):

    def __init__(self, input_path, num_images):

        with open(input_path, 'r') as f:
            self.img_list = f.readlines()
        self.num_images = num_images
        self.transforms_color = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = []
        for i in range(5):
            data.append(torch.unsqueeze(self.transforms_color(Image.open(img_path_list.split()[i])), dim=0))
            if i == 4:
                raw_image = np.array(Image.open(img_path_list.split()[i]))
        data = torch.cat(data, 0)
        label = np.array(Image.open(img_path_list.split()[5]))
        label = torch.squeeze(torch.tensor(label))
        # print(np.unique(label.numpy()))
        sample = {'data': data, 'label': label, 'raw_image': raw_image}
        return sample


class TrainModel:

    def __init__(self, model):

        self.model = model
        self.model.to(cfg.device)
        # print(self.model)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.learning_rate,
        #                            momentum=0.9, weight_decay=cfg.regularization_coef, nesterov=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        # class_weight = torch.Tensor(cfg.class_weight)
        # self.criterion = nn.CrossEntropyLoss(weight=class_weight).to(cfg.device)
        self.criterion = FocalLoss(gamma=cfg.loss_gamma, alpha=cfg.class_weight)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3,
        #                                            gamma=cfg.learning_rate / cfg.n_epochs)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.8)
        self.writer = SummaryWriter('logs/{}'.format(str(cfg.class_weight)+'_'+str(cfg.learning_rate)))
        # visualize graph
        # data = torch.zeros((cfg.batch_size, cfg.rnn_time_steps, cfg.input_image_c,
        #                     cfg.input_image_h, cfg.input_image_w)).to(cfg.device)
        # self.writer.add_graph(self.model, data)
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(
            ImageGenerator(input_path=cfg.train_dir, num_images=cfg.num_images['train']),
            batch_size=cfg.batch_size, shuffle=True),
            'val': torch.utils.data.DataLoader(
            ImageGenerator(input_path=cfg.val_dir, num_images=cfg.num_images['val']),
            batch_size=cfg.batch_size, shuffle=True)
            }

        if not os.path.exists(os.path.join(os.getcwd(), 'weights')):
             os.makedirs(os.path.join(os.getcwd(), 'weights'))
    
    def write_logs(self, labels, pred, i, epoch, phase, loss):
        y_true = torch.squeeze(labels).to('cpu').numpy().reshape(-1)
        y_pred = torch.squeeze(pred).to('cpu').numpy().reshape(-1)
        out_metrics = metrics.classification_report(y_true, y_pred, digits=3, output_dict=True)
        print("Phase: {}, Epoch: {:02}, Iter: {:05}, Loss: {:.5f}, Accuracy: {:.3f}"
              .format(phase, epoch, i, loss, out_metrics['accuracy']))
        writer_step = (epoch - 1) * (cfg.num_images[phase] // 100) // cfg.batch_size + i // 100
        if i % 100 == 0:
            self.writer.add_scalar('{} loss'.format(phase),
                                    loss,
                                    writer_step)
            self.writer.add_scalar('{} accuracy'.format(phase),
                                    out_metrics['accuracy'],
                                    writer_step)
            self.writer.add_scalar('{} avg precision'.format(phase),
                                    out_metrics['macro avg']['precision'],
                                    writer_step)
            self.writer.add_scalar('{} avg recall'.format(phase),
                                    out_metrics['macro avg']['recall'],
                                    writer_step)
            self.writer.add_scalar('{} avg f1-score'.format(phase),
                                    out_metrics['macro avg']['f1-score'],
                                    writer_step)

            for i_class in range(cfg.n_class):
                self.writer.add_scalar('{} {} precision'.format(phase, i_class),
                                        out_metrics[str(i_class)]['precision'],
                                        writer_step)
                self.writer.add_scalar('{} {} recall'.format(phase, i_class),
                                        out_metrics[str(i_class)]['recall'],
                                        writer_step)
                self.writer.add_scalar('{} {} f1-score'.format(phase, i_class),
                                        out_metrics[str(i_class)]['f1-score'],
                                        writer_step)

    def train_epoch(self, epoch, phase='train'):
        self.model.train()
        # Iterate over data.
        for i, sample in enumerate(self.dataloaders[phase], 0):

            start = time.time()            
            inputs = sample['data'].to(cfg.device)
            labels = sample['label'].type(torch.LongTensor).to(cfg.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            pred = outputs.max(1, keepdim=True)[1]
            
            loss.backward()
            self.optimizer.step()
            self.write_logs(labels=labels, pred=pred, i=i, epoch=epoch, phase=phase, loss=loss.item())
            print(time.time() - start)

        torch.save(self.model.state_dict(), os.path.join('weights', 'epoch_{:02}.pt'.format(epoch)))

    def validation_epoch(self, epoch, phase='val'):
        self.model.eval()

        for i, sample in enumerate(self.dataloaders[phase], 0):
            
            inputs = sample['data'].to(cfg.device)
            labels = sample['label'].type(torch.LongTensor).to(cfg.device)
            self.optimizer.zero_grad()

            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                pred = outputs.max(1, keepdim=True)[1]

            self.write_logs(labels=labels, pred=pred, i=i, epoch=epoch, phase=phase, loss=loss.item())

    def train_model(self):
        for epoch in range(1, cfg.n_epochs + 1):

            # if epoch < cfg.initial_epoch:
            #     self.scheduler.step(epoch)
            #     continue
            
            print('Epoch {}/{}'.format(epoch, cfg.n_epochs - 1))
            print('-' * 10)

            self.train_epoch(epoch)
            self.validation_epoch(epoch)
            # self.scheduler.step()

    def test_model(self):

        def create_color_mask(img):
            img_out = np.zeros((cfg.input_image_h, cfg.input_image_w, cfg.input_image_c), dtype=np.uint8)
            for x in range(cfg.input_image_h):
                for y in range(cfg.input_image_w):
                    if img[x][y] == 0:
                        img_out[x][y] = [0, 0, 0]
                    elif img[x][y] == 1:
                        img_out[x][y] = [0, 255, 0]
                    elif img[x][y] == 2:
                        img_out[x][y] = [0, 0, 255]
            return img_out

        test_dataloader = torch.utils.data.DataLoader(
            ImageGenerator(input_path=cfg.test_dir, num_images=cfg.num_images['test']))
        self.model.eval()

        for i, sample in enumerate(test_dataloader, 0):
                    
            inputs = sample['data'].to(cfg.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                pred = torch.squeeze(outputs.max(1, keepdim=True)[1])

            pred = create_color_mask(pred.to('cpu').numpy())
            raw = sample['raw_image'].numpy()[0]
            out = cv2.add(pred, raw)

            if not os.path.exists(os.path.join(os.getcwd(), 'out', cfg.pretrained_weights)):
                os.makedirs(os.path.join(os.getcwd(), 'out', cfg.pretrained_weights))
            dir_out = os.path.join(os.getcwd(), 'out', cfg.pretrained_weights, '{}.png'.format(i))
            plt.imshow(out)
            plt.savefig(dir_out)


            # plt.imshow(out)
            # plt.show()
            # plt.figure(1)
            # plt.imshow(pred.to('cpu').numpy())
            # plt.figure(2)
            # plt.imshow(torch.squeeze(sample['label']).to('cpu').numpy())
            # plt.show()



            print("Number: {}".format(i))
    

if __name__ == '__main__':
    model = generate_model()
    # print(model)
    trainer = TrainModel(model)
    trainer.train_model()
    # trainer.test_model()
    print('complite')
