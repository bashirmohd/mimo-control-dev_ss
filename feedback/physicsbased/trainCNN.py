import argparse
import datetime
import json
import numpy as np
from scipy.stats import special_ortho_group
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'laserGym'))

import CNN
from laser_cbc.physics.model import CBCModel

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

# for creating validation set
from sklearn.model_selection import train_test_split

class CombiningDataGenCNN(Dataset):
    '''
    len:            number of samples in dataset
    dither_range_deg:      training phase range in degrees / 2
    double_frame:   use difference phase mapping to double pattern
    '''
    def __init__(self, **kwargs):
        self.dither_range_deg = kwargs.get('dither_range_deg', 45)
        self.double_frame = kwargs.get('double_frame', False)
        self.ortho_sampling = kwargs.get('ortho_sampling', False)
        self.len = kwargs.get('n_samples', 1000)
        self.model = CBCModel(**kwargs)
        self.rng = np.random.default_rng()
        x, y = self.gen_data_set()
        # self.x, self.y = self.gen_raw_data()

        # # reshape input for CNN model ( number of samples/batch_size, channel, height, width)
        if self.double_frame:
            x = x.reshape(self.len, 2, 5, 5)
        else:
            x = x.reshape(self.len, 1, 5, 5)
        #
        self.X = torch.from_numpy(x)
        self.Y = torch.from_numpy(y)

    def __getitem__(self, idx):
        'Generates one sample of data'
        return self.X[idx], self.Y[idx]
        # return self.x, self.y

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def gen_random_sample(self, i):
        '''
        Generate random samples in N dimensional phase space by phase
        perturbing the system at range [-1, 1] * self.dither_range_deg.

        Use random, or special orthogonal group for more efficient mapping:
        orthognoally sampling N times, then random rotate.
        see: https://en.wikipedia.org/wiki/Orthogonal_group#SO(n)
        '''
        if not self.double_frame:
            self.model.reset()  # perturb system from optimal
        pattern0 = self.model.nonzero_pattern
        if self.ortho_sampling:
            idx = i % self.model.n_beams
            if idx == 0:
                self.ortho_samples = special_ortho_group.rvs(self.model.n_beams)
            phs_perturb_deg = self.ortho_samples[idx] * self.dither_range_deg
        else:
            phs_perturb_deg = (self.rng.random(self.model.n_beams) * 2 - 1) * self.dither_range_deg
        pattern1 = self.model.perturb_phase_arr(phs_perturb_deg)
        pattern = np.concatenate((pattern0, pattern1)) if self.double_frame else pattern1
        return pattern, phs_perturb_deg

    # Z-score normalization
    def normalize(self, array):
        mean, std = np.mean(array, axis=0), np.std(array, axis=0)
        return (array - mean) / std

    # def MaxMinNormalize(self, array):
    #     min_value, max_value = np.min(array), np.max(array)
    #     scaled_inputs = (array - min_value) / (max_value - min_value)
    #     return scaled_inputs

    def gen_data_set(self):
        '''
        Generate training dataset as pair of x: pattern, y: phase error
        '''
        x_size = self.model.n_pattern_beams
        x_size = 2 * x_size if self.double_frame else x_size
        y_size = self.model.n_beams
        x_set = np.empty((self.len, x_size), dtype=np.float32)
        y_set = np.empty((self.len, y_size), dtype=np.float32)
        for i in range(self.len):
            x_set[i], y_set[i] = self.gen_random_sample(i)
        self.config = self.get_normalize_config(x_set, y_set)

        # # MaxMin normalization
        # x = self.MaxMinNormalize(x_set)
        # y = self.MaxMinNormalize(y_set)

        # Z-score normalization
        x = self.normalize(x_set)
        y = self.normalize(y_set)

        return x, y

    def gen_raw_data(self):
        '''
        Generate training dataset as pair of x: pattern, y: phase error
        '''
        x_size = self.model.n_pattern_beams
        x_size = 2 * x_size if self.double_frame else x_size
        y_size = self.model.n_beams
        x_set = np.empty((self.len, x_size), dtype=np.float32)
        y_set = np.empty((self.len, y_size), dtype=np.float32)
        for i in range(self.len):
            x_set[i], y_set[i] = self.gen_random_sample(i)

        return x_set, y_set

    def get_normalize_config(self, x, y):
        config = {}
        config['mu_X'] = np.mean(x, axis=0).ravel().tolist()
        config['mu_Y'] = np.mean(y, axis=0).ravel().tolist()
        config['sigma_X'] = np.std(x, axis=0).ravel().tolist()
        config['sigma_Y'] = np.std(y, axis=0).ravel().tolist()
        return config

def train_CNNmodel(model, dataset, device='cpu',
                patience=7, n_epochs=43, batch_size=80, verbose=False): #7 5 128
    model.to(device)
    df = pd.DataFrame(columns=['train_loss', 'valid_loss'])
    sigma_y = np.average(dataset.config['sigma_Y'])

    # divide data to training and test randomly for every epoch
    # train_x -> torch.Size([4500, 5, 5, 25])
    # train_y -> torch.Size([4500, 8])
    vali_num = int(0.1 * len(dataset))
    train_num = len(dataset) - vali_num
    train_dataset, vali_dataset = random_split(dataset, [train_num, vali_num])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=True)

    # define optimizer Adam, SGD with mommentum
    # optimizer = Adam(model.parameters(), lr=0.0148) #learning rate
    optimizer = SGD(model.parameters(), lr=0.02364, momentum=0.9, weight_decay=0, nesterov=False)

    # Evaluation tool MSE
    criterion = nn.MSELoss()

    # Schedule learning rate
    # scheduler = MultiStepLR( optimizer, milestones = [25, 40], gamma = 0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience) # "min" mode or "max" mode

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []

    # initialize the early_stopping object
    # early_stopping = EarlyStopping(patience=patience, verbose=verbose)

    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, (x, y) in enumerate(train_loader, 0):

            # get the inputs;
            output_train, target_train = model( x.float().to(device) ), y.float().to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # calculate the loss
            loss_train = torch.sqrt( criterion(output_train, target_train) ) * sigma_y
            # backward pass: compute gradient of the loss with respect to model parameters
            loss_train.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss_train.item())

        #####################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for (x, y) in valid_loader:

            # forward pass: compute predicted outputs by passing inputs to the model
            output_val, target_val = model( x.float().to(device) ), y.float().to(device)
            # calculate the loss
            loss_val = torch.sqrt( criterion(output_val, target_val) ) * sigma_y
            # record validation loss
            valid_losses.append(loss_val.item())

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        df.loc[epoch] = [train_loss, valid_loss]

        # print training loss, validation loss, and learning rate
        epoch_len = len(str(n_epochs))
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
              + f"train_loss: {train_loss:4.2f} deg, valid_loss: {valid_loss:4.2f} deg, lr: {curr_lr:4.5f}" )

        # adjust lr
        # scheduler.step() # for MultiStepLR
        scheduler.step( valid_loss) # for ReduceLROnPlateau

        # # clear lists to track next epoch
        train_losses = []
        valid_losses = []

    #     # early_stopping needs the validation loss to check if it has decreased,
    #     # and if it has, it will make a checkpoint of the current model
    #     # early_stopping(valid_loss, model)
    #     #
    #     # if early_stopping.early_stop:
    #     #     print('Early stopping')
    #     #     break

    # # load the last checkpoint with the best model
    # model.load_state_dict(torch.load('checkpoint.pt'))

    return model, df

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('-n', '--n_samples', type=int, default=5000, help='n_samples per beam')
    # Beam shape parameter.
    p.add_argument('-m', type=int, default=3, choices=[3, 9], help='MxM beam shape')
    p.add_argument('-e', '--n_epochs', type=int, default=43, help='number of episodes')
    p.add_argument('--weight', help='output NN weight file')
    p.add_argument('--force_cpu', action='store_true', help='Forces CPU usage')
    p.add_argument('--test_3_in_9', action='store_true')
    # Meaning of double_frame and ortho_sampling
    p.add_argument('--double_frame', action="store_true", help='doubled frame training')
    p.add_argument('--ortho_sampling', action="store_true", help='use random orthogonal sampling')
    p.add_argument("--rms_measure_noise", type=float, default=0.1, help="rms camera noise")
    # Do we need to change phs_drift_step_deg and dither_range_deg?
    p.add_argument("--phs_drift_step_deg", type=float, default=5, help="phs_drift_step_deg")
    p.add_argument("--dither_range_deg", type=int, default=30, help="dither phase range")
    p.add_argument("--net_config", type=str, default="3x3", help="config in recognizer.py")
    args = p.parse_args()
    device = 'cpu' if args.force_cpu else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = {
        'M': args.m,
        'verbose': args.verbose,
        'n_samples': args.n_samples,
        'test_3_in_9': args.test_3_in_9,
        'double_frame': args.double_frame,
        'ortho_sampling': args.ortho_sampling,
        'rms_measure_noise': args.rms_measure_noise,
        'phs_drift_step_deg': args.phs_drift_step_deg,
        'dither_range_deg': args.dither_range_deg,
        'net_config': args.net_config
    }
    
    json_file = json.dumps(config)
    fout = 'nn_trained/{}by{}_{}deg_config.json'.format(args.m, args.m, args.dither_range_deg)
    with open(fout, 'w') as jsonfile:
    	jsonfile.write(json_file)
    	print("Successfully Write")
    
    if args.weight is None:
        fout = 'nn_trained/{}by{}_{}deg.pth'.format(args.m, args.m, args.dither_range_deg)
    else:
        fout = args.weight
    print('Training model ' + fout + '...')
    
    # defining the model
    model = CNN.Net(args.double_frame, args.net_config)
    print(model)

    dataset = CombiningDataGenCNN(**config)
    # dataset = CombiningDataGenCNN(**config).gen_raw_data()

    # open a file
    # with open("directory/fileName.xxx", "xx (e.g., w+)") as ...,
    # "w+": open a file for reading and writing. If file doesn't exit, create a file.
    with open(fout.replace('pth', 'json'), 'w+') as f:
        json.dump(dataset.config, f, indent=4)

    print('Start Training...')
    start = datetime.datetime.now()

    net_trained, df = train_CNNmodel(
        model, dataset, device,
        n_epochs=args.n_epochs, verbose=args.verbose)

    end = datetime.datetime.now()

    # torch.save(net_trained.state_dict(), fout)
    torch.save(net_trained, fout)
    df.to_csv(fout.replace('pth', 'csv'), index=False)
    # save dataset
    torch.save(dataset, fout.replace('.pth', '_dat.pt'))

    print('Finished Training, wrote {}'.format(fout))
    elapsed_sec = round((end - start).total_seconds(), 2)
    print('It took', elapsed_sec, 'seconds to Train.')
