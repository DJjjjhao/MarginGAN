# Thank the authors of pytorch-generative-model-collections and examples of pytorch.
# The github address is https://github.com/znxlwm/pytorch-generative-model-collections
# and https://github.com/pytorch/examples/blob/master/mnist/main.py respectively.
# Our code is widely adapted from their repositories.

import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import dataloader


def nll_loss_neg(y_pred, y_true):
    out = torch.sum(y_true * y_pred, dim=1)
    return torch.mean(- torch.log((1 - out) + 1e-6))


# batch_size * input_dim => batch_size * output_dim * input_size * input_size
class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x


# batch_size * input_dim * input_size * input_size => batch_size * output_dim
class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


# batch_size * 1 * 28 * 28 => batch_size * 10
class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        utils.initialize_weights(self)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x), F.log_softmax(x, dim=1)


class MarginGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = True
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        self.num_labels = args.num_labels
        self.index = args.index
        self.lrC = args.lrC

        # load dataset
        self.labeled_loader, self.unlabeled_loader, self.test_loader = dataloader(self.dataset, self.input_size, self.batch_size, self.num_labels)

        data = self.labeled_loader.__iter__().__next__()[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.C = classifier()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.C_optimizer = optim.SGD(self.C.parameters(), lr=args.lrC, momentum=args.momentum)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.C.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        utils.print_network(self.C)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['C_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.best_acc = 0
        self.best_time = 0

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()

            if epoch == 0:
                correct_rate = 0
                while True:
                    for iter, (x_, y_) in enumerate(self.labeled_loader):
                        if self.gpu_mode:
                            x_, y_ = x_.cuda(), y_.cuda()
                        self.C.train()
                        self.C_optimizer.zero_grad()
                        _, C_real = self.C(x_)
                        C_real_loss = F.nll_loss(C_real, y_)
                        C_real_loss.backward()
                        self.C_optimizer.step()

                        if iter == self.labeled_loader.dataset.__len__() // self.batch_size:
                            self.C.eval()
                            test_loss = 0
                            correct = 0
                            with torch.no_grad():
                                for data, target in self.test_loader:
                                    data, target = data.cuda(), target.cuda()
                                    _, output = self.C(data)
                                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                                    correct += pred.eq(target.view_as(pred)).sum().item()
                            test_loss /= len(self.test_loader.dataset)

                            print('\niter: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                                (iter), test_loss, correct, len(self.test_loader.dataset),
                                100. * correct / len(self.test_loader.dataset)
                                ))
                            correct_rate = correct / len(self.test_loader.dataset)
                    gate = 0.8
                    if self.num_labels == 600:
                        gate = 0.93
                    elif self.num_labels == 1000:
                        gate = 0.95
                    elif self.num_labels == 3000:
                        gate = 0.97
                    if correct_rate > gate:
                        break

            correct_wei = 0
            number = 0
            labeled_iter = self.labeled_loader.__iter__()
            # print(self.labeled_loader.dataset.__len__())
            for iter, (x_u, y_u) in enumerate(self.unlabeled_loader):
                self.C.train()
                if iter == self.unlabeled_loader.dataset.__len__() // self.batch_size:
                    if epoch > 0:
                        print('\nPseudo tag: Accuracy: {}/{} ({:.0f}%)\n'.format(
                            correct_wei, number,
                            100. * correct_wei / number))
                    break

                try:
                    x_l, y_l = labeled_iter.__next__()
                except StopIteration:
                    labeled_iter = self.labeled_loader.__iter__()
                    x_l, y_l = labeled_iter.__next__()

                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_l, y_l, x_u, y_u, z_ = \
                        x_l.cuda(), y_l.cuda(), x_u.cuda(), y_u.cuda(), z_.cuda()

                # update C network
                self.C_optimizer.zero_grad()

                _, C_labeled_pred = self.C(x_l)
                C_labeled_loss = F.nll_loss(C_labeled_pred, y_l)

                _, C_unlabeled_pred = self.C(x_u)
                C_unlabeled_wei = torch.max(C_unlabeled_pred, 1)[1]

                correct_wei += C_unlabeled_wei.eq(y_u).sum().item()
                number += len(y_u)
                C_unlabeled_loss = F.nll_loss(C_unlabeled_pred,  C_unlabeled_wei)

                G_ = self.G(z_)
                C_fake_pred, _ = self.C(G_)
                C_fake_wei = torch.max(C_fake_pred, 1)[1]
                C_fake_wei = C_fake_wei.view(-1, 1)
                C_fake_wei = torch.zeros(self.batch_size, 10).cuda().scatter_(1, C_fake_wei, 1)
                C_fake_loss = nll_loss_neg(C_fake_wei, C_fake_pred)

                C_loss = C_labeled_loss + C_unlabeled_loss + C_fake_loss
                self.train_hist['C_loss'].append(C_loss.item())

                C_loss.backward()
                self.C_optimizer.step()

                # update D network
                self.D_optimizer.zero_grad()

                D_labeled = self.D(x_l)
                D_labeled_loss = self.BCE_loss(D_labeled, torch.ones_like(D_labeled))

                D_unlabeled = self.D(x_u)
                D_unlabeled_loss = self.BCE_loss(D_unlabeled, torch.ones_like(D_unlabeled))

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_labeled_loss + D_unlabeled_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)

                D_fake = self.D(G_)
                G_loss_D = self.BCE_loss(D_fake, self.y_real_)

                _, C_fake_pred = self.C(G_)
                C_fake_wei = torch.max(C_fake_pred, 1)[1]
                G_loss_C  = F.nll_loss(C_fake_pred, C_fake_wei)

                G_loss = 0.9*G_loss_D + 0.1*G_loss_C
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss_D.backward(retain_graph=True)
                G_loss_C.backward()

                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, C_loss: %.8f" %
                          (
                          (epoch + 1), (iter + 1), self.unlabeled_loader.dataset.__len__() // self.batch_size, D_loss.item(),
                          G_loss.item(), C_loss.item()))

            self.C.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.cuda(), target.cuda()
                    _, output = self.C(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.test_loader.dataset)

            acc = 100. * correct / len(self.test_loader.dataset)
            cur_time = time.time() - start_time
            with open('acc_time/LUG/' + str(self.num_labels) + '_' + str(self.index) + '_' + str(self.lrC) + '.txt', 'a') as f:
                f.write(str(cur_time) + ' ' + str(acc) + '\n')

            if acc > self.best_acc:
                self.best_acc = acc
                self.best_time = cur_time

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch + 1))

        with open('acc_time/LUG/' + str(self.num_labels) + '_' + str(self.lrC) + '_best.txt', 'a') as f:
            f.write(str(self.index) + ' ' + str(self.best_time) + ' ' + str(self.best_acc) + '\n')
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch,
                                                                        self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()


    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        path = self.result_dir + '/LUG/' + self.model_name + '_' + str(self.index)

        if not os.path.exists(path):
            os.makedirs(path)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          path + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):

        save_dir = self.save_dir + '/LUG/' + self.model_name + '_' + str(self.index)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))