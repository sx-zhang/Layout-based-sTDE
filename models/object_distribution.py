import torch
import scipy.io as scio
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence
import copy
import numpy as np
import torch.nn as nn


class Object_Distribution(torch.nn.Module):

    def __init__(self, dataset='AI2THOR'):
        '''
        :param prior_alpha: class_num*(class_num-1)
        '''
        super(Object_Distribution, self).__init__()
        self.dataset = dataset
        self.init_prior_alpha = self.upload_prior_alpha()
        self.class_num = self.init_prior_alpha.shape[1]
        self.prior_alpha = None
        self.posterior_alpha = None
        self.feature_memory = []
        self.object_memory = []
        self.prob_matrix = None
        self.kl = []
        self.ii_pro = torch.ones(1)
        # self.base_scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.similarity_scale = nn.Parameter(torch.tensor(6.0), requires_grad=True)
        self.update_scale = nn.Sequential(
            nn.Linear(22, 22),
            nn.ReLU(),
            nn.Linear(22, 22),
            nn.ReLU(),
        )

    def upload_prior_alpha(self):
        init_prior_alpha = []
        if self.dataset == 'AI2THOR':
            scenes = ['Kitchens', 'Living_Rooms', 'Bedrooms', 'Bathrooms']
        elif self.dataset == 'MP3D' or 'RoboTHOR':
            scenes = ['scenes']
        data = scio.loadmat('prior_distribution/dis_param_{}_v2.mat'.format(self.dataset))
        for s in scenes:
            init_prior_alpha.append(data[s])
        return torch.tensor(init_prior_alpha, dtype=torch.float32)

    def get_scene_id(self, current_scene):
        id_number = "".join(filter(str.isdigit, current_scene))
        pass
        if 0 < int(id_number) < 200:
            idx = 0
        elif 200 < int(id_number) < 300:
            idx = 1
        elif 300 < int(id_number) < 400:
            idx = 2
        elif 400 < int(id_number) < 500:
            idx = 3
        return idx

    def observation_memory_update(self, observation_t, current_scene, target_object):
        '''
        observation_t['info']:bounding box + probability (22, 5)
        observation_t['indicator']:target object (22, 1)
        observation_t['appear']:region features (22, 512)
        '''

        if self.dataset == 'AI2THOR':
            scene_id = self.get_scene_id(current_scene)
        elif self.dataset == 'MP3D' or 'RoboTHOR':
            scene_id = 0

        if len(self.object_memory) == 0:  # init the pram
            device = observation_t['indicator'].device
            self.prior_alpha = copy.deepcopy(self.init_prior_alpha[scene_id, :, :]).to(device)
            self.posterior_alpha = copy.deepcopy(self.prior_alpha)
            self.pro_matrix_update()
            self.kl.append(torch.tensor([0.0]).to(device))

        object_t = torch.sign(observation_t['info'][:, -1])
        similarity_flag = False
        if object_t.sum(-1) > 1:  # the objects in view should greater than 1
            if len(self.object_memory) == 0:
                similarity_flag = False
            else:
                for inx, val in enumerate(self.object_memory):
                    feature_similarity = (val-object_t).pow(2).sum(0)
                    if feature_similarity == 0 \
                            and (self.feature_memory[inx]-observation_t['appear']).pow(2).sum(0).sum(0) < self.similarity_scale:
                        similarity_flag = True
                        break
            if not similarity_flag:
                self.prior_alpha.detach()
                self.posterior_alpha.detach()
                self.feature_memory.append(observation_t['appear'])
                self.object_memory.append(object_t)
                update_scale = self.update_scale(object_t+target_object.squeeze())
                update_scale = update_scale.mul(10.0)
                self.posterior_update(update_scale)
                self.compute_kl_and_matrix()
                self.pro_matrix_update()
                self.prior_update()
            else:
                self.kl.append(self.kl[-1] * 0.8)
        else:
            self.kl.append(self.kl[-1] * 0.8)

    def reset_memory(self):
        self.feature_memory.clear()
        self.object_memory.clear()
        self.kl.clear()

    def posterior_update(self, update_scale):
        observed_objects = self.object_memory[-1]
        posterior_alpha = None
        for i in range(self.class_num):
            if observed_objects[i] == 0:
                update_alpha = self.prior_alpha[i, :].unsqueeze(0)
            else:
                update_alpha = self.prior_alpha[i, :] + self.del_tensor(torch.mul(observed_objects, update_scale), i)
                update_alpha = update_alpha.unsqueeze(0)

            posterior_alpha = self.accumulate_posterior_alpha(posterior_alpha, update_alpha)
        self.posterior_alpha = posterior_alpha

    def accumulate_posterior_alpha(self, posterior_alpha, update_alpha):
        if posterior_alpha is None:
            posterior_alpha = update_alpha
        else:
            posterior_alpha = torch.cat((posterior_alpha, update_alpha), dim=0)
        return posterior_alpha

    def prior_update(self):
        self.prior_alpha = copy.deepcopy(self.posterior_alpha.detach())

    def del_tensor(self, arr, ind):
        arr_1 = arr[0: ind]
        arr_2 = arr[ind+1:]
        return torch.cat((arr_1, arr_2), dim=0)

    def expand_tensor(self, arr, ind):
        arr_1 = arr[0:ind]
        arr_2 = arr[ind:]
        expand_tensor = torch.cat((arr_1, torch.tensor([1.0]).to(arr_1.device), arr_2), dim=0)
        return expand_tensor.unsqueeze(0)

    def pro_matrix_update(self):
        post_distribution = Dirichlet(self.posterior_alpha)
        post_pro_matrix = post_distribution.mean
        pro_matrix = None
        for i in range(self.class_num):
            if pro_matrix is None:
                pro_matrix = self.expand_tensor(post_pro_matrix[i], i)
            else:
                pro_matrix = torch.cat((pro_matrix, self.expand_tensor(post_pro_matrix[i], i)), dim=0)

        self.prob_matrix = pro_matrix

    def compute_kl_and_matrix(self):
        pr_distribution = Dirichlet(self.prior_alpha)
        post_distribution = Dirichlet(self.posterior_alpha)
        kl = kl_divergence(post_distribution, pr_distribution).sum()
        self.kl.append(kl)




