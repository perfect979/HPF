import numpy as np
import torch
import torch.nn.functional as F
from models.pointnet2_utils import square_distance, index_points


class GradCAM2:
    def __init__(self, model, target_layer, use_cuda=True, normalize=True, number_class=40):
        self.model = model
        self.model.eval()
        if use_cuda:
            self.model.cuda()
        self.cuda = use_cuda
        self.normalize = normalize
        self.activations = dict()
        self.gradients = dict()
        self.target_layer1 = self.model.sa1.bn_blocks[2][2]
        self.target_layer2 = self.model.sa2.bn_blocks[2][2]
        self.target_layer3 = self.model.sa3.mlp_bns[2]
        self.num_cls = number_class


        def forward_hook1(module, input, output):
            if torch.cuda.is_available():
                self.activations['sa1'] = output.cuda()
            else:
                self.activations['sa1'] = output
            return None
        
        def backward_hook1(module, grad_input, grad_output):
            if torch.cuda.is_available():
                self.gradients['sa1'] = grad_output[0].cuda()
            else:
                self.gradients['sa1'] = grad_output
            return None
        
        def forward_hook2(module, input, output):
            if torch.cuda.is_available():
                self.activations['sa2'] = output.cuda()
            else:
                self.activations['sa2'] = output
            return None
        
        def backward_hook2(module, grad_input, grad_output):
            if torch.cuda.is_available():
                self.gradients['sa2'] = grad_output[0].cuda()
            else:
                self.gradients['sa2'] = grad_output
            return None
        
        def forward_hook3(module, input, output):
            if torch.cuda.is_available():
                self.activations['sa3'] = output.cuda()
            else:
                self.activations['sa3'] = output
            return None
        
        def backward_hook3(module, grad_input, grad_output):
            if torch.cuda.is_available():
                self.gradients['sa3'] = grad_output[0].cuda()
            else:
                self.gradients['sa3'] = grad_output
            return None
        
        self.target_layer1.register_backward_hook(backward_hook1)
        self.target_layer1.register_forward_hook(forward_hook1)
        self.target_layer2.register_forward_hook(forward_hook2)
        self.target_layer2.register_backward_hook(backward_hook2)
        self.target_layer3.register_backward_hook(backward_hook3)
        self.target_layer3.register_forward_hook(forward_hook3)
        

    def __call__(self, input):



        if self.cuda:
            pred, xyz1, xyz2, xyz3 = self.model(input.cuda())  # [1, 3, 512] [1, 3, 128] [1, 3, 1]
        else:
            pred, xyz1, xyz2, xyz3 = self.model(input)

        self.classifier_output = pred

        idx = torch.argmax(pred, dim=1)

        one_hot = F.one_hot(idx, self.num_cls)

        one_hot= one_hot.float().requires_grad_(True)

        self.model.zero_grad()
        pred.backward(gradient=one_hot, retain_graph=True)

        activation1 = self.activations['sa1']  # 1024, 128
        gradient1 = self.gradients['sa1']  # 1024, 128

        activation2 = self.activations['sa2']  # 1024, 128
        gradient2 = self.gradients['sa2']  # 1024, 128

        activation3 = self.activations['sa3']  # 1024, 128
        gradient3 = self.gradients['sa3']  # 1024, 128

        gradient3 = gradient3.reshape(1, 1024, 128)
        activation3 = activation3.reshape(1, 1024, 128)


        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        xyz0 = input.permute(0, 2, 1)
        B0, N0, C0 = xyz0.shape
        B2, N2, C1 = xyz2.shape
        B1, N1, C1 = xyz1.shape

        ###########将[1，1024，128，1]拼接到[1，256，64，128]####################


        activation2_ = torch.mean(activation2, dim=2)
        gradient2_ = torch.mean(gradient2, dim=2)

        cat_activation3_activation2_ = torch.cat([activation2_,activation3.squeeze(-1)], dim=1)  # [1, 1280, 128]
        cat_gradient3_gradient2_ = torch.cat([gradient2_, gradient3.squeeze(-1)], dim=1)

        ########################得到[1, 1280, 128]#############################

        ################将[1,1280,128]拼接到[1, 128,32,512]

        dists = square_distance(xyz1, xyz2)  # 512, 128
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        dists_weight = dist_recip / norm

        cat_activation3_activation2_ = cat_activation3_activation2_.permute(0, 2, 1)
        cat_gradient3_gradient2_ = cat_gradient3_gradient2_.permute(0, 2, 1)
        
        # [B, N, D]
        interpolated_points = torch.sum(index_points(cat_activation3_activation2_, idx) * dists_weight.view(B1, N1, 3, 1), dim=2)
        interpolated_grads = torch.sum(index_points(cat_gradient3_gradient2_, idx) * dists_weight.view(B1, N1, 3, 1), dim=2)

        activation1_ = torch.mean(activation1, dim=2).permute(0, 2, 1)  # [1,128,512]
        gradient1_ = torch.mean(gradient1, dim=2).permute(0, 2, 1)


        cat_activation3_activation2_activation1_ = torch.cat([  # [1, 512, 1408] 1024 + 256 + 128
            activation1_, interpolated_points
        ], dim=-1)
        cat_gradient3_gradient2_gradient1_ = torch.cat([
            gradient1_, interpolated_grads
        ], dim=-1)
        
        ##########################得到[1，512，1408]#############################################

        ##############将[1,512,1408]的特征和梯度传回原点数的点，通过距离传播给原点###################
        dists = square_distance(xyz0, xyz1)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        dists_weight = dist_recip / norm

        interpolated_activations = torch.sum(index_points(  # [1, 2048, 1408]
            cat_activation3_activation2_activation1_, idx) * dists_weight.view(B1, N0, 3, 1), dim=2)
        grads = torch.sum(index_points(  # [1, 2048, 1408]
            cat_gradient3_gradient2_gradient1_, idx) * dists_weight.view(B1, N0, 3, 1), dim=2)


        activations = interpolated_activations.permute(0, 2, 1).squeeze().detach().cpu().numpy()  # [1408, 2048]
        gradients = grads.permute(0, 2, 1).squeeze().detach().cpu().numpy()
        ####################计算完得激活图和梯度############################################



        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        #############Grad-CAM#######################
        # weights = np.mean(gradients, axis=1)
        # for i, w in enumerate(weights):
        #     cam += w * activations[i, :]
        ###########################################

        #############Grad-CAM++###############
        # gradients_2 = gradients ** 2
        # gradients_3 = gradients ** 3
        # alpha_numer = gradients_2
        # alpha_denom = 2 * gradients_2 + np.sum(activations * gradients_3, axis=(1), keepdims=True)
        # alpha = alpha_numer / alpha_denom
        # weights = np.sum(alpha * np.clip(gradients, 0, None), axis=(1), keepdims=True)
        # # for i, w in enumerate(weights):
        # #     cam += w * activations[i, :]
        # cam = activations * weights
        # cam = np.sum(cam, axis=0)
        #############################

        ###############XGrad-CAM############
        gradients = np.expand_dims(gradients, axis=0)

        x_weights = np.sum(gradients[0,:] * activations, axis=1)
        x_weights = x_weights / (np.sum(activations, axis=1)) + 1e-6
        for i, w in enumerate(x_weights):
            cam += w * activations[i, :]
        #####################################

        cam = np.maximum(cam, 0)
        
        cam = cam - np.min(cam)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return cam
