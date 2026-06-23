import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer_names, use_cuda=True, normalize=True):
        self.model = model
        self.model.eval()
        if use_cuda:
            self.model.cuda()
        self.cuda = use_cuda
        self.normalize = normalize
        self.activations = dict()
        self.gradients = dict()
        # self.target_layer = self.model.feat.feats[7]
        self.target_layer = self.model.features.feats[7]
        self.target_layer2 = self.model.features.feats[4]
        self.target_layer1 = self.model.features.feats[1]

        # self.target_layer = self.model.sa3.mlp_bns[2]
        # self.target_layer = self.model.feat.bn3
        def forward_hook(module, input, output):
            if torch.cuda.is_available():
                self.activations['value'] = output.cuda()
            else:
                self.activations['value'] = output
            return None
        
        def backward_hook(module, grad_input, grad_output):
            if torch.cuda.is_available():
                self.gradients['value'] = grad_output[0].cuda()
            else:
                self.gradients['value'] = grad_output
            return None
        
        def forward_hook1(module, input, output):
            if torch.cuda.is_available():
                self.activations['value1'] = output.cuda()
            else:
                self.activations['value1'] = output
            return None
        
        def backward_hook1(module, grad_input, grad_output):
            if torch.cuda.is_available():
                self.gradients['value1'] = grad_output[0].cuda()
            else:
                self.gradients['value1'] = grad_output
            return None

        def forward_hook2(module, input, output):
            if torch.cuda.is_available():
                self.activations['value2'] = output.cuda()
            else:
                self.activations['value2'] = output
            return None
        
        def backward_hook2(module, grad_input, grad_output):
            if torch.cuda.is_available():
                self.gradients['value2'] = grad_output[0].cuda()
            else:
                self.gradients['value2'] = grad_output
            return None

        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        self.target_layer1.register_forward_hook(forward_hook1)
        self.target_layer1.register_backward_hook(backward_hook1)
        self.target_layer2.register_forward_hook(forward_hook2)
        self.target_layer2.register_backward_hook(backward_hook2)


    def forward(self, input):
        return self.model(input)
    
    def __call__(self, input, index=None):
        if self.cuda:
            pred = self.model(input.cuda())
        else:
            pred = self.model(input)
        self.classifier_output = pred
        idx = torch.argmax(pred,dim=1)
        # 要写类别个数


        one_hot = F.one_hot(idx, 16)

        one_hot= one_hot.float().requires_grad_(True)


        self.model.zero_grad()
        pred.backward(gradient=one_hot, retain_graph=True)
        ######################## M Y !!!#####################
        # activation1 = self.activations['value1'].squeeze().detach().cpu().numpy()
        # activation2 = self.activations['value2'].squeeze().detach().cpu().numpy()
        # activation3 = self.activations['value'].squeeze().detach().cpu().numpy()

        # gradient1 = self.gradients['value1'].squeeze().detach().cpu().numpy()
        # gradient2 = self.gradients['value2'].squeeze().detach().cpu().numpy()
        # gradient3 = self.gradients['value'].squeeze().detach().cpu().numpy()

        # activations = np.concatenate([activation1, activation2, activation3], axis=0)
        # gradients = np.concatenate([gradient1, gradient2, gradient3], axis=0)
        

        #############################################




        activations = self.activations['value'].squeeze().detach().cpu().numpy()
        gradients = self.gradients['value'].squeeze().detach().cpu().numpy()  # 1024, 2048
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        ###########################
        # weights = np.mean(gradients, axis=1)
        # for i, w in enumerate(weights):
        #     cam += w * activations[i, :]

        ############################

        ############################
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
        #########XGrad-CAM##################
        # gradients = np.expand_dims(gradients, axis=0)

        # x_weights = np.sum(gradients[0,:] * activations, axis=1)
        # x_weights = x_weights / (np.sum(activations, axis=1) + 1e-6)
        # for i, w in enumerate(x_weights):
        #     cam += w * activations[i, :]
        ###################################
        ###########FBI#####################
        cam = np.sum(np.abs(activations), axis=0)
        ####################################
        
        cam = np.maximum(cam, 0)
        # cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cam - np.min(cam)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)


        return cam
        