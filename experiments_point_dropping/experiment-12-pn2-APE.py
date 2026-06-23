import import_base_path
import numpy as np
import torch
import random
from helpers.point_dropping_experiment_single import PointDropExperiment
from pointnet.dataset import ShapeNetDataset
from pointnet2.models import Pointnet2ClsMSG as Pointnet2
from grad_cam.grad_cam_pointnet2_APE import GradCamPointnet2
from helpers.args_helper import get_args
import time
from tqdm import tqdm
from eval import CausalMetric, CausalMetric_forme
from gradcam2_MSG import GradCAM2
from grad_cam.grad_cam_IHU_wrapper import IterativeHeatmappingAccum


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class GradcamWrapper:
    def __init__(self, grad_cam):
        self.grad_cam = grad_cam
        self.points = None

    def __call__(self, input, index=None):
        self.grad_cam(input, index)
        self.classifier_output = self.grad_cam.classifier_output
        self.points = self.grad_cam.xyz_aggregate.transpose(1, 2)
        return self.grad_cam.cam_aggregate

if __name__ == '__main__':
    args = get_args('pn2')

    torch.backends.cudnn.deterministic = True
    random.seed(1000)
    torch.manual_seed(1000)
    torch.cuda.manual_seed(1000)
    np.random.seed(1000)

    test_dataset = ShapeNetDataset(
        root=args.point_path,
        split='test',
        classification=True,
        npoints=args.num_points,
        data_augmentation=False,
        unique=False,)
        #unique_path='./unique-network-select')

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=args.shuffle)  # batch Size set to 1 obtain a single example

    # Grad Cam for pointnet by INPUT progressive point dropping
    net = Pointnet2(input_channels=0, num_classes=len(test_dataset.classes), use_xyz=True)
    net.cuda()
    net.load_state_dict(torch.load(args.model))
    net.eval()  # eval not needed at this stage unless grad cam is commented


    #grad_cam_wrapper = GradcamWrapper(grad_cam)

    # setup another classifier which jsut does classification without the additions for progressive-point dropping
    net_just_classification = Pointnet2(input_channels=0, num_classes=len(test_dataset.classes), use_xyz=True)

    if args.use_cuda:
        net_just_classification.cuda()
    net_just_classification.load_state_dict(torch.load(args.model))
    net_just_classification.eval()

    target_layer = '7'
    grad_cam = GradCAM2(net, target_layer, number_class=16)
    
    # grad_cam = GradCamPointnet2(args,
    #                     model=net,
    #                     module_n='1',
    #                     interp_mode=False,
    #                     progress_mode_input=True,
    #                     num_point_drops=1024,
    #                     progress_mode_input_plus_interpolate=True,
    #                     progress_mode_input_plus_interpolate_k=1,
    #                     progress_mode_input_old_droping_indecing_loop=True,
    #                     post_accumulation_normalization=False,
    #                     disable_relu=False)


    score = []
    for idx, (points, target) in tqdm(enumerate(testdataloader), total=len(testdataloader)):
    # if target.item() not in useful_cate:
    #     continue

        if target.item() == 0:
            a, b = 90, 135
        elif target.item() == 15:
            a, b = 30, 30
        elif target.item() == 9:
            a, b = 30, -45
        elif target.item() == 6:
            a, b = 0, 90
        elif target.item() == 5:
            a, b = 0, 75
        elif target.item() == 4:
            a, b = 30, -45
        else:
            a, b =0, 0
        


        # if not args.use_cpu:
        points, target = points.cuda(), target.cuda().squeeze(0)

        points = points.transpose(2, 1)
        B,C,N = points.shape
        cam = grad_cam(points, target)
        # cam = iterative_heatmapping(points, target)

        net.zero_grad()
        deletion = CausalMetric_forme(net_just_classification, 'ins', N//10)
        del_score = deletion.single_run(points, cam)

        if del_score is not None:
            score.append(auc(del_score))
        print(sum(score) / len(score))
        # print(sum(score) / len(score))




    # # test old method
    # start = time.time()
    # drop_experiment = PointDropExperiment(
    #     classifier=net_just_classification,
    #     grad_cam=grad_cam,
    #     testdataloader=testdataloader,
    #     num_drops=args.n_drops,
    #     steps=50,
    #     num_iterations=args.n_samples,
    #     update_cam=False,
    #     show_visualization=False,
    #     file_prefix='exp12-pn2-APE-',
    #     plot_title_prefix_text="EXP12 Point GradCAM APE PointNet++",
    #     create_png=True,
    #     random_drop=False,
    #     high_drop=True,
    #     low_drop=True,
    #     print_progress=True,
    #     use_cam_points=True,
    #     use_cuda=args.use_cuda,
    #     save_dropping_results=True,
    # ).run_experiment()
    # print("done in ", time.time() - start)

    # grad_cam.counterfactual = True
    # start = time.time()
    # drop_experiment = PointDropExperiment(
    #     classifier=net_just_classification,
    #     grad_cam=grad_cam_wrapper,
    #     testdataloader=testdataloader,
    #     num_drops=2048,
    #     steps=50,
    #     num_iterations=2874,  # max is 2874 = test set size
    #     update_cam=False,
    #     show_visualization=show_visualization,
    #     file_prefix='pn2-ppd-OLD-LOOP-exp2048-iterpolate_k1-001min_distance-COUNTERFACTUAL-FIX-',
    #     create_png=False,
    #     random_drop=True,
    #     high_drop=True,
    #     low_drop=True,
    #     print_progress=True,
    #     use_cam_points=True,
    #     alternative_colors=True,
    # ).run_experiment()
    # print("done in ", time.time() - start)

    # grad_cam.counterfactual = True
    # start = time.time()
    # drop_experiment = PointDropExperiment(
    #     classifier=net,
    #     grad_cam=grad_cam_wrapper,
    #     testdataloader=testdataloader,
    #     num_drops=2048,
    #     steps=125,
    #     num_iterations=2874,  # max is 2874 = test set size
    #     update_cam=False,
    #     show_visualization=show_visualization,
    #     file_prefix='pn2maxpoolprog-counterfactual-',
    #     alternative_colors=True,
    # ).run_experiment()
    # print("done in ", time.time() - start)