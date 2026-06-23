import import_base_path
import numpy as np
import torch
import random
from helpers.point_dropping_experiment_single import PointDropExperiment
from pointnet.dataset import ShapeNetDataset
from pointnet2.models import Pointnet2ClsMSG as Pointnet2
from helpers.args_helper import get_args
import time
from grad_cam.grad_cam_pointnet2 import GradCamThirdModule
from eval import CausalMetric
from tqdm import tqdm

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

if __name__ == '__main__':
    args = get_args('pn2')

    print("starting point dropping experiment")

    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    test_dataset = ShapeNetDataset(
        root=args.point_path,
        split='test',
        classification=True,
        npoints=args.num_points,
        data_augmentation=False,
        unique=False,
        #unique_path='./unique-network-select')
    )

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=args.shuffle)  # batch Size set to 1 obtain a single example

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    classifier = Pointnet2(input_channels=0, num_classes=len(test_dataset.classes), use_xyz=True)
    if args.use_cuda:
        classifier.cuda()
    classifier.load_state_dict(torch.load(args.model))
    classifier.eval()  # eval not needed at this stage unless grad cam is commented

    # experiment on newer method using gradient before max pooling of the third feature layer

    grad_cam = GradCamThirdModule(classifier, target_layer_names=["2"], cuda=args.use_cuda,
                                  counterfactual=False, disable_relu=False, k=3)

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
        points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        B,C,N = points.shape
        cam = grad_cam(points)
        deletion = CausalMetric(classifier, 'del', N//10)
        del_score = deletion.single_run(points, cam)
        score.append(auc(del_score))
        print(sum(score) / len(score))













    # start = time.time()
    # drop_experiment = PointDropExperiment(
    #     classifier=classifier,
    #     grad_cam=grad_cam,
    #     testdataloader=testdataloader,
    #     num_drops=args.n_drops,
    #     steps=50,
    #     num_iterations=args.n_samples,
    #     update_cam=False,
    #     show_visualization=False,
    #     file_prefix='exp11-pn2-interpolate-',
    #     plot_title_prefix_text="EXP11 Point GradCAM Interpolate PointNet++",
    #     create_png=True,
    #     random_drop=False,
    #     high_drop=True,
    #     low_drop=True,
    #     print_progress=True,
    #     use_cuda=args.use_cuda,
    #     save_dropping_results=True,
    # ).run_experiment()
    # print("done in ", time.time() - start)

    # grad_cam_wrapper = GradcamWrapper(grad_cam, k=9)
    #
    # start = time.time()
    # drop_experiment = PointDropExperiment(
    #     classifier=classifier,
    #     grad_cam=grad_cam_wrapper,
    #     testdataloader=testdataloader,
    #     num_drops=args.num_drops,
    #     steps=50,
    #     num_iterations=args.num_iterate,
    #     update_cam=True,
    #     show_visualization=False,
    #     file_prefix='pn2maxpool-newModel_249-k9',
    #     create_png=False,
    #     random_drop=True,
    #     high_drop=True,
    #     low_drop=True,
    #     print_progress=True,
    # ).run_experiment()
    # print("done in ", time.time() - start)

    # grad_cam.counterfactual = True
    # start = time.time()
    # drop_experiment = PointDropExperiment(
    #     classifier=classifier,
    #     grad_cam=grad_cam_wrapper,
    #     testdataloader=testdataloader,
    #     num_drops=2048,
    #     steps=50,
    #     num_iterations=2874,  # max is 2874 = test set size
    #     update_cam=True,
    #     show_visualization=show_visualization,
    #     file_prefix='pn2maxpool-newModel_249-counterfactual-',
    #     alternative_colors=True,
    #     random_drop=True,
    #     high_drop=True,
    #     low_drop=True,
    #     print_progress=True,
    # ).run_experiment()
    # print("done in ", time.time() - start)