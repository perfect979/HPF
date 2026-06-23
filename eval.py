import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt



def get_class_name(index):
    file_path = 'shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/synsetoffset2category.txt'
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, start=1):
            if i == index:
                # 假设每行的单词和数字之间是由制表符或多个空格分隔
                word = line.split()[0]  # 分割每行并获取第一个元素，即单词
                return word



class CausalMetric():
    def __init__(self, model, mode, step):
        self.model = model
        self.mode = mode
        self.step = step
    
    def single_run(self, input, cam):
        B, C, N = input.shape
        # points = input.clone().detach()
        # pred = F.softmax(self.model(input.cuda())[0], dim=1)  # me
        pred = F.softmax(self.model(input.cuda()), dim=1)  # his
        index = torch.argmax(pred).item()  # 最开始的类别

        if pred[0][index] < 0.9:
            return

        n_steps = (N + self.step - 1) // self.step  # 总点数，除以每次删除的点，得需要执行循环得次数
        if self.mode == 'del':  # 模式
            title = 'Deletion'
            ylabel = 'Pixels Deleted'
            start = input.clone()
        else:
            title = 'ins'
            ylabel = 'Pixels Insertion'
            start = input.clone().detach()
            start[:, :, :] = 0  # 将所有点初始化到坐标原点(变相删除)

        points = start

        scores = np.empty(n_steps + 1)  # 设计一个储存得分得数组
        exp = np.copy(cam)
        for i in range(n_steps+1):  # 开始循环
            pred = F.softmax(self.model(points.cuda()), dim=1)  # his
            # pred = F.softmax(self.model(points.cuda())[0], dim=1)  # me

            pred_index = torch.argmax(pred)  # 预测的类别
            scores[i] = pred[0, index]  # 将每次迭代得得分储存
            
            if i < n_steps:
                
                for j in range(self.step):
                    idx = np.argmax(exp)
                    # if (exp[idx] * 255) < 170:
                    #     break
                    exp[idx] = -1.5

                    if self.mode == 'del':
                        points[:, :, idx] = 0  # 如果是删除，则将点移动至原点
                    else:
                        points[:, :, idx] = input[:, :, idx]  # 如果是插入，则将点从原点移动至原位置



        x_values = np.linspace(0, 1, n_steps+1)
        plt.figure(figsize=(5, 5))  # 可以调整图表大小
        plt.plot(x_values, scores)  # 用圆点标记每个点
        plt.title('Scores Distribution')  # 图表标题
        plt.xlabel('Normalized Steps')  # x轴标题
        plt.ylabel('Scores')  # y轴标题
        plt.ylim(0, 1)  # 设置y轴的范围为0到1
        plt.grid(True)  # 显示网格
        plt.savefig('./plot.png', format='png', dpi=300)
        plt.close()  # 显示图表

        # plt.figure(figsize=(5, 5))
        # plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
        # plt.xlim(-0.1, 1.1)
        # plt.ylim(0, 1.05)
        # plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
        # plt.title(title)
        # plt.xlabel(ylabel)
        # plt.ylabel(get_class_name(index))
        # plt.savefig('./plot.png', dpi=300)
        # plt.close()
        return scores
                
                

    def drop_red_point(self, input, exp, target_idx):
        B, C, N = points.shape
        if self.mode == 'del':  # 删除从所有点开始
            points = input.clone().detach()
        else:  # 插入从0开始
            centerd_input = input.clone().detach()
            centerd_input[:, :, :] = 0  # 将所有点初始化到坐标原点(变相删除)
            points = centerd_input
        
        cam = np.copy(exp)
        cam_add_gate = np.zeros(cam.size, dtype=np.dtype(np.float32))
        k = 1

        n_steps = (N + self.step - 1) // self.step
        score = np.empty(n_steps + 1)

        # 做删除或插入操作
        for i in range(self.step):
            idx = np.argmax(cam)
            if (cam[idx] * 255) < 170:
                return score
            
            cam[idx] = -1.5
            cam_add_gate = -2.5
            if self.mode == 'del':
                points[:, :, idx] = 0  # 如果是删除，则将点移动至原点
            else:
                points[:, :, idx] = input[:, :, idx]  # 如果是插入，则将点从原点移动至原位置
        
        # 重新预测
        pred = F.softmax(self.model(points.cuda())[0], dim=0)
        pred_index = torch.argmax(pred).cpu().numpy()[0]


class CausalMetric_forme():
    def __init__(self, model, mode, step):
        self.model = model
        self.mode = mode
        self.step = step
    
    def single_run(self, input, cam):
        B, C, N = input.shape
        # points = input.clone().detach()
        # pred = F.softmax(self.model(input.cuda())[0], dim=1)  # me
        pred = F.softmax(self.model(input.cuda()), dim=1)  # his
        index = torch.argmax(pred).item()  # 最开始的类别

        if pred[0][index] < 0.9:
            return

        n_steps = (N + self.step - 1) // self.step  # 总点数，除以每次删除的点，得需要执行循环得次数
        if self.mode == 'del':  # 模式
            title = 'Deletion'
            ylabel = 'Pixels Deleted'
            start = input.clone()
        else:
            title = 'ins'
            ylabel = 'Pixels Insertion'
            start = input.clone().detach()
            start[:, :, :] = 0  # 将所有点初始化到坐标原点(变相删除)

        points = start

        scores = np.empty(n_steps + 1)  # 设计一个储存得分得数组
        exp = np.copy(cam)
        for i in range(n_steps+1):  # 开始循环
            pred = F.softmax(self.model(points.cuda()), dim=1)  # his
            # pred = F.softmax(self.model(points.cuda())[0], dim=1)  # me

            pred_index = torch.argmax(pred)  # 预测的类别
            scores[i] = pred[0, index]  # 将每次迭代得得分储存
            
            if i < n_steps:
                
                for j in range(self.step):
                    idx = np.argmax(exp)
                    # if (exp[idx] * 255) < 170:
                    #     break
                    exp[idx] = -1.5

                    if self.mode == 'del':
                        points[:, :, idx] = 0  # 如果是删除，则将点移动至原点
                    else:
                        points[:, :, idx] = input[:, :, idx]  # 如果是插入，则将点从原点移动至原位置



        x_values = np.linspace(0, 1, n_steps+1)
        plt.figure(figsize=(5, 5))  # 可以调整图表大小
        plt.plot(x_values, scores)  # 用圆点标记每个点
        plt.title('Scores Distribution')  # 图表标题
        plt.xlabel('Normalized Steps')  # x轴标题
        plt.ylabel('Scores')  # y轴标题
        plt.ylim(0, 1)  # 设置y轴的范围为0到1
        plt.grid(True)  # 显示网格
        plt.savefig('./plots.png', format='png', dpi=300)
        plt.close()  # 显示图表

        # plt.figure(figsize=(5, 5))
        # plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
        # plt.xlim(-0.1, 1.1)
        # plt.ylim(0, 1.05)
        # plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
        # plt.title(title)
        # plt.xlabel(ylabel)
        # plt.ylabel(get_class_name(index))
        # plt.savefig('./plot.png', dpi=300)
        # plt.close()
        return scores