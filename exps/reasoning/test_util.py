import numpy as np
import copy
import torch
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import cv2
import json
import os

import random

from config import cfg
from lib.utils.post_3d import get_3d_points

params_transform = dict()
params_transform['crop_size_x'] = 832
params_transform['crop_size_y'] = 512
params_transform['center_perterb_max'] = 40
params_transform['scale_max'] = 1.1
params_transform['scale_min'] = 0.8

joint_to_limb_heatmap_relationship = cfg.DATASET.PAF.VECTOR
paf_z_coords_per_limb = list(range(cfg.DATASET.KEYPOINT.NUM))
NUM_LIMBS = len(joint_to_limb_heatmap_relationship)

def draw_heat(heatmap, pred_bodys_2d, plt_path, j):

    joint = [0,1,2,3,9,6,12]

    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    heatmap = heatmap.numpy()
    heatmap_vis = heatmap/np.max(heatmap)*255
    im = ax.imshow(heatmap_vis)
    for i in range(len(pred_bodys_2d)):
        x = pred_bodys_2d[i][joint[j]][0]
        y = pred_bodys_2d[i][joint[j]][1]
        circle = plt.Circle((x,y), 5, color='k', fill=False)
        plt.text(x,y,str(pred_bodys_2d[i][joint[j]][3]),color='k')
        ax.add_patch(circle)
    plt.colorbar(im)
    plt.title(plt_path)
    plt.savefig(plt_path)
    plt.close()


def predict_root(pred_bodys_2d, root):
    print('root.shape')
    print(root.shape)
    print('pred_bodys_2d.shape')
    print(pred_bodys_2d.shape)
    for i in range(len(root)):
        path = 'heatmap'+str(i)+'.png'
        draw_heat(root[i],pred_bodys_2d,path,i)
    return root

def register_pred(pred_bodys, gt_bodys, root_n=2):
    if len(pred_bodys) == 0:
        return np.asarray([])
    if gt_bodys is not None:
        root_gt = gt_bodys[:, root_n, :2]
        root_pd = pred_bodys[:, root_n, :2]
        distance_array = np.linalg.norm(root_gt[:, None, :] - root_pd[None, :, :], axis=2)
        corres = np.ones(len(gt_bodys), np.int) * -1
        occupied = np.zeros(len(pred_bodys), np.int)
        while np.min(distance_array) < 30:
            min_idx = np.where(distance_array == np.min(distance_array))
            for i in range(len(min_idx[0])):
                distance_array[min_idx[0][i]][min_idx[1][i]] = 50
                if corres[min_idx[0][i]] >= 0 or occupied[min_idx[1][i]]:
                    continue
                else:
                    corres[min_idx[0][i]] = min_idx[1][i]
                    occupied[min_idx[1][i]] = 1
        new_pred_bodys = np.zeros((len(gt_bodys), len(gt_bodys[0]), 4), np.float)
        for i in range(len(gt_bodys)):
            if corres[i] >= 0:
                new_pred_bodys[i] = pred_bodys[corres[i]]
    else:
        new_pred_bodys = pred_bodys[pred_bodys[:, root_n, 3] != 0]
    return new_pred_bodys


def chain_bones(pred_bodys, depth_v, i, depth_0=0, root_n=2):
    if root_n == 2:
        start_number = 2
        pred_bodys[i][2][2] = depth_0
        pred_bodys[i][0][2] = pred_bodys[i][2][2] - depth_v[i][1]
    else:
        start_number = 1
        pred_bodys[i][0][2] = depth_0
    pred_bodys[i][1][2] = pred_bodys[i][0][2] + depth_v[i][0]
    for k in range(start_number, NUM_LIMBS):
        src_k = joint_to_limb_heatmap_relationship[k][0]
        dst_k = joint_to_limb_heatmap_relationship[k][1]
        pred_bodys[i][dst_k][2] = pred_bodys[i][src_k][2] + depth_v[i][k]

def generate_relZ_oneRoot(pred_bodys, paf_3d_upsamp, root_d_upsamp, scale, num_intermed_pts=10, root_n=2):
    limb_intermed_coords = np.empty((2, num_intermed_pts), dtype=np.intp)
    depth_v = np.zeros((len(pred_bodys), NUM_LIMBS), dtype=np.float)
    depth_roots_pred = np.zeros(len(pred_bodys), dtype=np.float)
    for i, pred_body in enumerate(pred_bodys):
        if pred_body[root_n][3] > 0:
            depth_roots_pred[i] = root_d_upsamp[int(pred_body[root_n][1]), int(pred_body[root_n][0])] * scale['scale'] * scale['f_x']
            for k, bone in enumerate(joint_to_limb_heatmap_relationship):
                joint_src = pred_body[bone[0]]
                joint_dst = pred_body[bone[1]]
                if joint_dst[3] > 0 and joint_src[3] > 0:
                    depth_idx = paf_z_coords_per_limb[k]
                    # Linearly distribute num_intermed_pts points from the x
                    # coordinate of joint_src to the x coordinate of joint_dst
                    limb_intermed_coords[1, :] = np.round(np.linspace(
                        joint_src[0], joint_dst[0], num=num_intermed_pts))
                    limb_intermed_coords[0, :] = np.round(np.linspace(
                        joint_src[1], joint_dst[1], num=num_intermed_pts))  # Same for the y coordinate
                    intermed_paf = paf_3d_upsamp[limb_intermed_coords[0, :],
                                                 limb_intermed_coords[1, :], depth_idx]
                    min_val, max_val = np.percentile(intermed_paf, [10, 90])
                    intermed_paf[intermed_paf < min_val] = min_val
                    intermed_paf[intermed_paf > max_val] = max_val
                    mean_val = np.mean(intermed_paf)
                    depth_v[i][k] = mean_val
            chain_bones(pred_bodys, depth_v, i, depth_0=0)
    return depth_roots_pred


def generate_relZ(pred_bodys, paf_3d_upsamp, root_d_upsamp, scale, num_intermed_pts=10, root_n=2):
    joints_list = [0,1,2,3,9,6,12]

    limb_intermed_coords = np.empty((2, num_intermed_pts), dtype=np.intp)
    depth_v = np.zeros((len(pred_bodys), NUM_LIMBS), dtype=np.float)
    depth_roots_pred = np.zeros(len(pred_bodys), dtype=np.float)
    for i, pred_body in enumerate(pred_bodys):
        if pred_body[root_n][3] > 0:
            scores = 0.0
            depth_temp = 0.0
            if pred_body[root_n][3] > 3:
                depth_temp_out = root_d_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])]
            else:
                for m in range(len(joints_list)):
                    if pred_body[joints_list[m]][3] > 2:
                        depth_temp += pred_body[joints_list[m]][3] * root_d_upsamp[m][int(pred_body[joints_list[m]][1]), int(pred_body[joints_list[m]][0])]
                        scores += pred_body[joints_list[m]][3]
                    if scores > 0:
                        depth_temp_out = depth_temp / scores
                    else:
                        depth_temp_out = root_d_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])]
            '''# TODO: need to update. Now the naive solution is used.
            for m in range(len(joints_list)):
                if pred_body[joints_list[m]][3] > 2:
                    depth_temp += pred_body[joints_list[m]][3] * root_d_upsamp[m][int(pred_body[joints_list[m]][1]), int(pred_body[joints_list[m]][0])]
                    scores += pred_body[joints_list[m]][3]
            if scores > 0:
                depth_temp_out = depth_temp / scores
            else:
                depth_temp_out = root_d_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])]
            #
            if pred_body[joints_list[0]][3] > 2 and pred_body[joints_list[1]][3] > 2 and pred_body[joints_list[2]][3] > 2:
                for m in range(len(joints_list)):
                    depth_temp += root_d_upsamp[sample_list[m]][int(pred_body[joints_list[m]][1]), int(pred_body[joints_list[m]][0])]
                depth_temp_out = depth_temp / len(joints_list)
            elif pred_body[joints_list[root_n]][3] > 2:
                depth_temp_out = root_d_upsamp[2][int(pred_body[root_n][1]), int(pred_body[root_n][0])] # original vision
            elif pred_body[joints_list[1]][3] > 2 and pred_body[joints_list[2]][3] > 2:
                depth_temp_1 = root_d_upsamp[sample_list[1]][int(pred_body[joints_list[1]][1]), int(pred_body[joints_list[1]][0])]
                depth_temp_2 = root_d_upsamp[sample_list[2]][int(pred_body[joints_list[2]][1]), int(pred_body[joints_list[2]][0])]
                depth_temp_out = (depth_temp_1 + depth_temp_2)/2
            elif pred_body[joints_list[1]][3] > 2:
                depth_temp_out = root_d_upsamp[sample_list[1]][int(pred_body[joints_list[1]][1]), int(pred_body[joints_list[1]][0])]
            elif pred_body[joints_list[2]][3] > 2:
                depth_temp_out = root_d_upsamp[sample_list[2]][int(pred_body[joints_list[2]][1]), int(pred_body[joints_list[2]][0])]
            elif pred_body[0][3] > 2:
                depth_temp_out = root_d_upsamp[0][int(pred_body[0][1]), int(pred_body[0][0])] # use neck
            else:
                depth_temp_out = root_d_upsamp[2][int(pred_body[root_n][1]), int(pred_body[root_n][0])] # original vision
                '''
            '''
            #
            for m in range(len(joints_list)):
                if pred_body[joints_list[m]][3] > 1:
                    depth_temp += root_d_upsamp[m][int(pred_body[joints_list[m]][1]), int(pred_body[joints_list[m]][0])]
                    scores += 1
            if scores > 0:
                depth_temp_out = depth_temp / scores
            else:
                depth_temp_out = root_d_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])]
                '''
            '''
            #
            for m in range(len(joints_list)):
                if pred_body[joints_list[m]][3] > 2:
                    if m == 3:
                        if pred_body[9][3] > 2:
                            depth_temp += pred_body[joints_list[m]][3] * root_d_upsamp[m][int(pred_body[joints_list[m]][1]), int(pred_body[joints_list[m]][0])]
                            depth_temp += pred_body[9][3] * root_d_upsamp[m][int(pred_body[9][1]), int(pred_body[9][0])]
                            scores += pred_body[joints_list[m]][3]
                            scores += pred_body[9][3]
                    elif m == 4:
                        if pred_body[12][3] > 2:
                            depth_temp += pred_body[joints_list[m]][3] * root_d_upsamp[m][int(pred_body[joints_list[m]][1]), int(pred_body[joints_list[m]][0])]
                            depth_temp += pred_body[12][3] * root_d_upsamp[m][int(pred_body[12][1]), int(pred_body[12][0])]
                            scores += pred_body[joints_list[m]][3]
                            scores += pred_body[12][3]
                    else:
                        depth_temp += pred_body[joints_list[m]][3] * root_d_upsamp[m][int(pred_body[joints_list[m]][1]), int(pred_body[joints_list[m]][0])]
                        scores += pred_body[joints_list[m]][3]
            if scores > 0:
                depth_temp_out = depth_temp / scores
            else:
                depth_temp_out = root_d_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])]
            '''
            #depth_temp_out = root_d_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])] # original vision
            depth_roots_pred[i] = depth_temp_out * scale['scale'] * scale['f_x']
            for k, bone in enumerate(joint_to_limb_heatmap_relationship):
                joint_src = pred_body[bone[0]]
                joint_dst = pred_body[bone[1]]
                if joint_dst[3] > 0 and joint_src[3] > 0:
                    depth_idx = paf_z_coords_per_limb[k]
                    # Linearly distribute num_intermed_pts points from the x
                    # coordinate of joint_src to the x coordinate of joint_dst
                    limb_intermed_coords[1, :] = np.round(np.linspace(
                        joint_src[0], joint_dst[0], num=num_intermed_pts))
                    limb_intermed_coords[0, :] = np.round(np.linspace(
                        joint_src[1], joint_dst[1], num=num_intermed_pts))  # Same for the y coordinate
                    intermed_paf = paf_3d_upsamp[limb_intermed_coords[0, :],
                                                 limb_intermed_coords[1, :], depth_idx]
                    min_val, max_val = np.percentile(intermed_paf, [10, 90])
                    intermed_paf[intermed_paf < min_val] = min_val
                    intermed_paf[intermed_paf > max_val] = max_val
                    mean_val = np.mean(intermed_paf)
                    depth_v[i][k] = mean_val
            chain_bones(pred_bodys, depth_v, i, depth_0=0)
    return depth_roots_pred

def generate_relZ_combine(pred_bodys, pred_bodys_ori, paf_3d_upsamp, paf_3d_infer_upsamp, root_d_upsamp, root_d_infer_upsamp, scale, num_intermed_pts=10, root_n=2):
    joints_list = [0,1,2,3,9,6,12]

    limb_intermed_coords = np.empty((2, num_intermed_pts), dtype=np.intp)
    depth_v = np.zeros((len(pred_bodys), NUM_LIMBS), dtype=np.float)
    depth_roots_pred = np.zeros(len(pred_bodys), dtype=np.float)
    for i, pred_body in enumerate(pred_bodys):
        if pred_body[root_n][3] > 0: 
            scores = 0.0
            depth_temp = 0.0 
            ''' # multi-root V1
            if pred_body[root_n][3] > 3: 
                depth_temp_out = root_d_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])]
            else:
                for m in range(len(joints_list)):
                    if pred_body[joints_list[m]][3] > 2:
                        depth_temp += pred_body[joints_list[m]][3] * root_d_infer_upsamp[m][int(pred_body[joints_list[m]][1]), int(pred_body[joints_list[m]][0])]
                        scores += pred_body[joints_list[m]][3]
                    if scores > 0:
                        depth_temp_out = depth_temp / scores
                    else:
                        depth_temp_out = root_d_infer_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])]
            '''
            if pred_bodys_ori[i][root_n][3] > 3: 
                depth_temp_out = root_d_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])]
            elif pred_body[root_n][3] > 3:
                depth_temp_out = root_d_infer_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])]
            else:
                for m in range(len(joints_list)):
                    if pred_body[joints_list[m]][3] > 2:
                        depth_temp += pred_body[joints_list[m]][3] * root_d_infer_upsamp[m][int(pred_body[joints_list[m]][1]), int(pred_body[joints_list[m]][0])]
                        scores += pred_body[joints_list[m]][3]
                if scores > 0:
                    depth_temp_out = depth_temp / scores
                else:
                    depth_temp_out = root_d_infer_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])]
            
            #depth_temp_out = root_d_upsamp[root_n][int(pred_body[root_n][1]), int(pred_body[root_n][0])] # original vision
            depth_roots_pred[i] = depth_temp_out * scale['scale'] * scale['f_x']
            for k, bone in enumerate(joint_to_limb_heatmap_relationship):
                joint_src = pred_body[bone[0]]
                joint_dst = pred_body[bone[1]]
                if joint_dst[3] > 0 and joint_src[3] > 0:
                    depth_idx = paf_z_coords_per_limb[k]
                    # Linearly distribute num_intermed_pts points from the x
                    # coordinate of joint_src to the x coordinate of joint_dst
                    limb_intermed_coords[1, :] = np.round(np.linspace(
                        joint_src[0], joint_dst[0], num=num_intermed_pts))
                    limb_intermed_coords[0, :] = np.round(np.linspace(
                        joint_src[1], joint_dst[1], num=num_intermed_pts))  # Same for the y coordinate
                    if joint_dst[3] >3 and joint_src[3] > 3:
                        intermed_paf = paf_3d_upsamp[limb_intermed_coords[0, :],
                            limb_intermed_coords[1, :], depth_idx]
                    else:
                        intermed_paf = paf_3d_infer_upsamp[limb_intermed_coords[0, :],
                            limb_intermed_coords[1, :], depth_idx]
                    min_val, max_val = np.percentile(intermed_paf, [10, 90])
                    intermed_paf[intermed_paf < min_val] = min_val
                    intermed_paf[intermed_paf > max_val] = max_val
                    mean_val = np.mean(intermed_paf)
                    depth_v[i][k] = mean_val
            chain_bones(pred_bodys, depth_v, i, depth_0=0)
    return depth_roots_pred

def gen_3d_pose(pred_bodys, depth_root, scale):
    bodys = copy.deepcopy(pred_bodys)
    bodys[:, :, 0] = bodys[:, :, 0]/scale['scale'] - (scale['net_width']/scale['scale']-scale['img_width'])/2
    bodys[:, :, 1] = bodys[:, :, 1]/scale['scale'] - (scale['net_height']/scale['scale']-scale['img_height'])/2
    K = np.asarray([[scale['f_x'], 0, scale['cx']], [0, scale['f_y'], scale['cy']], [0, 0, 1]])
    bodys_3d = get_3d_points(bodys, depth_root, K)
    for i in range(bodys_3d.shape[0]):
        for j in range(bodys_3d.shape[1]):
            if bodys_3d[i, j, 3] == 0:
                bodys_3d[i, j] = 0
    return bodys_3d


def lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, device, root_n=2):
    root_3d_bodys = copy.deepcopy(pred_bodys_3d)
    root_2d_bodys = copy.deepcopy(pred_bodys_2d)
    score_after_refine = np.ones([pred_bodys_3d.shape[0], pred_bodys_3d.shape[1], 1], dtype=np.float)
    input_point = np.zeros((pred_bodys_3d.shape[0], 15, 5), dtype=np.float)
    input_point[:, root_n, :2] = root_2d_bodys[:, root_n, :2]
    input_point[:, root_n, 2:] = root_3d_bodys[:, root_n, :3]
    for i in range(len(root_3d_bodys)):
        if root_3d_bodys[i, root_n, 3] == 0:
            score_after_refine[i] = 0
        for j in range(len(root_3d_bodys[0])):
            if j != root_n and root_3d_bodys[i, j, 3] > 0:
                input_point[i, j, :2] = root_2d_bodys[i, j, :2] - root_2d_bodys[i, root_n, :2]
                input_point[i, j, 2:] = root_3d_bodys[i, j, :3] - root_3d_bodys[i, root_n, :3]
    input_point = np.resize(input_point, (input_point.shape[0], 75))
    inp = torch.from_numpy(input_point).float().to(device)
    pred = refine_model(inp)
    if pred.device.type == 'cuda':
        pred = pred.cpu().numpy()
    else:
        pred = pred.numpy()
    pred = np.resize(pred, (pred.shape[0], 15, 3))
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if j != root_n: #and pred_bodys_3d[i, j, 3] == 0:
                pred[i, j] += pred_bodys_3d[i, root_n, :3]
            else:
                pred[i, j] = pred_bodys_3d[i, j, :3]
    pred = np.concatenate([pred, score_after_refine], axis=2)
    return pred

## 使用 MGCN 版本的 refineNet
def lift_and_refineMGCN_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, device, root_n=2):
    root_3d_bodys = copy.deepcopy(pred_bodys_3d)
    root_2d_bodys = copy.deepcopy(pred_bodys_2d)
    score_after_refine = np.ones([pred_bodys_3d.shape[0], pred_bodys_3d.shape[1], 1], dtype=np.float)
    input_point = np.zeros((pred_bodys_3d.shape[0], 15, 5), dtype=np.float)
    input_point[:, root_n, :2] = root_2d_bodys[:, root_n, :2]
    input_point[:, root_n, 2:] = root_3d_bodys[:, root_n, :3]
    for i in range(len(root_3d_bodys)):
        if root_3d_bodys[i, root_n, 3] == 0:
            score_after_refine[i] = 0
        for j in range(len(root_3d_bodys[0])):
            if j != root_n and root_3d_bodys[i, j, 3] > 0:
                input_point[i, j, :2] = root_2d_bodys[i, j, :2] - root_2d_bodys[i, root_n, :2]
                input_point[i, j, 2:] = root_3d_bodys[i, j, :3] - root_3d_bodys[i, root_n, :3]
    #input_point = np.resize(input_point, (input_point.shape[0], 75))
    #input_point = input_point[:,:,2:]
    inp = torch.from_numpy(input_point).float().to(device)
    pred = refine_model(inp)
    if pred.device.type == 'cuda':
        pred = pred.cpu().numpy()
    else:
        pred = pred.numpy()
    #pred = np.resize(pred, (pred.shape[0], 15, 3))
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if j != root_n: #and pred_bodys_3d[i, j, 3] == 0:
                pred[i, j] += pred_bodys_3d[i, root_n, :3]
            else:
                pred[i, j] = pred_bodys_3d[i, j, :3]
    pred = np.concatenate([pred, score_after_refine], axis=2)
    return pred

def lift_and_refine_3d_pose_LQH(img_path, pred_bodys_2d, pred_bodys_3d, refine_model, refine_CNN_model, gt_bodys, device, root_n=2):

    root_3d_bodys = copy.deepcopy(pred_bodys_3d)
    root_2d_bodys = copy.deepcopy(pred_bodys_2d)

    ## Refine_CNN
    pred_point = np.zeros((pred_bodys_2d.shape[0], 15, 2), dtype=np.float)
    img_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
    for i in range(len(root_2d_bodys)):
        input_point_2d = copy.deepcopy(root_2d_bodys[i])
        img = img_ori.copy()
        img = aug_croppad(img, params_transform)
        img, clip_info = clip_img(img, gt_bodys[i][:,:2])

        ### generate heatmap
        heatmap = np.zeros((img.shape[0],img.shape[1]),dtype='float32')
        heatmap = generate_heatmap(heatmap, input_point_2d, clip_info)

        ### resize heatmap
        person_scrop = {}
        person_scrop["size_u"] = 130
        person_scrop["size_v"] = 260
        img_i, clip_info = person_croppad(img, person_scrop, clip_info)
        heatmap, _ = person_croppad(heatmap, person_scrop, clip_info, is_heatmap=True)

        img_in = torch.from_numpy(img_i.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0).float().to(device)
        heatmap_in = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float().to(device)

        preds = refine_CNN_model(img_in, heatmap_in)
        pred_2d = resize_back(preds,clip_info)

        # E2 , E3
        pred_point[i] = input_point_2d[:,:2] + pred_2d.detach().cpu().numpy() * (input_point_2d[:,:2] == 0).astype(np.int32)

        if random.random()<0.005:
            temp = pred_point[i]
            temp_name = random.randint(1,100)
            cv2.imwrite("debug/"+str(temp_name)+"cut.jpg",img_i)
            cv2.imwrite("debug/"+str(temp_name)+"heatmap.jpg",heatmap)
            heatmap_out = np.zeros((img.shape[0],img.shape[1]),dtype='float32')
            heatmap_out = generate_heatmap(heatmap_out, temp, clip_info, is_gt=True)
            heatmap_out, _ = person_croppad(heatmap_out, person_scrop, clip_info, is_heatmap=True)
            cv2.imwrite("debug/"+str(temp_name)+"heatmap_out.jpg",heatmap_out)


    root_2d_bodys = copy.deepcopy(pred_point)


    ## Refine
    score_after_refine = np.ones([pred_bodys_3d.shape[0], pred_bodys_3d.shape[1], 1], dtype=np.float)
    input_point = np.zeros((pred_bodys_3d.shape[0], 15, 5), dtype=np.float)
    input_point[:, root_n, :2] = root_2d_bodys[:, root_n, :2]
    input_point[:, root_n, 2:] = root_3d_bodys[:, root_n, :3]
    for i in range(len(root_3d_bodys)):
        if root_3d_bodys[i, root_n, 3] == 0:
            score_after_refine[i] = 0
        for j in range(len(root_3d_bodys[0])):
            if j != root_n and root_3d_bodys[i, j, 3] > 0:
                input_point[i, j, :2] = root_2d_bodys[i, j, :2] - root_2d_bodys[i, root_n, :2]
                input_point[i, j, 2:] = root_3d_bodys[i, j, :3] - root_3d_bodys[i, root_n, :3]
    input_point = np.resize(input_point, (input_point.shape[0], 75))
    inp = torch.from_numpy(input_point).float().to(device)
    pred = refine_model(inp)
    if pred.device.type == 'cuda':
        pred = pred.cpu().numpy()
    else:
        pred = pred.numpy()
    pred = np.resize(pred, (pred.shape[0], 15, 3))
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if j != root_n: #and pred_bodys_3d[i, j, 3] == 0:
                pred[i, j] += pred_bodys_3d[i, root_n, :3]
            else:
                pred[i, j] = pred_bodys_3d[i, j, :3]
    pred = np.concatenate([pred, score_after_refine], axis=2)
    return pred


def save_result_for_train_refine(pred_bodys_2d, pred_bodys_3d, gt_bodys, pred_rdepths, img_path,
                                 result, root_n=2):
    for i, pred_body in enumerate(pred_bodys_3d):
        if pred_body[root_n][3] != 0:
            pair = {}
            pair['pred_3d'] = pred_body.tolist()
            pair['pred_2d'] = pred_bodys_2d[i].tolist()
            pair['gt_3d'] = gt_bodys[i][:, 4:7].tolist()
            pair['gt_2d'] = gt_bodys[i][:, :3].tolist()
            pair['root_d'] = pred_rdepths[i]
            pair['img_path'] = img_path 
            result['3d_pairs'].append(pair)


def save_result(pred_bodys_2d, pred_bodys_3d, gt_bodys, pred_rdepths, img_path, result):
    pair = dict()
    pair['pred_2d'] = pred_bodys_2d.tolist()
    pair['pred_3d'] = pred_bodys_3d.tolist()
    pair['root_d'] = pred_rdepths.tolist()
    pair['image_path'] = img_path
    if gt_bodys is not None:
        pair['gt_3d'] = gt_bodys[:, :, 4:].tolist()
        pair['gt_2d'] = gt_bodys[:, :, :4].tolist()
    else:
        pair['gt_3d'] = list()
        pair['gt_2d'] = list()
    result['3d_pairs'].append(pair)

def clip_img(img, gt_point_2d):
    clip_info = dict()

    v_min, u_min = np.min(gt_point_2d,axis=0)[1], np.min(gt_point_2d,axis=0)[0]
    v_max, u_max = np.max(gt_point_2d,axis=0)[1], np.max(gt_point_2d,axis=0)[0]

    img_shape = img.shape
    v_min = int(max(0,v_min-25))
    u_min = int(max(0,u_min-25))
    v_max = int(min(img_shape[0]-1, v_max+25))
    u_max = int(min(img_shape[1]-1, u_max+25))

    clip_info['top_left_u'] = u_min
    clip_info['top_left_v'] = v_min
    clip_info['size_u'] = u_max-u_min
    clip_info['size_v'] = v_max-v_min

    img = img[clip_info['top_left_v']:clip_info['top_left_v']+clip_info['size_v'],
            clip_info['top_left_u']:clip_info['top_left_u']+clip_info['size_u'],:]

    return img, clip_info

def generate_heatmap(heatmap, input_point_2d, clip_info, kernel=(7, 7), is_gt=False):
    for i in range(len(input_point_2d)):
        u = int(max(0,input_point_2d[i][0] - clip_info['top_left_u']))
        v = int(max(0,input_point_2d[i][1] - clip_info['top_left_v']))
        if int(v)>=heatmap.shape[0] or int(u)>=heatmap.shape[1]:
            continue
        if is_gt:
            heatmap[int(v),int(u)] = 1
        else:
            heatmap[int(v),int(u)] = input_point_2d[i][3]
    heatmap = cv2.GaussianBlur(heatmap, kernel, 0)
    maxi = np.max(heatmap)

    if maxi <= 1e-8:
        maxi = 1

    heatmap /= maxi / 255

    return heatmap

def person_croppad(img, person_scrop, clip_info, is_heatmap=False):

    if is_heatmap:
        out_img = np.zeros([person_scrop['size_v'],person_scrop['size_u']])
    else:
        out_img = np.zeros([person_scrop['size_v'],person_scrop['size_u'],3])
   
    scale = min(1, person_scrop['size_u'] / float(img.shape[1]),
                person_scrop['size_v'] / float(img.shape[0]))

    clip_info["scale"] = scale
    
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    if is_heatmap:
        out_img[:int(img.shape[0]),:int(img.shape[1])] = img
    else:
        out_img[:int(img.shape[0]),:int(img.shape[1]),:] = img

    return out_img, clip_info

def point_croppad(input_point_2d, clip_info):
    out_point = np.zeros(input_point_2d.shape)
    out_point = (input_point_2d - np.array([clip_info['top_left_u'],clip_info['top_left_v']])) * clip_info["scale"]
    return out_point

def resize_back(preds,clip_info):
    pred_2d = preds.clone().cpu().reshape((15, 2))
    pred_2d = (pred_2d / clip_info["scale"]) + torch.tensor([clip_info['top_left_u'],clip_info['top_left_v']])
    return pred_2d

def aug_croppad(img, params_transform):
    dice_x = random.random()
    dice_y = random.random()
    scale_random = random.random()
    scale_multiplier = ((params_transform['scale_max'] - params_transform['scale_min']) *
                    scale_random + params_transform['scale_min'])
    crop_x = int(params_transform['crop_size_x'])
    crop_y = int(params_transform['crop_size_y'])
   
    scale = min(params_transform['crop_size_x'] / float(img.shape[1]),
            params_transform['crop_size_y'] / float(img.shape[0]))
        
    x_offset = int((dice_x - 0.5) * 2 *
               params_transform['center_perterb_max'])
    y_offset = int((dice_y - 0.5) * 2 *
               params_transform['center_perterb_max'])

    width = img.shape[1]//2
    height = img.shape[0]//2
    center = np.array([width, height]) * scale # + np.array([x_offset, y_offset])
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)


    # pad up and down
    pad_v = np.ones((crop_y, img.shape[1], 3), dtype=np.uint8) * 128
    img = np.concatenate((pad_v, img, pad_v), axis=0)

    # pad right and left
    pad_h = np.ones((img.shape[0], crop_x, 3), dtype=np.uint8) * 128
    img = np.concatenate((pad_h, img, pad_h), axis=1)


    img = img[int(center[1] + crop_y / 2):int(center[1] + crop_y / 2 + crop_y),
          int(center[0] + crop_x / 2):int(center[0] + crop_x / 2 + crop_x), :]


    return img


