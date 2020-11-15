import os.path as osp
import pickle
import shutil
import tempfile
import time
import numpy as np
import csv
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

import numpy as np
from scipy.spatial import ConvexHull

from imantics import Polygons, Mask
from PIL import Image, ImageDraw 

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results

BINARY = False

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.round(np.dot([x1, y2], r))
    rval[1] = np.round(np.dot([x2, y2], r))
    rval[2] = np.round(np.dot([x2, y1], r))
    rval[3] = np.round(np.dot([x1, y1], r))
    
    for i in range(4):
        for j in range(2):
            rval[i][j] = int(rval[i][j])

    return rval

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3, out=False):
    model.eval()
    results = []
    poly_results = []
    filenames = []
    sides_list = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result) # 1
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            height = img_metas[0]['ori_shape'][0]
            width = img_metas[0]['ori_shape'][1]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas) #1
            for j, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                filename = img_meta['ori_filename']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                filenames.append(filename)

                model.module.show_result(
                    img_show,
                    result[j],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
#             for i in range(100):
#                 image = np.array(result[0][1])[:,i,:,:].squeeze()*255.0  # class, proposed, width, height
#                 im = Image.fromarray(image)
#                 im = im.convert("L")
#                 im.save(f"./images_for_check/{i}.png")  # why only 100??
        

            
            
            # get poly from mask
            if batch_size ==1 and out:
                poly_results_per_image = []
                             
                mask = np.array(result[0][1]) # (1, 105, 852, 852) 
                bbox = np.array(result[0][0]) # (1, 105, 5)
                
#                 print(np.array(mask[0]).shape)  # 15, 852, 852)
#                 print(np.array(bbox[0]).shape) # 15,5
                
    
    
    
                if BINARY:
                    proposed_num = mask.shape[1]

                    assert len(bbox[0]) == proposed_num, "the num of bbox should equal to that of segm"

                    #get poly
                    for k in range(proposed_num):
                        mask_array = mask[:,k,:,:].squeeze()*1 #852,852
                        polygons = Mask(mask_array).polygons()
                        coords = (polygons.points)


                        # one proposal should have one blob
                        if len(coords) ==1:
                            coords = coords[0]
                        else:
                            main_coord_idx = 0
                            len_main_coord = len(coords[main_coord_idx])
    #                         print("main", len_main_coord)
                            for l in range(1,len(coords)):
    #                             print(len(coords[l]))
                                if len(coords[l]) > len_main_coord:
                                    main_coord_idx = l
                                    len_main_coord = len(coords[l])
                            coords = coords[main_coord_idx]


                        ##  get only 4 points
                        x = []
                        y = []
                        for coord in coords:
                            x.append(coord[0])
                            y.append(coord[1])

                        x = np.array(x)
                        y = np.array(y)
                        x_max_idx = np.argmax(x)
                        x_min_idx = np.argmin(x)
                        y_max_idx = np.argmax(y)
                        y_min_idx = np.argmin(y)

                        coords = np.array([[x[x_min_idx], y[x_min_idx]],
                                  [x[y_min_idx], y[y_min_idx]],
                                  [x[x_max_idx], y[x_max_idx]],
                                  [x[y_max_idx], y[y_max_idx]]
                                  ])

                        #calculate sides
                        s = [0]*len(coords)
                        for m in range(len(coords)):
                            s[m] = norm(coords[m+1] - coords[m]) if m+1 != len(coords) else norm(coords[0] - coords[m]) 

                        #averaging max_sides
                        max_idx = np.argmax(s)
                        max_length = s[max_idx]
                        sub_max_idx = (max_idx + 2)%len(coords)
                        sub_max_length = s[sub_max_idx]
                        avg_max_length = (max_length + sub_max_length)/2



                        if max_idx == 1 or max_idx == 3:
                            coords = list(coords[-1:]) + list(coords[:-1])
                            coords = np.array(coords)


                        if not (s[sub_max_idx] > s[(sub_max_idx+1)%4] and s[sub_max_idx]> s[(sub_max_idx-1)%4]):
            
                            if (x[x_max_idx]- x[x_min_idx]) > (y[y_max_idx]- y[y_min_idx]):
                                avg_max_length = (x[x_max_idx]- x[x_min_idx])

                                coords = np.array([[x[x_min_idx], y[y_min_idx]],
                                  [x[x_max_idx], y[y_min_idx]],
                                  [x[x_max_idx], y[y_max_idx]],
                                  [x[x_min_idx], y[y_max_idx]]
                                  ])
                            else:
                                coords = np.array([[x[x_max_idx], y[y_min_idx]],
                                  [x[x_max_idx], y[y_max_idx]],
                                  [x[x_min_idx], y[y_max_idx]],
                                  [x[x_min_idx], y[y_min_idx]]
                                  ])
                                avg_max_length = (y[y_max_idx]- y[y_min_idx])
                            # 회전하면 되는데 귀찮다
                            image = mask_array*255.0  # class, proposed, width, height
                            im = Image.fromarray(image)
                            im = im.convert("L")
                            im.save(f"./images_for_check/special{i}_{k}.png")  # why only 100??


                        #class check!
                        if height != width:
                            raise ValueError("image should have the form of square!")

                        normed_ml = avg_max_length/height
                        # 35.5, 44, 52, 60,68,76.5
                        if normed_ml < 35.5/852:
                            class_pred = 1
                        elif normed_ml < 44/852:
                            class_pred = 2
                        elif normed_ml < 52/852:
                            class_pred = 3
                        elif normed_ml < 60/852:
                            class_pred = 4
                        elif normed_ml < 68/852:
                            class_pred = 5
                        elif normed_ml < 76.5/852:
                            class_pred = 6
                        elif normed_ml >= 76.5/852:
                            class_pred = 7
                        sides_list.append(normed_ml)
#                         print(normed_ml, class_pred)
                        poly_results_per_image.append((class_pred, bbox[0][k][-1], coords ))
                        
                else:
                    num_of_classes = len(mask)
                    for k in range(num_of_classes):
                        class_mask= np.array(mask[k]) # 15, 852, 852
                        class_bbox = np.array(bbox[k]) # 15,5
                        
                        proposed_num = class_mask.shape[0]
                        
#                         print("rop", proposed_num)
                        for l in range(proposed_num):
                            mask_array = class_mask[l,:,:]
#                             polygons = Mask(mask_array).polygons()
#                             coords = polygons.points
                            
#                             # one proposal should have one blob
                            
#                             if len(coords) ==1:
#                                 coords = coords[0]
#                             else:
#                                 main_coord_idx = 0
#                                 len_main_coord = len(coords[main_coord_idx])
#         #                         print("main", len_main_coord)
#                                 for m in range(1,len(coords)):
#         #                             print(len(coords[l]))
#                                     if len(coords[m]) > len_main_coord:
#                                         main_coord_idx = m
#                                         len_main_coord = len(coords[m])
#                                 coords = coords[main_coord_idx]
                            
                            coords = np.argwhere(mask_array.T == True)
                            coords= minimum_bounding_rectangle(coords)

#                             ##  get only 4 points
#                             x = []
#                             y = []
#                             for coord in coords:
#                                 x.append(coord[0])
#                                 y.append(coord[1])

#                             x = np.array(x)
#                             y = np.array(y)
#                             x_max_idx = np.argmax(x)
#                             x_min_idx = np.argmin(x)
#                             y_max_idx = np.argmax(y)
#                             y_min_idx = np.argmin(y)

#                             coords = np.array([[x[x_min_idx], y[x_min_idx]],
#                                       [x[y_min_idx], y[y_min_idx]],
#                                       [x[x_max_idx], y[x_max_idx]],
#                                       [x[y_max_idx], y[y_max_idx]]
#                                       ])
                            
#                             #calculate sides
#                             s = [0]*len(coords)
#                             for n in range(len(coords)):
#                                 s[n] = norm(coords[n+1] - coords[n]) if n+1 != len(coords) else norm(coords[0] - coords[n]) 

#                             #averaging max_sides
#                             max_idx = np.argmax(s)
#                             max_length = s[max_idx]
#                             sub_max_idx = (max_idx + 2)%len(coords)
#                             sub_max_length = s[sub_max_idx]
#                             avg_max_length = (max_length + sub_max_length)/2
                            
#                             if max_idx == 1 or max_idx == 3:
#                                 coords = list(coords[-1:]) + list(coords[:-1])
#                                 coords = np.array(coords)
                                
#                             if not (s[sub_max_idx] > s[(sub_max_idx+1)%4] and s[sub_max_idx]> s[(sub_max_idx-1)%4]):
#                                 if (x[x_max_idx]- x[x_min_idx]) > (y[y_max_idx]- y[y_min_idx]):
#                                     avg_max_length = (x[x_max_idx]- x[x_min_idx])

#                                     coords = np.array([[x[x_min_idx], y[y_min_idx]],
#                                       [x[x_max_idx], y[y_min_idx]],
#                                       [x[x_max_idx], y[y_max_idx]],
#                                       [x[x_min_idx], y[y_max_idx]]
#                                       ])
#                                 else:
#                                     coords = np.array([[x[x_max_idx], y[y_min_idx]],
#                                       [x[x_max_idx], y[y_max_idx]],
#                                       [x[x_min_idx], y[y_max_idx]],
#                                       [x[x_min_idx], y[y_min_idx]]
#                                       ])
#                                     avg_max_length = (y[y_max_idx]- y[y_min_idx])
#                                 image = mask_array*255.0  # class, proposed, width, height
#                                 im = Image.fromarray(image)
#                                 im = im.convert("L")
#                                 im.save(f"./images_for_check/special{i}_{k}.png")  # why only 100??

                            class_pred = k+1
                            
                        
                            
                            poly_results_per_image.append((class_pred, class_bbox[l][-1], coords ))
                            
                            imcheck = Image.fromarray(mask_array).convert("RGB")
                            img1 = ImageDraw.Draw(imcheck) 
#                             coords_tuple = [(round(coord[0]), round(coord[1]))for coord in coords]
#                             img1.polygon(coords_tuple, fill=(128, 0, 0, 50), outline ="blue")  
                            
#                             imcheck.save(f"./images_for_check2/special{i}_{k}.png")  # why only 100??
                        

                    
                      
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]            

        results.extend(result)
        if out:
            poly_results.append(poly_results_per_image)

        for _ in range(batch_size):
            prog_bar.update()
    if out:
        assert len(poly_results) == len(filenames) 
    if out:
        with open('../your_file2.txt', 'w') as f:
            for item in sides_list:
                f.write("%s\n" % item)
            
        return results, poly_results, filenames
    else:
        return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
