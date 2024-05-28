

import numpy as np
import cv2
import torch 
import time
import os

from .model import create_model,load_model
from .image import get_affine_transform, transform_preds
from .decode import ctdet_4ps_decode, ctdet_cls_decode


def ctdet_4ps_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, 0:2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        dets[i, :, 4:6] = transform_preds(dets[i, :, 4:6], c[i], s[i], (w, h))
        dets[i, :, 6:8] = transform_preds(dets[i, :, 6:8], c[i], s[i], (w, h))
        classes = dets[i, :, 9]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :8].astype(np.float32),
                dets[i, inds, 8:].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


def pnms(dets, thresh):
    if len(dets) < 2:
        return dets
    scores = dets[:, 8]
    index_keep = []
    keep = []
    for i in range(len(dets)):
        box = dets[i]
        if box[8] < thresh:
            continue
        max_score_index = -1
        ctx = (dets[i][0] + dets[i][2] + dets[i][4] + dets[i][6]) / 4
        cty = (dets[i][1] + dets[i][3] + dets[i][5] + dets[i][7]) / 4
        for j in range(len(dets)):
            if i == j or dets[j][8] < thresh:
                continue
            x1, y1 = dets[j][0], dets[j][1]
            x2, y2 = dets[j][2], dets[j][3]
            x3, y3 = dets[j][4], dets[j][5]
            x4, y4 = dets[j][6], dets[j][7]
            a = (x2 - x1) * (cty - y1) - (y2 - y1) * (ctx - x1)
            b = (x3 - x2) * (cty - y2) - (y3 - y2) * (ctx - x2)
            c = (x4 - x3) * (cty - y3) - (y4 - y3) * (ctx - x3)
            d = (x1 - x4) * (cty - y4) - (y1 - y4) * (ctx - x4)
            if ((a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0)):
                if dets[i][8] > dets[j][8] and max_score_index < 0:
                    max_score_index = i
                elif dets[i][8] < dets[j][8]:
                    max_score_index = -2
                    break
        if max_score_index > -1:
            index_keep.append(max_score_index)
        elif max_score_index == -1:
            index_keep.append(i)
    for i in range(0, len(index_keep)):
        keep.append(dets[index_keep[i]])

    return np.array(keep)

class Detector:
    def __init__(self,model_dir):
        # check if use CUDA
        self.arch = "dlav0subfield_34"
        self.heads = {'hm': 11, 'cls': 4, 'ftype': 3, 'wh': 8, 'hm_sub': 2, 'wh_sub': 8, 'reg': 2, 'reg_sub': 2}
        self.head_conv = 256
        self.convert_onnx = 0
        self.load_model = model_dir
        self.device = torch.device('cpu')
        self.model = create_model(self.arch, self.heads, self.head_conv, self.convert_onnx, {})
        self.model = load_model(self.model, self.load_model)
        self.model = self.model.to(self.device)
        self.model.eval()

        __mean = [[[0.40789655, 0.44719303, 0.47026116]]]
        __std = [[[0.2886383,  0.27408165, 0.27809834]]]

        self.mean = np.array(__mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(__std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = 13
        self.scales = [1.0]
        # self.opt = opt
        self.pause = True
        self.input_h = 768
        self.input_w = 768
        self.pad = 0
        self.flip_test = False
        self.down_ratio = 4
        self.fix_res = True
        self.K = 100
        self.flip_test = False
        self.reg_offset = 0
        self.nms = True
        self.scores_thresh = 0.3


    def preprocess(self,image,scale,meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.fix_res:
            inp_height, inp_width = self.input_h, self.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.pad)  # + 1
            inp_width = (new_width | self.pad)  # + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        vis_image = inp_image
        # import pdb; pdb.set_trace()
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'input_height': inp_height,
                'input_width': inp_width,
                'vis_image': vis_image,
                'out_height': inp_height // self.down_ratio,
                'out_width': inp_width // self.down_ratio}
        return images, meta

    def process(self,images):
        with torch.no_grad():
            output = self.model(images)[-1]
            """if self.opt.convert_onnx == 1:
                torch.cuda.synchronize()
                inputs = ['data']
                outputs = ['hm.0.sigmoid', 'hm.0.maxpool', 'cls.0.sigmoid', 'ftype.0.sigmoid', 'wh.2', 'reg.2', 'hm_sub.0.sigmoid', 'hm_sub.0.maxpool', 'wh_sub.2', 'reg_sub.2' ]
                dynamic_axes = {'data': {2: 'h', 3: 'w'}, 'hm.0.sigmoid': {2: 'H', 3: 'W'},
                                'hm.0.maxpool': {2: 'H', 3: 'W'}, 'cls.0.sigmoid': {2: 'H', 3: 'W'},
                                'ftype.0.sigmoid': {2: 'H', 3: 'W'}, 'wh.2': {2: 'H', 3: 'W'},
                                'reg.2': {2: 'H', 3: 'W'},
                                'hm_sub.0.sigmoid': {2: 'H', 3: 'W'},
                                'hm_sub.0.maxpool': {2: 'H', 3: 'W'}, 
                                'wh_sub.2': {2: 'H', 3: 'W'},
                                'reg_sub.2': {2: 'H', 3: 'W'}}

                onnx_path = self.opt.onnx_path
                if self.opt.onnx_path == "auto":
                    onnx_path = "{}_{}cls_{}ftype.onnx".format(self.opt.dataset, self.opt.num_classes,
                                                               self.opt.num_secondary_classes)

                torch.onnx.export(self.model, images, onnx_path,
                                  input_names=inputs, output_names=outputs,
                                  dynamic_axes=dynamic_axes, do_constant_folding=True,
                                  opset_version=10)
                print("--> info: onnx is saved at: {}".format(onnx_path))
                cls = output['cls_sigmoid']
                hm = output['hm_sigmoid']
                ftype = output['ftype_sigmoid']
                
                # add sub
                hm_sub = output['hm_sigmoid_sub']"""

            hm = output['hm'].sigmoid_()
            cls = output['cls'].sigmoid_()
            ftype = output['ftype'].sigmoid_()

            hm_sub = output['hm_sub'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.reg_offset else None
            
            # add sub
            wh_sub = output['wh_sub']
            reg_sub = output['reg_sub'] if self.reg_offset else None
            
            torch.cuda.synchronize()
            # forward_time = time.time()
            # return dets [bboxes, scores, clses]
            dets, inds = ctdet_4ps_decode(hm, wh, reg=reg, K=self.K)
            
            #add sub
            dets_sub, inds_sub = ctdet_4ps_decode(hm_sub, wh_sub, reg=reg_sub, K=self.K)
            
            box_cls = ctdet_cls_decode(cls, inds)
            box_ftype = ctdet_cls_decode(ftype, inds)
            clses = torch.argmax(box_cls, dim=2, keepdim=True)
            ftypes = torch.argmax(box_ftype, dim=2, keepdim=True)
            dets = np.concatenate(
                (dets.detach().cpu().numpy(), clses.detach().cpu().numpy(), ftypes.detach().cpu().numpy()), axis=2)
            dets = np.array(dets)
             
            # add subfield
            dets_sub = np.concatenate(
                (dets_sub.detach().cpu().numpy(), clses.detach().cpu().numpy(), ftypes.detach().cpu().numpy()), axis=2)
            dets_sub = np.array(dets_sub)
            dets_sub[:,:,-3] += 11  
             
            corner = 0
        return output, dets, dets_sub

    def postprocess(self, dets, corner, meta, scale=1):
        if self.nms:
            detn = pnms(dets[0], self.scores_thresh)
            if detn.shape[0] > 0:
                dets = detn.reshape(1, -1, detn.shape[1])
        k = dets.shape[2] if dets.shape[1] != 0 else 0
        if dets.shape[1] != 0:
            dets = dets.reshape(1, -1, dets.shape[2])
            # return dets is list and what in dets is dict. key of dict is classes, value of dict is [bbox,score]
            dets = ctdet_4ps_post_process(
                dets.copy(), [meta['c']], [meta['s']],
                meta['out_height'], meta['out_width'], self.num_classes)
            for j in range(1, self.num_classes + 1):
                dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, k)
                dets[0][j][:, :8] /= scale
        else:
            ret = {}
            dets = []
            for j in range(1, self.num_classes + 1):
                ret[j] = np.array([0] * k, dtype=np.float32)  # .reshape(-1, k)
            dets.append(ret)
        return dets[0], corner


    def Duplicate_removal(self, results):
        bbox = []
        for box in results:
            if box[8] > self.scores_thresh:
                bbox.append(box)
        if len(bbox) > 0:
            return np.array(bbox)
        else:
            return np.array([[0] * 12])

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            # if len(self.scales) > 1 or self.opt.nms:
            #  results[j] = pnms(results[j],self.opt.nms_thresh)
        shape_num = 0
        for j in range(1, self.num_classes + 1):
            shape_num = shape_num + len(results[j])
        if shape_num != 0:
            # print(np.array(results[1]))
            scores = np.hstack(
                [results[j][:, 8] for j in range(1, self.num_classes + 1)])
        else:
            scores = []
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 8] >= thresh)
                results[j] = results[j][keep_inds]
        return results


    def run(self,image,meta=None):
        """
        image: np.ndarray
        """
        # load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        # merge_time, tot_time = 0, 0
        # start_time = time.time()
        # pre_processed = False
        # if isinstance(image_or_path_or_tensor, np.ndarray):
        #     image = image_or_path_or_tensor
        # elif type(image_or_path_or_tensor) == type(''):
        #     image = cv2.imread(image_or_path_or_tensor)
        # else:
        #     image = image_or_path_or_tensor['image'][0].numpy()
        #     pre_processed_images = image_or_path_or_tensor
        #     pre_processed = True

        # loaded_time = time.time()
        # load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:
            images, meta = self.preprocess(image, scale, meta)
            # scale_start_time = time.time()
            # if not pre_processed:
            #     images, meta = self.pre_process(image, scale, meta)
            # else:
            #     images = pre_processed_images['images'][scale][0]
            #     meta = pre_processed_images['meta'][scale]
            #     meta = {k: v.numpy()[0] for k, v in meta.items()}

            # import ipdb;ipdb.set_trace()
            # images = np.load('data.npy').astype(np.float32)
            # images = torch.from_numpy(images)
            
            images = images.to(self.device)
            torch.cuda.synchronize()
            # pre_process_time = time.time()
            # pre_time += pre_process_time - scale_start_time
            output, dets, dets_sub = self.process(images)
            torch.cuda.synchronize()
            # net_time += forward_time - pre_process_time
            # decode_time = time.time()
            # dec_time += decode_time - forward_time

            # if self.opt.debug >= 2:
            #     self.debug(debugger, images, dets, output, scale)

            dets, corner = self.postprocess(dets, corner, meta, scale)
            for j in range(1, self.num_classes + 1):
                dets[j] = self.Duplicate_removal(dets[j])
                
            # add sub
            dets_sub, corner = self.postprocess(dets_sub, corner, meta, scale)
            for j in range(1, self.num_classes + 1):
                dets_sub[j] = self.Duplicate_removal(dets_sub[j])
                
            # import ipdb;ipdb.set_trace()   
            torch.cuda.synchronize()
            # post_process_time = time.time()
            # post_time += post_process_time - decode_time
            dets[12] = dets_sub[12]
            dets[13] = dets_sub[13]
            detections.append(dets)

        results = self.merge_outputs(detections)
        torch.cuda.synchronize()

        # end_time = time.time()
        # merge_time += end_time - post_process_time
        # tot_time += end_time - start_time
        # import pdb; pdb.set_trace()
        # if self.opt.debug >= 1:
        #     if isinstance(image_or_path_or_tensor, str):
        #         image_name = os.path.basename(image_or_path_or_tensor)
        #     else:
        #         print("--> warning: use demo.py for a better visualization")
        #         image_name = "{}.jpg".format(time.time())
        #     self.show_results(debugger, image, results, corner, image_name)

        # return {'results': results, 'tot': tot_time, 'load': load_time,
        #         'pre': pre_time, 'net': net_time, 'dec': dec_time, 'corner': corner,
        #         'post': post_time, 'merge': merge_time, 'output': output}
        return {
            "results": results,
            "corner": corner,
            "output": output
        }




class DocLayoutReconize:
    def __init__(self, model_dir=None):
        assert model_dir != None,("model dir must be not None!")
        model_path = os.path.join(model_dir,"DocXLayout_231012.pth")
        self.detector = Detector(model_path)

    def __call__(self,image_list):
        """
        image_list: List[PIL.Image]
        """
        # task = ctdet_subfield
        for _,img in enumerate(image_list):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            result = self.detector.run(img)
            print(result)



    
