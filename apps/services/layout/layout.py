
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
from copy import deepcopy

import sys
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../"
    )
)


from setting import config
from services.ocr import load_model
from services.ocr.operators import *
from services.utils import get_project_path

class LayoutRecognize:

    def __init__(self,model_dir=None):
        assert model_dir!=None,"model_dir must be not None!"
        self.predictor,_= load_model(model_dir,"layout")
        self.input_names = [node.name for node in self.predictor.get_inputs()]
        self.output_names = [node.name for node in self.predictor.get_outputs()]
        self.input_shape = self.predictor.get_inputs()[0].shape[2:4]
        self.label_list = [
            "_background_",
            "Text",
            "Title",
            "Figure",
            "Figure caption",
            "Table",
            "Table caption",
            "Header",
            "Footer",
            "Reference",
            "Equation"
        ]

    def preprocess(self, image_list):
        inputs = []
        if "scale_factor" in self.input_names:
            preprocess_ops = []
            for op_info in [
                {
                    'interp': 2, 'keep_ratio': False, 'target_size': [800, 608], 
                    'type': 'LinearResize'
                },{
                    'is_scale': True, 'mean': [0.485, 0.456, 0.406], 
                    'std': [0.229, 0.224, 0.225], 'type': 'StandardizeImage'
                },{
                    'type': 'Permute'
                },{
                    'stride': 32, 'type': 'PadStride'
                }
            ]:
                new_op_info = op_info.copy()
                op_type = new_op_info.pop('type')
                preprocess_ops.append(eval(op_type)(**new_op_info))

            for im_path in image_list:
                im, im_info = preprocess(im_path, preprocess_ops)
                inputs.append({"image": np.array((im,)).astype('float32'),
                               "scale_factor": np.array((im_info["scale_factor"],)).astype('float32')})
        else:
            hh, ww = self.input_shape
            for img in image_list:
                h, w = img.shape[:2]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(np.array(img).astype('float32'), (ww, hh))
                # Scale input pixel values to 0 to 1
                img /= 255.0
                img = img.transpose(2, 0, 1)
                img = img[np.newaxis, :, :, :].astype(np.float32)
                inputs.append({self.input_names[0]: img, "scale_factor": [w/ww, h/hh]})
        return inputs

    def postprocess(self, boxes, inputs, thr):
        if "scale_factor" in self.input_names:
            bb = []
            for b in boxes:
                clsid, bbox, score = int(b[0]), b[2:], b[1]
                if score < thr:
                    continue
                if clsid >= len(self.label_list):
                    continue
                bb.append({
                    "type": self.label_list[clsid].lower(),
                    "bbox": [float(t) for t in bbox.tolist()],
                    "score": float(score)
                })
            return bb

        def xywh2xyxy(x):
            # [x, y, w, h] to [x1, y1, x2, y2]
            y = np.copy(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            return y

        def compute_iou(box, boxes):
            # Compute xmin, ymin, xmax, ymax for both boxes
            xmin = np.maximum(box[0], boxes[:, 0])
            ymin = np.maximum(box[1], boxes[:, 1])
            xmax = np.minimum(box[2], boxes[:, 2])
            ymax = np.minimum(box[3], boxes[:, 3])

            # Compute intersection area
            intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

            # Compute union area
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            union_area = box_area + boxes_area - intersection_area

            # Compute IoU
            iou = intersection_area / union_area

            return iou

        def iou_filter(boxes, scores, iou_threshold):
            sorted_indices = np.argsort(scores)[::-1]

            keep_boxes = []
            while sorted_indices.size > 0:
                # Pick the last box
                box_id = sorted_indices[0]
                keep_boxes.append(box_id)

                # Compute IoU of the picked box with the rest
                ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

                # Remove boxes with IoU over the threshold
                keep_indices = np.where(ious < iou_threshold)[0]

                # print(keep_indices.shape, sorted_indices.shape)
                sorted_indices = sorted_indices[keep_indices + 1]

            return keep_boxes

        boxes = np.squeeze(boxes).T
        # Filter out object confidence scores below threshold
        scores = np.max(boxes[:, 4:], axis=1)
        print(f"thr:{thr}")
        boxes = boxes[scores > thr, :]
        scores = scores[scores > thr]
        if len(boxes) == 0: return []

        # Get the class with the highest confidence
        class_ids = np.argmax(boxes[:, 4:], axis=1)
        boxes = boxes[:, :4]
        input_shape = np.array([inputs["scale_factor"][0], 
                                inputs["scale_factor"][1], 
                                inputs["scale_factor"][0], 
                                inputs["scale_factor"][1]])
        boxes = np.multiply(boxes, input_shape, dtype=np.float32)
        boxes = xywh2xyxy(boxes)

        unique_class_ids = np.unique(class_ids)
        indices = []
        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices, :]
            class_scores = scores[class_indices]
            class_keep_boxes = iou_filter(class_boxes, class_scores, 0.2)
            indices.extend(class_indices[class_keep_boxes])

        return [{
            "type": self.label_list[class_ids[i]].lower(),
            "bbox": [float(t) for t in boxes[i].tolist()],
            "score": float(scores[i])
        } for i in indices]


    def __call__(self, image_list, thr=0.5, batch_size=16):
        res = []
        imgs = []
        for i in range(len(image_list)):
            if not isinstance(image_list[i], np.ndarray):
                imgs.append(np.array(image_list[i]))
            else: imgs.append(image_list[i])

        batch_loop_cnt = math.ceil(float(len(imgs)) / batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(imgs))
            batch_image_list = imgs[start_index:end_index]
            inputs = self.preprocess(batch_image_list)
            for ins in inputs:
                bb = self.postprocess(self.predictor.run(None, 
                                                        {k:v for k,v in ins.items() if k in self.input_names})[0], 
                                    ins, thr)
                res.append(bb)

        return res

    def mask_entity(self,image_list,res_list,threshold = 0.5,entity=["table","figure"]):
        image_list = deepcopy(image_list)
        for idx,(image,res) in enumerate(list(zip(image_list,res_list))):
            for _,info in enumerate(res):
                if info["type"] in entity and info["score"] > threshold:
                    # mask
                    draw = ImageDraw.Draw(image)
                    x,t,y,b = info["bbox"]
                    draw.rectangle([math.ceil(x), math.ceil(t), math.ceil(y), math.ceil(b)], fill=(255, 255, 255))
                    image_list[idx] = image
        return image_list
    
    def crop_tables(self,image_list,res_list,threshold = 0.5):
        table_list = []
        for idx, (image,res) in enumerate(list(zip(image_list,res_list))):
            image_table_list = []
            for _,info in enumerate(res):
                if info["type"] == "table" and info["score"] > threshold:
                    copy_img = deepcopy(image)
                    x,t,y,b = info["bbox"]
                    area = (math.ceil(x),math.ceil(t),math.ceil(y),math.ceil(b))
                    crop_img = copy_img.crop(area)
                    image_table_list.append(crop_img)
            table_list.append(image_table_list)
        return table_list

import random,math
def draw_bbox(image, bbox,label, color=(0, 255, 0), thickness=2,font_scale=0.5, font=cv2.FONT_HERSHEY_SIMPLEX):
  print(f"bbox:{bbox}")
  x0, y0, x1, y1 = bbox
  x0=int(x0)
  y0=int(y0)
  x1=int(x1)
  y1=int(y1)

  cv2.rectangle(image, (math.ceil(x0), math.ceil(y0)), (math.ceil(x1), math.ceil(y1)), color, thickness)
  cv2.putText(image, label, (x1-random.randint(0, (x1-x0)), y0 - 5), font, font_scale, color, thickness)
  return image

if __name__ == "__main__":
    filepath = "/workspaces/GIES/static/paper6.png"
    _model_path = os.path.join(get_project_path(),config.OCR.MODEL_PATH)
    model = LayoutRecognize(_model_path)
    image_list = [
        Image.open(filepath)
    ]

    res_list = model(image_list)

    img2 = np.array(image_list[0])
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    
    for _,info in enumerate(res_list):
        for _info in info:
            img2 = draw_bbox(img2,_info['bbox'],_info['type'])
            print(_info)
    cv2.imwrite("rectangle.jpg",img2)
    
    mask_image_list = model.mask_entity(image_list,res_list)
    for idx, img in enumerate(mask_image_list):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        filesave = f"save_{idx}.jpg"
        img.save(filesave)

    crop_image_list = model.crop_tables(image_list,res_list)
    for idx, img in enumerate(crop_image_list):
        for jdx,_img in enumerate(img):
            if _img.mode == 'RGBA':
                _img = _img.convert('RGB')
            filesave = f"save_crop_{idx}_{jdx}.jpg"
            _img.save(filesave)