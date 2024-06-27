


import sys,os
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../"
    )
)

import time
import numpy as np
from PIL import Image
import cv2
import math
import json

from services.ocr import OCR
from services.tsr import Tsr
from services.layout.docx import DocLayoutReconize
from services.recovery.wrapper import wrap_result
from services.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx,merge_text_in_line
from services.recovery.recovery_to_html import convert_docx_to_html

class Recovery:
    def __init__(self,model_dir):
        self.__ocr = OCR(model_dir)
        self.__tsr = Tsr(model_dir)
        self.__layout_recog = DocLayoutReconize(model_dir)
        self.__category = self.__get_category()

        self.__ocr_type = [
            "header","figure caption","title","plain text","footer","table caption"
        ]
        self.__tb_type = [
            "table"
        ]
        self.__figure_type = [
            "figure"
        ]

    def __get_category(self):
        abspath = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(abspath,"map_info.json")
        map_info = json.load(open(filepath))
        category_map = {}
        for cate, idx in map_info["huntie"]["primary_map"].items():
            category_map[idx] = cate
        return category_map

    def __add_white_border(self,image, top_bottom, left_right):
        if isinstance(image,np.ndarray):
            height, width = image.shape[:2]
            new_height = height + 2*top_bottom
            new_width = width + 2*left_right
            bordered_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255  # 255代表白色
            start_y = top_bottom
            start_x = left_right
            bordered_image[start_y:start_y+height, start_x:start_x+width] = image
            return bordered_image
        else:
            width, height = image.size
            new_width = width + left_right * 2
            new_height = height + top_bottom * 2
            bordered_image = Image.new('RGB', (new_width, new_height), color='white')
            bordered_image.paste(image, (left_right, top_bottom))
            return bordered_image

    def __crop_img(self,raw_image,region,expand = False,add_border = True):
        if isinstance(raw_image,np.ndarray):
            h,w = raw_image.shape[:2]
        else:
            w,h = raw_image.size
        if expand:
            x=max(0,math.ceil(region[0][0])-5)
            t=max(0,math.ceil(region[0][1])-5)
            y=min(w,math.ceil(region[2][0])+5)
            b=min(h,math.ceil(region[2][1])+5)
        else:
            x=math.ceil(region[0][0])
            t=math.ceil(region[0][1])
            y=math.ceil(region[2][0])
            b=math.ceil(region[2][1])
        area = (x,t,y,b)
        if isinstance(raw_image,np.ndarray):
            raw_image = cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)
            raw_image = Image.fromarray(raw_image)

        crop_img = raw_image.crop(area)
        if add_border:
            crop_img = self.__add_white_border(crop_img,50,30)
        return crop_img
    
    def ocr_process(self,image,region,save_img=False):
        import copy
        crop_image = self.__crop_img(image,region,expand = True)
        cv_image = np.array(crop_image)
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR)
        
        resp = self.__ocr(cv_image)
        ret = []
        for info in resp:
            info = copy.deepcopy(info)
            info_dict = {
                "text_region":info[0],
                "text":info[1][0],
                "confidence":info[1][1],
            }
            ret.append(info_dict)
        
        if save_img == True:
            save_path = f"test_{time.time()}.jpg"
            cv2.imwrite(save_path,cv_image)
            print(f"save_path:{save_path}   ret:{resp}")

        if ret == []:
            return ""
        return ret

    def tsr_process(self,image,region):

        def get_boxes_recs(ocr_result, h, w) :
            def __ocr_res_adaptor(ocr_res):
                new_ocr_res = []
                for i in range(len(ocr_res)):
                    new_data = []
                    data = ocr_res[i]
                    new_data.append(data[0])
                    for d in data[1]:
                        new_data.append(d)
                    new_ocr_res.append(new_data)
                return new_ocr_res
            
            ocr_result = __ocr_res_adaptor(ocr_result)
            dt_boxes, rec_res, scores = list(zip(*ocr_result))
            rec_res = list(zip(rec_res, scores))
            r_boxes = []
            for box in dt_boxes:
                box = np.array(box)
                x_min = max(0, box[:, 0].min() - 1)
                x_max = min(w, box[:, 0].max() + 1)
                y_min = max(0, box[:, 1].min() - 1)
                y_max = min(h, box[:, 1].max() + 1)
                box = [x_min, y_min, x_max, y_max]
                r_boxes.append(box)
            dt_boxes = np.array(r_boxes)
            return dt_boxes, rec_res
        
        crop_image = self.__crop_img(image,region)
        crop_image.save("table.png")
        pred_structures, pred_bboxes,wh = self.__tsr(crop_image)
        ocr_ret = self.__ocr(np.array(crop_image.convert('RGB')))
        dt_boxes, rec_res = get_boxes_recs(ocr_ret, wh.get("h"), wh.get("w"))
        pred_html = self.__tsr.match_ocr(pred_structures, pred_bboxes, dt_boxes, rec_res)
        return pred_html

    def figure_process(self,image,info,region,save_foler=None,filename=None,img_idx = None):
        folder = os.path.join(save_foler,filename)
        if not os.path.exists(folder):
            os.makedirs(folder,exist_ok=True)
        bbox = [info['pts'][0],info['pts'][1],info['pts'][4],info['pts'][5]]
        image_path = f"{folder}/{bbox}_{img_idx}.jpg"
        crop_image = self.__crop_img(image,region,expand = True)
        # crop_image.save(image_path)
        cv_image = np.array(crop_image)
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path,cv_image)


    def __call__(self,image,filename,visual = True, html = True, save_folder = "./save"):
        # layout
        if isinstance(image,np.ndarray):
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(image,str):
            image = Image.open(image)
        else:
            # PIL.Image
            pass
        layout_res = self.__layout_recog([image])
        layout_info = wrap_result(layout_res, self.__category)
        layouts = layout_info['layouts']

        final_ret = []
        for info in layouts:
            region = [[info["pts"][0],info["pts"][1]],
                    [info["pts"][2],info["pts"][3]],
                    [info["pts"][4],info["pts"][5]],
                    [info["pts"][6],info["pts"][7]]]
            if info["category"] in self.__ocr_type:
                text = self.ocr_process(image,region,save_img=False)
                process_res = merge_text_in_line(text)
                new_info = {
                    "img_idx":0,
                    "type":info["category"],
                    "bbox":[info['pts'][0],info['pts'][1],info['pts'][4],info['pts'][5]],
                    "res":process_res
                }
                final_ret.append(new_info)
            elif info["category"] in self.__figure_type:
                img_idx = info.get('img_idx',0)
                _ = self.figure_process(image,info,region,save_folder,filename,img_idx=img_idx)
                new_info = {
                    "img_idx":img_idx,
                    "type":info["category"],
                    "bbox":[info['pts'][0],info['pts'][1],info['pts'][4],info['pts'][5]],
                    "res":{},
                }
                final_ret.append(new_info)
            elif info["category"] in self.__tb_type:
                tsr_ret = self.tsr_process(image,region)
                new_info = {
                    "img_idx":0,
                    "type":info["category"],
                    "bbox":[info['pts'][0],info['pts'][1],info['pts'][4],info['pts'][5]],
                    "res":{
                        "html":tsr_ret,
                    },
                }
                final_ret.append(new_info)
            else:
                raise ValueError(f"unsupport type:{info['category']}")
        
        img = np.array(image)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        _, w, _ = img.shape
        res = sorted_layout_boxes(final_ret, w)

        convert_info_docx(img, res, save_folder, filename)
        visual_path = None
        if visual:
            img = self.__layout_recog.draw_pic(img,layout_info)
            visual_path = f"./{save_folder}/{filename}/{filename}.jpg"
            cv2.imwrite(visual_path,img)
        html_path = None
        if html:
            html_path = f"./{save_folder}/{filename}/{filename}.html"
            convert_docx_to_html(f"./{save_folder}/{filename}/{filename}.docx",html_path)
        return {
            "doc": f"./{save_folder}/{filename}/{filename}.docx",
            "visual": visual_path,
            "html": html_path
        }


if __name__ == "__main__":
    model_dir = "/workspaces/GIES/apps/services/ocr/models"
    image_path = "/workspaces/GIES/static/paper6.png"
    filename = os.path.basename(image_path).split('.')[0]
    recv = Recovery(model_dir)

    image = Image.open(image_path)
    resp = recv(image,filename,True)
    print(resp)