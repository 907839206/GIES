# pip install Spire.Doc

import base64
import os,re
from spire.doc import *

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def embed_images_in_html(html_path):
    with open(html_path, "r+", encoding="utf-8") as file:
        html_content = file.read()
        for img_match in re.findall(r'<img[^>]+src="([^"]+)"', html_content):
            img_path = os.path.join(os.path.dirname(html_path), img_match)
            if os.path.exists(img_path):
                img_base64 = convert_image_to_base64(img_path)
                img_tag_pattern = r'src="{}"'.format(re.escape(img_match)) 
                html_content = re.sub(img_tag_pattern, f'src="data:image/png;base64,{img_base64}"', html_content)
        file.seek(0)
        file.write(html_content)
        file.truncate()


def convert_docx_to_html(filepath,savepath):
    # 使用Spire.Doc转换Word到HTML
    document = Document()
    document.LoadFromFile(filepath)
    document.SaveToFile(savepath, FileFormat.Html)
    document.Close()
    # 调用函数将图片嵌入HTML
    embed_images_in_html(savepath)

if __name__=="__main__":
    filepath = "./save/paper2_ocr.docx"
    savepath = "./save/paper2.html"
    convert_docx_to_html(filepath,savepath)
