# pip install Spire.Doc

import base64
import os,re
import shutil

from spire.doc import *
from bs4 import BeautifulSoup

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


def remove_warning(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')
    warning_text = "Evaluation Warning: The document was created with Spire.Doc for Python."
    for tag in soup.find_all(string=lambda text: warning_text in text if text and isinstance(text, str) else False):
        if tag.parent.name != 'script' and tag.parent.name != 'style':  # 过滤掉<script>和<style>标签内的内容
            tag.extract()  # extract()方法会移除标签及其内容（如果是文本节点，则直接移除文本）

    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(str(soup.prettify()))

def convert_docx_to_html(filepath,savepath):
    document = Document()
    document.LoadFromFile(filepath)
    document.SaveToFile(savepath, FileFormat.Html)
    document.Close()
    # 调用函数将图片嵌入HTML
    embed_images_in_html(savepath)

    # 清理中间文件
    filedir,filename = os.path.split(savepath)
    filename = filename.split(".")[0]
    image_dir = os.path.join(filedir,f"{filename}_images")
    css_path = os.path.join(filedir,f"{filename}_styles.css")
    if os.path.isdir(image_dir):
        shutil.rmtree(image_dir)
    if os.path.isfile(css_path):
        os.remove(css_path)

    remove_warning(savepath)
    
if __name__=="__main__":
    filepath = "/workspaces/GIES/apps/services/recovery/save/paper3/paper3.docx"
    savepath = "./save/paper2.html"
    convert_docx_to_html(filepath,savepath)
