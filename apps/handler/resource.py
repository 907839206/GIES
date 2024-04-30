
from werkzeug.utils import secure_filename

from .base import BaseHandler

from constant import CodeEnum,iglobal
from services.utils import gen_random_str
from setting import config

class ResourceHandler(BaseHandler):

    allow_pic = [".jpg","jpeg","png"]
    allow_doc = [".doc","docx"]
    allow_pdf = [".pdf"]

    @classmethod
    def handler(cls,request):
        if 'image_file' not in request.files:
            return {"ec":CodeEnum.Fail,"em":"no file in part","data":{}}
        uploaded_file = request.files['image_file']
        if uploaded_file.filename == '':
            return {"ec":CodeEnum.Fail,"em":"no file select","data":{}}

        filename = secure_filename(uploaded_file.filename)
        filename,ext =  os.path.splitext(filename)
        if ext not in cls.allow_doc+cls.allow_pdf.cls.allow_pic:
            return {"ec":CodeEnum.Fail,"em":"not support file type","data":{}}
        
        image_id = gen_random_str(8)
        full_save_path = os.path.join(
            config.FILESTORE.LOCALPATH.RAWPATH,
            f"{filename}-{image_id}.{fileext}"
        )
        uploaded_file.save(full_save_path)
        # 先不考虑异步处理了，直接同步吧
        # task = {
        #     "task_id": gen_random_str(8)
        #     "image_id": image_id,
        #     "image_path":full_save_path
        # }
        # iglobal.put(task)
        return {"ec":CodeEnum.SUCCESS,"em":"success","data":{
            "image_id":image_id
        }}
