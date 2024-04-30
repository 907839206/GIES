import sys,os
import copy
import time,traceback
from multiprocessing import Queue,Process
from threading import Thread
from queue import Queue as tQueue

import gradio as gr
import cv2
import uuid

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "apps"
    )
)
from handler import InformationHandler
from services.utils import gen_uuid


class Request:
    def __init__(self,rid,fid, fpth, extract_fields):
        self.rid = rid
        self.fid = fid
        self.fpth = fpth
        self.extract_fields = extract_fields

    def get_json(self):
        return {
            "fid": self.fid,
            "fpth": self.fpth,
            "extract_fields": self.extract_fields
        }


class Executor:

    task_queue = Queue()
    result_queue = Queue()
    recorder = {}

    execute_process_num = 5
    timeout = 10 * 1000


    @classmethod
    def init(cls,*args,**kwargs):
        cls._tear_down()
        cls.dispatcher()
        cls._setup()
        
    @staticmethod
    def url_fn():
        return ""

    @staticmethod
    def extract_fn(select_img,local_files,extract_fields):
        if len(extract_fields) == 0:
            gr.Warning("请输入待提取的字段信息！")
            return """
            ```json
            {}
            """,local_files,extract_fields
        if local_files is not None and len(local_files) != 0:
            gr.Info("开始对上传图片进行信息提取！")
            return Executor.process_upload_fn(local_files,
                                            extract_fields),[],extract_fields
        elif select_img is not None:
            gr.Info("开始对选中图片进行信息抽取！")
            return Executor.process_select_fn(select_img,extract_fields),[],extract_fields
        else:
            gr.Warning("未选中任何图片，请首先选中/上传图片文件！")
            return """
            ```json
            {}
            """,local_files,extract_fields

    @classmethod
    def process_upload_fn(cls, local_files, extract_fields):
        def _get_fid(filepath):
            return os.path.split(
                os.path.dirname(filepath)
            )[-1]

        _request_id = gen_uuid()
        cls.recorder[_request_id] ={
            "_st": time.time(),
            "done": False,
            "task":{},
            "task_size":len(local_files),
            "ready":tQueue()
        }
        for _file in local_files:
            _fid = _get_fid(_file)
            _fpth = _file
            _extract_fields = extract_fields
            _task = Request(_request_id,_fid, _fpth, _extract_fields)
            cls.task_queue.put(_task)
            cls.recorder[_request_id]["task"][_fid] = tQueue()
        print(f"[INFO] process_upload_fn get task local_files:{local_files}"
             "extract_fields:{extract_fields}")
        while (cls.recorder.get(_request_id,None) and  
                time.time() - cls.recorder.get(_request_id)["_st"] < 
                    cls.timeout+100):
            _request_info = cls.recorder.get(_request_id)
            print(f"[DEBUG] _request_info:{_request_info}")
            if not _request_info["done"]:
                time.sleep(1)
            else:
                _all_task = _request_info["task"]
                _response = ""
                for _,v in _all_task.items():
                    _response += ("\n" + v.get().get("result",""))
                return _response
            time.sleep(1)
        return "Timeout"

    @classmethod
    def process_select_fn(cls,img,extract_fields):
        print(f"[DEBUG] process_select_fn type img:{type(img)} extract_fields:{extract_fields}")
        _fid = gen_uuid()
        _rid = gen_uuid()
        _fpth = f"/tmp/gradio/{_fid}.jpg"
        cv2.imwrite(_fpth,img[:,:,::-1])
        cls.recorder[_rid] ={"_st": time.time(),
                                    "done": False,
                                    "task":{},
                                    "task_size":1,
                                    "ready":tQueue()}
        _task = Request(_rid,_fid, _fpth, extract_fields)
        print(f"[INFO] add task: {_task.get_json()}")
        cls.task_queue.put(_task)
        cls.recorder[_rid]["task"][_fid] = tQueue()
        while (cls.recorder.get(_rid,None) and  
                time.time() - cls.recorder.get(_rid)["_st"] < 
                    cls.timeout+100):
            _request_info = cls.recorder.get(_rid)
            print(f"[DEBUG] _request_info:{_request_info}")
            if not _request_info["done"]:
                time.sleep(1)
            else:
                _all_task = _request_info["task"]
                _response = ""
                for _,v in _all_task.items():
                    _response += ("\n" + v.get().get("result",""))
                return _response
            time.sleep(1)
        return "Timeout"

    @classmethod
    def _setup(cls):

        def _process(task_queue, result_queue):
            while True:
                task = task_queue.get()
                if not task:
                    time.sleep(3)
                try:
                    _result = InformationHandler.handler(task)[0]
                    _rid = task.rid
                    _fid = task.fid
                    result = {
                        "rid": _rid,
                        "fid": _fid,
                        "result": _result
                    }
                    result_queue.put(result)
                    print(f"[INFO] process finished, result:{_result}")
                except Exception as e:
                    print(f"[ERROR] error: {traceback.format_exc()}")
                    time.sleep(1)

        processes = []
        for _ in range(cls.execute_process_num):
            _p = Process(target = _process,args=(cls.task_queue,
                                                 cls.result_queue))
            _p.start()
            processes.append(_p)
        print("[INFO] all setup subprocess is working...")

    @classmethod
    def _tear_down(cls):

        def _clean(records, timeout):
            while True:
                try:
                    _record_keys = list(records.keys())
                    for _rid in _record_keys:
                        _task = records[_rid]
                        _st = _task.get("_st")
                        _during_time = time.time() - _st
                        if _during_time > timeout:
                            print(f"[ERROR] {_rid} timeout, during time:{_during_time}")
                            del cls.recorder[_rid]
                except Exception as e:
                    print(f"[ERROR] _clean error, msg:{traceback.format_exc()}")
                    continue
                finally:
                    time.sleep(1)
        _t = Thread(target = _clean,args = (cls.recorder, cls.timeout))
        _t.start()
        print("[INFO] tear down thread is working...")

    @classmethod
    def dispatcher(cls):
        def _dispatch():
            while True:
                try:
                    _result = cls.result_queue.get()
                    _rid = _result.get("rid")
                    _fid = _result.get("fid")
                    info = cls.recorder.get(_rid,None)
                    if not info:
                        print(f"[INFO] request:{_rid} may be timeout?")
                        continue
                    
                    info["task"][_fid].put(_result)
                    info["ready"].put(_fid)
                    if info["task_size"] == info["ready"].qsize():
                        info["done"] = True
                except Exception as e:
                    print(f"[ERROR] {traceback.format_exc()}")
                finally:
                    time.sleep(1)
        _t = Thread(target = _dispatch)
        _t.start()
        print("[INFO] dispatcher thread is working...") 