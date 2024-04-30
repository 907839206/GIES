from flask import Flask,request,jsonify

from handler import ResourceHandler,InformationHandler

app = Flask(__name__)


def initialize():
    pass


@app.route("/v1/resource/upload")
def upload():
    """ 
    上传资源:
    1）保存资源； 
    """
    response = ResourceHandler.handler(request)
    return jsonify(response)


@app.route("/v1/rule/execute",methods=["POST"])
def execute():
    """ 
    业务逻辑执行：
    1）OCR
    2) 提取Embedding
    3) 存储ES
    4) 对ES进行检索
    5）请求LLM
    6）业务规则判断
    """
    pass

@app.route("/v1/information/extract",methods=["POST"])
def extract():
    """
    信息抽取：
    1）OCR
    2) 提取Embedding
    3) 存储ES
    4) 对ES进行检索
    5）请求LLM
    """
    response = InformationHandler.handler(request)
    return jsonify(response)


if __name__ == "__main__":
    initialize()
    app.run()
