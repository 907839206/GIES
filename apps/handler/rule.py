import re
import copy
import json
import traceback
from collections import OrderedDict



from .base import BaseHandler

from llm import prompt_default,ModelService
from constant import iglobal
from utils import generate_uuid,calculate_md5
from tools import func_desc,func_mapping



class RuleHandler(BaseHandler):

    support_ruler = [
        "loan",
        "general"
    ]

    llm_service = ModelService()

    @staticmethod
    def handler(request):
        _task = request.get_json()
        print(f"ruler get task:{_task}")
        _model_name = _task.get("model","qwen-plus-oai")
        _question = _task.get("question",None)
        _ruler_name = _task.get("ruler_name",None)
        if not _question or _ruler_name not in RuleHandler.support_ruler:
            return "问题为空或不支持的规则"
        if _model_name not in iglobal.supported_model_mapping.keys():
            return "不支持的模型！"
        if _ruler_name == "general":
            for info in RuleHandler._GeneralRulerProcess(_question,_model_name):
                yield info
        elif _ruler_name == "":
            pass
        else:
            return "不支持的规则！"


    @classmethod
    def _GeneralRulerProcess(cls,content,model):
        """ Use Plan-2-Act processing """
        # Write a Plan
        plan_prompt = prompt_default.PlanToActPrompt
        messages = [{'role': 'system', 'content': plan_prompt},
                    {'role': 'user', 'content': content}]
        yield "开始制定计划...<br>"

        _model_client = iglobal.supported_model.get(model)
        # print(f"model:{model}  _model_client:{_model_client}")
        model = iglobal.supported_model_mapping.get(model)
        gen_conf = {
            "temperature":0.001,
            "model":model
        }
        response = cls.llm_service.chat(_model_client,plan_prompt, messages, **gen_conf)
        _plans = cls.extract_plans(response.choices[0].message.content)
        _plan_info = OrderedDict()
        for _task in _plans:
            _task_id = generate_uuid()
            _plan_info[_task_id] = {
                "task":_task,
                "result":None
            }
        _all_plan ="制定的计划如下："
        for _idx,_plan in enumerate(_plans):
            _all_plan += "<br>" + f"{_idx+1}. {_plan}"
        yield _all_plan + "<br>"

        # Act
        _finished_task = []
        for _idx,_task in enumerate(_plans):
            _step_prompt = copy.deepcopy(prompt_default.PlanStepAct)
            _step_prompt = _step_prompt.replace("{step}",str(len(_plans)))
            _step_prompt = _step_prompt.replace("{cur_task_idx}",str(_idx+1))
            _step_prompt = _step_prompt.replace("{cur_task}",_task)
            _step_prompt = _step_prompt.replace("{raw_requirement}",content)
            _step_prompt = _step_prompt.replace("{function_desc}",json.dumps(func_desc,ensure_ascii=False))

            _finished_task_info = ""
            for _task_info in _finished_task:
                if not isinstance(_task_info["result"],str):
                    _result = json.dumps(_task_info["result"],ensure_ascii=False)
                else:
                    _result = _task_info["result"]
                _tmp = f"任务：{_task_info['task']}\n结果：{_result}\n"
                _finished_task_info += _tmp
            _step_prompt = _step_prompt.replace("{finished_task}",_finished_task_info)
            messages = [
                {'role': 'user', 'content': _task}
            ]
            yield f"开始执行第{_idx+1}步：{_task}..."
            # response = client.chat.completions.create(
            #     model=model,
            #     messages = messages,
            #     temperature=0,
            # )
            ans = cls.llm_service.chat(_model_client,_step_prompt, messages, **gen_conf)
            answer_json = cls.extract_answer(ans.choices[0].message.content)
            if isinstance(answer_json,str):
                # plan_info[cur_task_id]["result"]=answer_json
                _finished_task.append(
                    {
                        "task": _task,
                        "result": answer_json
                     }
                )
            elif isinstance(answer_json,dict):
                answer_type = answer_json.get("type",None)
                if not answer_type:
                    _finished_task.append(
                        {
                            "task": _task,
                            "result": answer_json
                        }
                    )
                elif answer_type == "text_answer":
                    # 直接回答
                    _finished_task.append(
                        {
                            "task": _task,
                            "result": answer_json
                        }
                    )
                elif answer_type=="function_calling":
                    # 需要调用工具
                    tool_calls = answer_json["functions"]
                    call_result = ""
                    use_raw_result = False
                    deduplication = {}
                    for tool in tool_calls:
                        yield f"开始调用工具: {tool['function_name']}进行处理..."
                        info = tool["function_name"] + json.dumps(tool["arguments"])
                        key = calculate_md5(info)
                        if  key in deduplication.keys():
                            yield f"工具: {tool['function_name']}重复！"
                            continue
                        if tool["function_name"] not in func_mapping.keys():
                            # logger.error("---"*5+f" 工具:{tool['function_name']}不存在！ "+"---"*5)
                            # logger.info(f"will using raw data as result:{answer_json['result']}")
                            yield f"工具: {tool['function_name']}不存在，将直接使用返回结果进行后续步骤！"
                            use_raw_result = True
                            break
                        deduplication[key] = True
                        func = func_mapping[tool['function_name']]["func"]
                        arguments = tool["arguments"]
                        result = func(arguments["data"])
                        result = f"{func_mapping[tool['function_name']]['desc']}的结果是:{result}"
                        call_result += "\n" + result if call_result !="" else result
                        # logger.info("---"*5+f" 工具:{tool['function_name']}计算结果:{result} "+"---"*5)
                        yield f"工具: {tool['function_name']}计算结束，结果: {result}"
                    if use_raw_result:
                        _finished_task.append(
                            {
                                "task": _task,
                                "result": answer_json["result"]
                            }
                        )
                    else:
                        _finished_task.append(
                            {
                                "task": _task,
                                "result": call_result
                            }
                        )
                    # plan_info[cur_task_id]["result"]=call_result if not use_raw_result else answer_json["result"]
                else:
                    # logger.error(response.choices[0].to_dict())
                    yield f"不支持的返回类型: {response.choices[0].finish_reason} <br> response: {response.choices[0].to_dict()}"
                    # raise ValueError(f"unrecognized function type:{response.choices[0].finish_reason}")
            elif isinstance(answer_json,list):
                # plan_info[cur_task_id]["result"]=json.dumps(answer_json,ensure_ascii=False)
                _finished_task.append(
                    {
                        "task": _task,
                        "result": json.dumps(answer_json,ensure_ascii=False)
                    }
                )
            else:
                # logger.error(f"content: {response.choices[0].message.content}")
                # logger.error(f"answer_json: {answer_json}")
                yield f"不可识别的类型: {{type(answer_json)}} <br>response:{response.choices[0].message.content}"
                # raise ValueError(f"unrecognized type:{type(answer_json)}")
            # logger.info(plan_info[cur_task_id]["result"])
            # logger.info("==="*5+" 执行结束！"+"==="*5)
            yield f"第{_idx+1}步执行结束，结果: {_finished_task[-1]['result']} <br>"
        yield "所有步骤执行结束！"

    @classmethod
    def extract_answer(cls,content):
        counter = cls.get_target_str_count(content,'```')
        if counter == 1:
            content = content.replace("```","")
        elif counter == 0:
            try:
                return json.loads(content)
            except Exception as e:
                print(traceback.format_exc())
                return content
        else:
            match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if match:
                content = match.group(1)
                content = re.sub(r',\s*(\]|\})\s*$', r'\1', content)
                try:
                    return json.loads(content)
                except Exception as e:
                    data_fixed = re.sub(r'\\\"', r'"', content)
                    data_fixed = re.sub(r'\"\[{', r'[{', data_fixed)
                    data_fixed = re.sub(r'}\]\"', r'}]', data_fixed)
                    if cls.check_trailing_comma(data_fixed):
                        return json.loads(cls.remove_trailing_comma(data_fixed))
                    return json.loads(data_fixed)
            else:
                try:
                    return json.loads(content)
                except Exception as e:
                    print(traceback.format_exc())
                    return content

    @classmethod
    def get_target_str_count(cls,data,target):
        count = 0
        start = 0
        while True:
            start = data.find(target, start)
            if start == -1:
                break
            count += 1
            start += len(target)
        return count
    
    @classmethod
    def check_trailing_comma(cls,json_string):
        if json_string.endswith(',\n}') or json_string.endswith('},') or json_string.endswith('},\n'):
            return True
        return False

    @classmethod
    def remove_trailing_comma(cls,json_string):
        if json_string.endswith(',\n}'):
            corrected_json_string = json_string[:-3] + '}'
        elif json_string.endswith('},'):
            corrected_json_string = json_string[:-2]
        elif json_string.endswith('},\n'):
            corrected_json_string = json_string[:-4] + '}'
        else:
            corrected_json_string = json_string
        return corrected_json_string

    @classmethod
    def extract_plans(cls, content):
        # 正则表达式模式匹配每个步骤的描述
        pattern = re.compile(r'\d+\.\s(.*?)(?=；|。)')
        matches = pattern.findall(content)
        return matches

        
