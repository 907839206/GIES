
from .tools import calc_avg, calc_max, calc_min


func_mapping ={
    "calc_avg":{
        "func":calc_avg,
        "desc":"计算平均值"
    },
    "calc_max":{
        "func":calc_max,
        "desc":"计算最大值"
    },
    "calc_min":{
        "func":calc_min,
        "desc":"计算最小值"
    }
}

func_desc = [
    {
        "type": "function",
        "function": {
            "name": "calc_avg",
            "description": "计算所有输入数据平均值",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "description": "输入数据",
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                    }
                },
                "required": ["data"]
            }
        }
    },{
        "type": "function",
        "function": {
            "name": "calc_max",
            "description": "计算所有输入数据最大值",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "description": "输入数据",
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                    }
                },
                "required": ["data"]
            }
        }
    },{
        "type": "function",
        "function": {
            "name": "calc_min",
            "description": "计算所有输入数据最小值",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "description": "输入数据",
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                    }
                },
                "required": ["data"]
            }
        }
    }
]


__all__ = [
    "func_mapping",
    "func_desc"
]