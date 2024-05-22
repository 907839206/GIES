

class DefaultPrompt:

    InformationExtractPrompt = {
        "system": """你是一个智能助手，请总结知识库的内容来回答问题，请列举知识库中的数据详细回答。当所有知识库内容都与问题无关时，你的回答必须包括“知识库中未找到您要的答案！”这句话。回答需要考虑聊天历史。
以下是知识库：
{knowledge}
以上是知识库。""",
        "prologue": "你好！ 我是你的助理，有什么可以帮到你的吗？",
        "querylogue": "\n请你帮我从知识库中提取{fields}字段的内容，并按照json格式返回。知识库可能会没有明确展示出待提取字段，此时可以在知识库的基础上进行合理总结，但不要返回与结果无关的其他内容和任何的解释性语句。请注意：如果字段没有在知识库中出现，则此字段的值返回为null；对于大写数字（如金额等），请将其转换为阿拉伯数字再返回。",
        "empty_response": "Sorry! 知识库中未找到相关内容！"
    }


    PlanToActPrompt = """请你仔细分析给定的输入的数据和问题，并一步步列出详细分析步骤并输出。以下是示例：

输入：{\"data\":[1,5,8,3,9,10],\"question\":\"所有偶数的和是多少？\"}
输出：
    1. 筛选输入数据中的偶数元素；
    2. 计算筛选出来的偶数元素的和；
    3. 返回结果。

输入：{\"data\":\"小明比小李大3岁，小红的年龄是小李的一半，小红明年15岁\",\"question\":\"小明今年多少岁？\"}
输出：
    1. 计算小红今年的年龄；
    2. 计算小李今年的年龄；
    3. 计算小明今年的年龄；
    4. 返回结果。

请注意，仅需要列出解决问题的步骤即可，不需要执行。每个步骤必须语义清晰、条件完整明确，且无其他歧义！

开始！
"""
    PlanStepAct = """
## 任务信息
你是一个强大的人工智能助手，可以帮助用户完成很多事情。 
我们根据用户的原始需求，制定了执行计划，共包含{step}个任务，你需要完成其中的第{cur_task_idx}个任务：{cur_task}，并输出结果，需要考虑已执行完成任务的结果。

## 背景信息
为了帮助你理解整个执行计划，输出当准确结果，我们给出了当前任务的背景信息，包括用户原始需求、已执行的任务和结果：
- 用户原始需求：
{raw_requirement}

- 已执行任务信息：
{finished_task}

## 可用工具
有以下函数的描述：
{function_desc}
请判断现有工具/函数是否适用于处理当前任务，如果工具/函数满足处理条件，请优先使用，并以下面的格式：
```json
{
    "type": "function_calling",
    "functions":[
        {
            "function_name": 工具/函数名称，
            "arguments": 可以直接输入的参数
        },...
    ],
    "result":"",
}
```
返回所有工具/函数名称和参数；如果现有工具无法适用，请你仔细分析当前任务并直接给出自己处理的结果，确保返回的结果严格准确，不要包含任何解释性的语句，并以下面的格式：
```json
{
    "type": "text_answer",
    "functions": [],
    "result": 你的回答,
}
```
返回结果。

开始吧！
"""

prompt_default = DefaultPrompt()