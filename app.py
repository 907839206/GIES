import numpy as np
import gradio as gr

from executor import Executor,colab_env

Executor.init()
demo = gr.Blocks(css = "./static/custom.css")

MODELS = ["qwen-plus-oai","qwen-max-oai","gpt-3.5-turbo"]

with demo:
    gr.Markdown("""
    ### <h2><span style="color: #e8313e;">欢迎体验GIES智能信息提取系统</span></h2>\n
    """)
    with gr.Tab("信息抽取"):
        with gr.Row():
            with gr.Column():
                img = gr.Image("static/2.jpg",
                                sources=None,
                                show_share_button=False,
                                interactive=False)
                with gr.Row():
                    gr.Examples(["static/2.jpg",
                                "static/11.png",
                                "static/12.png",
                                "static/4.jpg",
                                "static/13.jpg",
                                "static/14.webp",
                                "static/7.jpg",
                                "static/15.webp",
                                "static/8.jpg"], 
                            img,label=None)
                    localFiles = gr.components.File(label="Upload image",scale=1,file_count="multiple")
            
            with gr.Column():
                with gr.Row():
                    _ = gr.Markdown("""
                    ### <h3><span style="color: #e8313e;">关键字段提取：</span></h3>
                    """)
                with gr.Row():
                    extractFields = gr.Textbox(
                        placeholder="输入待提取字段名称,以“;”分隔,例如：姓名;月收入;年收入",
                        scale=11)
                    extractBtn = gr.Button("开始提取",scale=1)

                with gr.Row():
                    _ = gr.Markdown("""
                    ### <h3><span style="color: #e8313e;">提取结果：</span></h3>
                    """)
                textbox = gr.Markdown("{}".format("""
                ```json
                {}
                """))

                with gr.Row():
                    _ = gr.Markdown("""
                    ### <h3><span style="color: #e8313e;">业务规则：</span></h3>
                    """)
                with gr.Row():
                    # checkFiels = gr.Textbox(
                    #     placeholder = "请输入已提取字段", scale=3)
                    # checkOp = gr.Dropdown([">", "<", "="],interactive=True,scale=1)
                    # checkValue = gr.Textbox(
                    #     placeholder = "请输入判断值", scale=3)
                    orderValue = gr.Textbox(
                        placeholder = "请输入贷款月供", scale=11)
                    checkRatio = gr.Textbox(
                        placeholder = "月收入/贷款月供 最低比例", scale=11)
                    checkBtn = gr.Button("贷款审批",scale=1)

                with gr.Row():
                    checkRets = gr.DataFrame(interactive=False,
                                            wrap=True,
                                            headers=None)

            extractBtn.click(Executor.extract_fn,
                            inputs = [img,localFiles,extractFields],
                            outputs = [textbox,localFiles,extractFields])
            # checkBtn.click(Executor.check_fn,
            #                inputs = [textbox,checkFiels,checkOp,checkValue],
            #                outputs = [checkRets])
            checkBtn.click(Executor.check_income_fn,
                        inputs = [textbox,orderValue,checkRatio],
                        outputs = [checkRets])
    with gr.Tab("业务处理"):
        with gr.Row():
            with gr.Column():
                # 数据选择
                with gr.Row():
                    _ = gr.Markdown("""<h3 style="margin-top:0px;"><span style="color: #e8313e;">1、选择数据：</span></h3>""")
                with gr.Column(elem_id="add_border"):
                    with gr.Row():
                        img = gr.Image("static/1.jpg",
                                        sources=None,
                                        show_share_button=False,
                                        interactive=False)
                    with gr.Row():
                        gr.Examples(["static/2.jpg",
                                     "static/lpr.jpeg"], 
                                    img,label=None)
            with gr.Column():
                with gr.Row():
                    _ = gr.Markdown("""<h3 style="margin-top:0px;"><span style="color: #e8313e;">2、字段提取：</span></h3>""")
                    # modelSelected = gr.Dropdown(
                    #     label="选择模型", choices=MODELS, multiselect=False, value=MODELS[0], interactive=True,
                    #     show_label=False, container=False, elem_id="model-select-dropdown", filterable=False,scale=2
                    # )
                    
                with gr.Column(elem_id="add_border"):
                    with gr.Row():
                        extractFields = gr.Textbox(
                                placeholder="输入待提取字段名称，以“;”分隔。例如：姓名;月收入",
                                scale=11)
                    with gr.Row():
                        extractBtn = gr.Button("开始提取",scale=1)
                    with gr.Row():
                        textbox = gr.Markdown("{}".format("""提取结果：
                        ```json
                        {}
                        """))
                extractBtn.click(Executor.extract_fn_with_model,
                            inputs = [img,extractFields],
                            outputs = [textbox])
                
            with gr.Column():
                with gr.Row():
                    _ = gr.Markdown("""<h3 style="margin-top:0px;"><span style="color: #e8313e;">3、规则处理：</span></h3>""")
                    modelSelected = gr.Dropdown(
                        label="选择模型", choices=MODELS, multiselect=False, value=MODELS[0], interactive=True,
                        show_label=False, container=False, elem_id="model-select-dropdown", filterable=False,scale=2
                    )
                with gr.Column(elem_id="add_border"):
                    with gr.Row():
                        businessRule = gr.Textbox(
                                placeholder="请输入业务处理规则；",
                                scale=11)
                    with gr.Row():
                        processBtn = gr.Button("开始处理",scale=1)
                    with gr.Row():
                        rulebox = gr.Markdown("{}".format("""处理结果：
                        ```json
                        {}
                        """))
                processBtn.click(Executor.process_fn_with_ruler,
                                 inputs=[textbox,businessRule,modelSelected],
                                 outputs=[rulebox])


if __name__ == "__main__":
    if colab_env:
        demo.launch(share=True)
    else:
        demo.launch(server_name="0.0.0.0", server_port=8080)

