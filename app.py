import numpy as np
import gradio as gr

from executor import Executor

Executor.init()
demo = gr.Blocks(css = "./static/custom.css")

with demo:
    gr.Markdown("""
    ### <h2><span style="color: #e8313e;">欢迎体验GIES智能信息提取系统</span></h2>\n
    """)
    with gr.Row():
        with gr.Column():
            img = gr.Image("static/10.jpg",
                            sources=None,
                            show_share_button=False,
                            interactive=False)
            with gr.Row():
                gr.Examples(["static/1.jpg", 
                             "static/2.jpg", 
                             "static/3.jpg",
                             "static/4.jpg", 
                             "static/5.jpg", 
                             "static/6.jpg",
                             "static/7.jpg", 
                             "static/8.jpg", 
                             "static/9.jpg"], 
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


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)

