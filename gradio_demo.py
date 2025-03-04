"""
OCR识别DEMO，使用Gradio构建Web界面
输入单个得到APP长截屏PDF文件，进行OCR识别和DeepSeek文本优化
输出为text格式的文本，用来测试优化效果
"""

import gradio as gr
from OCR import pdf2ocr
import os
import time
from dotenv import load_dotenv
import traceback
from async_processor import task_manager, deepseek_process

# 仅执行 OCR 识别，不进行 DeepSeek 校对
def process_pdf(pdf_path, psm, block_height, preserve_layout):
    try:
        ocr_engine = pdf2ocr(psm, block_height)
        images = pdf2ocr.pdf_to_images(pdf_path)
        ocr_text = ""
        
        # 根据用户选择使用不同的OCR方法
        for img in images:
            if preserve_layout:
                # 使用保留行结构的方法
                ocr_text += ocr_engine.ocr_image_preserve_lines(img) + "\n\n"
            else:
                # 使用默认OCR方法
                ocr_text += ocr_engine.ocr_image(img) + "\n"
        
        print("OCR处理完成")
        return ocr_text, None  # 返回OCR结果，DeepSeek部分返回None
    except Exception as e:
        error_msg = f"处理过程中发生错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # 在控制台输出详细错误
        return f"OCR处理失败: {str(e)}", None

# 单独处理文本校对
def polish_text(ocr_text):
    try:
        if not ocr_text or ocr_text.strip() == "":
            return "请先进行OCR识别获取文本"
            
        print("开始调用DeepSeek API进行文本优化...")
        # 创建实例时设置max_retries, timeout和max_chunk_size
        ocr_engine = pdf2ocr(max_retries=3, timeout=600, max_chunk_size=64000)  
        polished_text = ocr_engine.deepseek_text_filter(ocr_text)
        return polished_text
        
    except Exception as e:
        error_msg = f"DeepSeek处理错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

# 单独处理文本校对 - 流式版本
def polish_text_stream(ocr_text):
    if not ocr_text or ocr_text.strip() == "":
        yield "请先进行OCR识别获取文本"
        return
        
    print("开始调用DeepSeek API进行文本优化（流式模式）...")
    # 创建实例时设置max_retries和timeout
    ocr_engine = pdf2ocr(max_retries=3, timeout=600)
    
    try:
        # 使用生成器进行流式传输，不再传递max_retries和timeout参数
        for result in ocr_engine.deepseek_text_filter_stream(ocr_text):
            if result.startswith("[Error:"):
                print("DeepSeek API调用出错")
            yield result
    except Exception as e:
        error_msg = f"DeepSeek处理错误: {str(e)}"
        print(error_msg)
        yield error_msg

# 进行异步处理，立即返回任务ID
def start_polish_text(ocr_text):
    if not ocr_text or ocr_text.strip() == "":
        return "请先进行OCR识别获取文本", None
        
    print("提交DeepSeek处理任务...")
    
    # 创建OCR引擎，指定更大的max_chunk_size
    ocr_engine = pdf2ocr(max_retries=3, timeout=600, max_chunk_size=64000)
    
    # 提交异步任务
    task_id = task_manager.submit_task(
        function=deepseek_process,
        args=(ocr_text, ocr_engine),
        kwargs={"max_retries": 5, "timeout": 600}
    )
    
    # 返回任务信息
    return f"DeepSeek处理任务已提交，任务ID: {task_id}\n\n请点击右侧「查询处理状态」按钮查看进度", task_id

# 查询任务状态 - 添加自动更新逻辑
def check_task_status(task_id):
    if not task_id:
        return "没有正在处理的任务"
        
    task_info = task_manager.get_task_status(task_id)
    
    if task_info["status"] == "not_found":
        return f"找不到任务: {task_id}"
    
    if task_info["status"] == "pending":
        return f"任务 {task_id} 正在排队等待处理..."
    
    elif task_info["status"] == "running":
        elapsed = time.time() - task_info["start_time"]
        return f"任务 {task_id} 正在处理中...\n已耗时: {elapsed:.1f} 秒"
    
    elif task_info["status"] == "completed":
        elapsed = task_info["end_time"] - task_info["start_time"]
        if "result_file" in task_info:
            result_file = task_info["result_file"]
            with open(result_file, "r", encoding="utf-8") as f:
                result = f.read()
            return result
        else:
            return task_info["result"]
    
    elif task_info["status"] == "failed":
        return f"任务处理失败: {task_info['error']}"
    
    return f"任务状态: {task_info['status']}"

# 查询任务状态并自动更新结果
def auto_check_status(task_id):
    """自动检查任务状态并返回状态和结果"""
    if not task_id:
        return "没有正在处理的任务", None
        
    task_info = task_manager.get_task_status(task_id)
    
    if task_info["status"] == "not_found":
        return f"找不到任务: {task_id}", None
    
    if task_info["status"] == "pending":
        return f"任务 {task_id} 正在排队等待处理...", None
    
    elif task_info["status"] == "running":
        elapsed = time.time() - task_info["start_time"]
        return f"任务 {task_id} 正在处理中...\n已耗时: {elapsed:.1f} 秒", None
    
    elif task_info["status"] == "completed":
        elapsed = task_info["end_time"] - task_info["start_time"]
        result = None
        
        if "result_file" in task_info:
            with open(task_info["result_file"], "r", encoding="utf-8") as f:
                result = f.read()
        else:
            result = task_info["result"]
            
        status_msg = f"处理完成，耗时: {elapsed:.1f} 秒"
        return status_msg, result
    
    elif task_info["status"] == "failed":
        return f"任务处理失败: {task_info['error']}", None
    
    return f"任务状态: {task_info['status']}", None

with gr.Blocks(title="PDF OCR + DeepSeek语言优化", theme=gr.themes.Default()) as demo:
    gr.Markdown("## PDF OCR + DeepSeek语言优化")
    gr.Markdown("上传PDF文件进行OCR识别，并使用DeepSeek API优化识别结果。仅用于处理得到APP长截屏PDF文档。")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 输入部分
            pdf_file = gr.File(label="上传PDF文件", file_types=[".pdf"])
            psm = gr.Dropdown(choices=[3, 6, 7, 8, 13], value=6, label="PSM模式")
            block_height = gr.Radio(
                choices=[1000, 2000, 5000, 10000], 
                value=10000, 
                label="图像分块高度", 
                info="较小值适合内存受限环境，较大值提高识别质量但可能内存不足"
            )
            # 添加布局选项
            preserve_layout = gr.Checkbox(
                label="保留原始段落结构", 
                value=True,
                info="选中时尽量保留原文档的段落和换行，否则优化文本流畅性"
            )
            ocr_button = gr.Button("开始OCR识别", variant="primary")
        
        with gr.Column(scale=2):
            ocr_text = gr.Textbox(label="OCR识别结果", lines=10)
            
            # 添加一个进度条来显示任务进度
            auto_polling = gr.Checkbox(value=True, label="自动更新状态（gradio5.20已移出此特性）", visible=True,
                                     info="开启后会自动每5秒查询一次处理状态")
            with gr.Row():
                polish_button = gr.Button("使用DeepSeek优化文本", variant="primary")
                check_status_button = gr.Button("查询处理状态", variant="secondary")
            
            task_id_text = gr.Textbox(label="任务ID", visible=False)
            status_text = gr.Textbox(label="处理状态", visible=True)
            polished_text = gr.Textbox(label="DeepSeek优化结果", lines=20)
    
    # 设置事件处理
    ocr_button.click(
        fn=process_pdf,
        inputs=[pdf_file, psm, block_height, preserve_layout],
        outputs=[ocr_text]
    )
    
    # 提交DeepSeek处理任务后自动开始定时检查
    polish_button.click(
        fn=start_polish_text,
        inputs=[ocr_text],
        outputs=[status_text, task_id_text]
    )
    
    # 手动查询处理状态
    check_status_button.click(
        fn=auto_check_status,  # 使用新函数
        inputs=[task_id_text],
        outputs=[status_text, polished_text]  # 更新状态和结果
    )
    


if __name__ == "__main__":
    # 使用最新版Gradio支持的参数
    demo.launch(
        debug=True,
        server_name="0.0.0.0",  # 允许外部访问
        show_api=False,  # 隐藏API文档
        share=False      # 不公开到Internet
    )