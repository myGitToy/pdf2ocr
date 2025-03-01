"""
查看已处理的OCR和DeepSeek结果
"""
import gradio as gr
import os
from pathlib import Path
import glob

def list_processed_files(dir_path='.'):
    """列出已处理的文件"""
    ocr_files = glob.glob(os.path.join(dir_path, '*_ocr.txt'))
    result_files = []
    for ocr_file in ocr_files:
        base_name = ocr_file[:-8]  # 移除'_ocr.txt'
        deepseek_file = f"{base_name}_deepseek.txt"
        if os.path.exists(deepseek_file):
            result_files.append((os.path.basename(base_name), ocr_file, deepseek_file))
    return result_files

def load_file_content(file_path):
    """加载文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败: {str(e)}"

def update_display(file_info):
    """更新显示内容"""
    if not file_info:
        return "请选择一个处理结果", "请选择一个处理结果"
    
    name, ocr_file, deepseek_file = file_info
    ocr_text = load_file_content(ocr_file)
    deepseek_text = load_file_content(deepseek_file)
    
    return ocr_text, deepseek_text

with gr.Blocks(title="PDF OCR结果查看器") as demo:
    gr.Markdown("## PDF OCR + DeepSeek优化结果查看器")
    gr.Markdown("查看已处理的PDF文件的OCR和DeepSeek优化结果。")
    
    processed_files = list_processed_files()
    file_choices = [(f"{name}", (name, ocr, deepseek)) for name, ocr, deepseek in processed_files]
    
    with gr.Row():
        file_selector = gr.Dropdown(
            choices=file_choices,
            label="选择已处理的文件",
            value=file_choices[0][1] if file_choices else None
        )
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            ocr_text = gr.Textbox(label="OCR原始结果", lines=20)
        with gr.Column(scale=1):
            deepseek_text = gr.Textbox(label="DeepSeek优化结果", lines=20)
    
    file_selector.change(update_display, inputs=[file_selector], outputs=[ocr_text, deepseek_text])
    
    if file_choices:
        # 初始化显示第一个文件的内容
        _, ocr_file, deepseek_file = file_choices[0][1]
        ocr_text.value = load_file_content(ocr_file)
        deepseek_text.value = load_file_content(deepseek_file)

if __name__ == "__main__":
    demo.launch(debug=True)
