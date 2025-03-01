import gradio as gr
from OCR import pdf_to_images, ocr_image
import requests
import json
from dotenv import load_dotenv
import os
# 加载 .env 文件
load_dotenv()

def llm_polish(text):
    # 从环境变量中读取 DeepSeek API 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "Error: DEEPSEEK_API_KEY not found in .env file"

    # DeepSeek API 的 URL
    api_url = "https://api.deepseek.com"

    headers = {
        "Content-Type": "application/json; charset=utf-8",  # 添加 charset=utf-8
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "deepseek-v3",  # 替换为 DeepSeek 提供的模型名称
        "prompt": text,
        "max_tokens": 100,  # 根据需要调整
        "temperature": 0.7  # 根据需要调整
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data).encode('utf-8'))  # 使用 utf-8 编码
        response.raise_for_status()  # 检查是否有 HTTP 错误
        result = response.json()
        polished_text = result['choices'][0]['text']  # 根据 DeepSeek API 的响应格式调整
        return polished_text
    except requests.exceptions.RequestException as e:
        return f"Error calling DeepSeek API: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing DeepSeek API response: {e}"

def process_pdf(pdf_path):
    images = pdf_to_images(pdf_path)
    ocr_text = ""
    for img in images:
        ocr_text += ocr_image(img) + "\n"
    #polished_text = llm_polish(ocr_text.encode('utf-8'))  # 确保OCR使用 UTF-8
    polished_text = ocr_text
    return ocr_text, polished_text

iface = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(file_types=['.pdf']),
    outputs=[
        gr.Textbox(label="OCR Text"),
        gr.Textbox(label="LLM Polished Text")
    ],
    title="PDF OCR with LLM Polish",
    description="Upload a PDF file to extract text and polish it with LLM."
)

iface.launch()