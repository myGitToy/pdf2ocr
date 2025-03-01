"""
批处理PDF文件，避开Gradio的超时限制
需求：
    我有100个pdf文档需要做ocr和deepseek文字优化处理，根据orc.py的代码逻辑，我需要确定以下几个问题：

    使用何种处理手段，gradio的界面形式还是使用批处理形式。pdf文档第一次需要上传，非本地文件
    处理过程中容易出错从而中断，如果进行再次处理但可以利用之前已处理好的文件，以解决时间和计算资源
    上传的文件目前设定为当前目录下的upload文件夹，里面再新建两个文件夹，分别叫ocr和polish    

实施方案：
    1. 选择批处理方式，与用户手动上传pdf文件到当前目录下的upload文件夹中的ocr和polish子目录配合使用。
    2. 对每个pdf文件，先调用 OCR 模块将pdf转换为图片并进行ocr识别，合并得到ocr文本，再调用 deepseek_text_filter 方法进行文字优化处理。
    3. 分别将OCR文本和DeepSeek返回的文本写入 upload/ocr 和 upload/polish 文件夹下（文件名称与原pdf对应）。
    4. 如果发现处理后的文件已存在，则跳过该文件，避免重复计算和中断后重新计算。
"""
import os
from OCR import pdf2ocr
import traceback

# 配置目录
BASE_DIR = os.path.join(os.getcwd(), "upload")
OCR_DIR = os.path.join(BASE_DIR, "ocr")
POLISH_DIR = os.path.join(BASE_DIR, "polish")

# 确保目标目录存在
for d in [OCR_DIR, POLISH_DIR]:
    os.makedirs(d, exist_ok=True)

def process_pdf(file_path):
    try:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        ocr_output_path = os.path.join(OCR_DIR, f"{filename}.txt")
        polish_output_path = os.path.join(POLISH_DIR, f"{filename}.txt")

        # 如果两个输出文件都存在则跳过处理
        if os.path.exists(ocr_output_path) and os.path.exists(polish_output_path):
            print(f"{filename} 已处理，跳过")
            return

        print(f"处理文件: {file_path}")
        # 实例化 OCR 类
        ocr_processor = pdf2ocr()

        # 从PDF中获取图像列表
        images = ocr_processor.pdf_to_images(file_path)

        # 对每张图片逐段OCR识别，合并文本
        ocr_text = ""
        for img in images:
            ocr_text += ocr_processor.ocr_image(img) + "\n"

        # 保存OCR结果
        with open(ocr_output_path, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        print(f"OCR结果已保存至: {ocr_output_path}")

        # 调用DeepSeek文本优化接口
        polished_text = ocr_processor.deepseek_text_filter(ocr_text)

        # 保存DeepSeek处理后的文本
        with open(polish_output_path, "w", encoding="utf-8") as f:
            f.write(polished_text)
        print(f"文字优化结果已保存至: {polish_output_path}")
        
    except Exception as e:
        print(f"处理 {file_path} 发生错误: {e}")
        traceback.print_exc()

def main():
    # 遍历上传目录中所有PDF文件（非本地文件需先上传到upload文件夹），按文件名排序
    for file in sorted(os.listdir(BASE_DIR)):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(BASE_DIR, file)
            process_pdf(file_path)

if __name__ == "__main__":
    main()
