import pytesseract
from OCR import pdf2ocr

# 指定 Tesseract 的路径
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# 创建OCR引擎实例
ocr_engine = pdf2ocr(psm=6, block_height=10000)

# 打开 PDF 并提取图像
pdf_path = './000 每一代人都必须回到这里.pdf'
images = pdf2ocr.pdf_to_images(pdf_path)

all_text = ""

# 对每个图像进行 OCR 识别
for img in images:
    text = ocr_engine.ocr_image(img)
    all_text += text + "\n"
    print(text)

# 使用DeepSeek进行文本过滤和重构
print("\n==== 处理后文本 ====\n")
filtered_text = ocr_engine.deepseek_text_filter(all_text)
print(filtered_text)