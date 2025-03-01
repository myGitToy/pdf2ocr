"""
批处理PDF文件，避开Gradio的超时限制
单个pdf文档处理
"""
import os
import sys
import argparse
from OCR import pdf2ocr

def process_file(pdf_path, output_dir=None, psm=6, block_height=10000, preserve_layout=True):
    """批处理单个PDF文件"""
    if not output_dir:
        output_dir = os.path.dirname(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ocr_output_path = os.path.join(output_dir, f"{base_name}_ocr.txt")
    deepseek_output_path = os.path.join(output_dir, f"{base_name}_deepseek.txt")
    
    print(f"处理文件: {pdf_path}")
    print(f"OCR结果将保存到: {ocr_output_path}")
    print(f"DeepSeek优化结果将保存到: {deepseek_output_path}")
    
    ocr_engine = pdf2ocr(psm=psm, block_height=block_height, max_retries=5, timeout=900)
    images = pdf2ocr.pdf_to_images(pdf_path)
    
    ocr_text = ""
    for i, img in enumerate(images):
        print(f"处理第 {i+1}/{len(images)} 页...")
        if preserve_layout:
            page_text = ocr_engine.ocr_image_preserve_lines(img) + "\n\n"
        else:
            page_text = ocr_engine.ocr_image(img) + "\n"
        ocr_text += page_text
    
    # 保存OCR结果
    with open(ocr_output_path, 'w', encoding='utf-8') as f:
        f.write(ocr_text)
    print(f"OCR完成，结果已保存")
    
    # 使用DeepSeek优化
    print(f"开始DeepSeek优化，这可能需要几分钟...")
    deepseek_text = ocr_engine.deepseek_text_filter(ocr_text)
    
    # 保存DeepSeek优化结果
    with open(deepseek_output_path, 'w', encoding='utf-8') as f:
        f.write(deepseek_text)
    print(f"DeepSeek优化完成，结果已保存")
    
    return ocr_output_path, deepseek_output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批处理PDF文件进行OCR和DeepSeek优化")
    parser.add_argument("pdf_path", help="PDF文件路径")
    parser.add_argument("--output-dir", help="输出目录", default=None)
    parser.add_argument("--psm", help="Tesseract PSM模式", type=int, default=6)
    parser.add_argument("--block-height", help="图像分块高度", type=int, default=10000)
    parser.add_argument("--preserve-layout", help="保留原始布局", action="store_true", default=True)
    
    args = parser.parse_args()
    ocr_path, deepseek_path = process_file(
        args.pdf_path, 
        args.output_dir, 
        args.psm, 
        args.block_height, 
        args.preserve_layout
    )
    print(f"处理完成！\nOCR结果: {ocr_path}\nDeepSeek优化结果: {deepseek_path}")

"""
批处理PDF文件的脚本，用于离线处理大型文档
"""
import os
import sys
import time
import argparse
from OCR import pdf2ocr

def process_pdf_file(pdf_path, output_dir=None, psm=6, block_height=10000, preserve_layout=True):
    """处理单个PDF文件并保存结果"""
    # 设置输出目录
    if not output_dir:
        output_dir = os.path.dirname(pdf_path) or "."
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ocr_output = os.path.join(output_dir, f"{base_name}_ocr.txt")
    deepseek_output = os.path.join(output_dir, f"{base_name}_deepseek.txt")
    
    print(f"开始处理: {pdf_path}")
    print(f"OCR结果将保存到: {ocr_output}")
    print(f"DeepSeek处理结果将保存到: {deepseek_output}")
    
    # 初始化OCR引擎
    ocr_engine = pdf2ocr(psm=psm, block_height=block_height, max_retries=5, timeout=900)
    
    # 提取图片
    try:
        print("从PDF提取图像...")
        images = pdf2ocr.pdf_to_images(pdf_path)
        print(f"提取了 {len(images)} 张图像")
    except Exception as e:
        print(f"提取图像失败: {e}")
        return False
    
    # OCR处理
    try:
        print("开始OCR处理...")
        ocr_text = ""
        for i, img in enumerate(images):
            print(f"处理图像 {i+1}/{len(images)}...")
            if preserve_layout:
                page_text = ocr_engine.ocr_image_preserve_lines(img)
            else:
                page_text = ocr_engine.ocr_image(img)
            ocr_text += page_text + "\n\n"
            
        # 保存OCR结果
        with open(ocr_output, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        print(f"OCR处理完成，结果已保存")
    except Exception as e:
        print(f"OCR处理失败: {e}")
        return False
    
    # DeepSeek处理
    try:
        print("开始DeepSeek处理...")
        start_time = time.time()
        deepseek_text = ocr_engine.deepseek_text_filter(ocr_text)
        elapsed = time.time() - start_time
        
        # 保存DeepSeek结果
        with open(deepseek_output, "w", encoding="utf-8") as f:
            f.write(deepseek_text)
        print(f"DeepSeek处理完成，耗时 {elapsed:.1f} 秒，结果已保存")
    except Exception as e:
        print(f"DeepSeek处理失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理PDF文件并进行OCR和DeepSeek优化")
    parser.add_argument("pdf_path", help="PDF文件路径")
    parser.add_argument("--output-dir", help="输出目录", default=None)
    parser.add_argument("--psm", help="Tesseract PSM模式", type=int, default=6)
    parser.add_argument("--block-height", help="图像分块高度", type=int, default=10000)
    parser.add_argument("--no-preserve-layout", help="不保留原始布局", action="store_true")
    
    args = parser.parse_args()
    
    success = process_pdf_file(
        args.pdf_path,
        args.output_dir,
        args.psm,
        args.block_height,
        not args.no_preserve_layout  # 反转标志
    )
    
    sys.exit(0 if success else 1)
