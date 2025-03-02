"""
OCR主模块，用于识别PDF中的文本
本模块仅适用于得到APP长截屏pdf文件的OCR识别，不适用于其他场景
"""
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import requests
import json
import os
import shutil
import subprocess
import tempfile
from dotenv import load_dotenv
import time

# 指定 Tesseract 的路径
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
class pdf2ocr():
    def __init__(self, psm=6, block_height=10000, max_retries=3, timeout=600):
        """
        初始化 OCR 识别引擎
        
        参数:
            psm (int): Page Segmentation Mode (页面分割模式)
                3 - 自动页面分割，无OSD（假设文本是一列）
                6 - 假设是单个文本块（默认）
                7 - 假设是单行文本
                8 - 假设是单词
                13 - 将图像视为单个文本行
            block_height (int): 图像分块高度，用于处理大图像
                可选值: 1000, 2000, 5000, 10000(默认)
                较小的值可能更适合内存受限的环境，但会影响切分处的识别度
                较大的值会较好的进行OCR识别，但可能会导致内存不足错误
            max_retries (int): API请求最大重试次数
            timeout (int): API请求超时时间(秒) 这里不能很小，如果仅设置60，等不到API返回的结果
        """
        self.pdf_path = None
        self.psm = psm
        self.block_height = block_height
        self.max_retries = max_retries
        self.timeout = timeout
        # 加载环境变量
        load_dotenv()
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

        # 创建临时目录
        #self.temp_dir = tempfile.mkdtemp()
        #pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        #os.environ['TESSDATA_PREFIX'] = self.temp_dir

        # 不检查和安装中文训练数据
        self.check_and_install_tessdata()
        
    def check_and_install_tessdata(self):
        """
        检查并安装Tesseract中文训练数据
        """
        print("检查Tesseract中文语言包...")
        # 确定tessdata目录
        tessdata_dir = os.getenv('TESSDATA_PREFIX', '/usr/share/tesseract-ocr/4.00/tessdata')
        
        if not os.path.exists(tessdata_dir):
            os.makedirs(tessdata_dir, exist_ok=True)
            print(f"创建tessdata目录: {tessdata_dir}")
            
        # 检查中文训练数据是否存在
        chi_sim_path = os.path.join(tessdata_dir, 'chi_sim.traineddata')
        if not os.path.exists(chi_sim_path):
            print("中文训练数据文件不存在，尝试安装...")
            try:
                # 方法1: 使用apt安装
                print("尝试使用apt安装tesseract-ocr-chi-sim...")
                subprocess.run(['apt-get', 'update'], check=True)
                subprocess.run(['apt-get', 'install', '-y', 'tesseract-ocr-chi-sim'], check=True)
                print("tesseract-ocr-chi-sim安装成功")
            except subprocess.SubprocessError:
                # 方法2: 直接下载训练数据
                print("apt安装失败，尝试直接下载训练数据...")
                import requests
                url = "https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata"
                try:
                    response = requests.get(url, stream=True, timeout=600)
                    response.raise_for_status()
                    with open(chi_sim_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"中文训练数据下载成功: {chi_sim_path}")
                except Exception as e:
                    raise RuntimeError(f"无法下载中文训练数据: {e}")
                
        # 确保环境变量设置正确
        os.environ['TESSDATA_PREFIX'] = tessdata_dir
        print(f"已设置TESSDATA_PREFIX={tessdata_dir}")
        
        # 验证训练数据是否可访问
        if os.path.exists(chi_sim_path):
            print(f"中文训练数据文件存在: {chi_sim_path}")
        else:
            raise RuntimeError("无法找到中文训练数据文件，请手动安装tesseract-ocr-chi-sim")

    @staticmethod
    def pdf_to_images(pdf_path):
        # 打开 PDF 文件
        pdf_document = fitz.open(pdf_path)
        images = []

        # 遍历每一页
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        return images

    def preprocess_image(self, image):
        # 转换为灰度图像
        gray_image = image.convert('L')
        # 增强对比度
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(2)
        # 二值化
        binary_image = enhanced_image.point(lambda x: 0 if x < 180 else 255, '1')  # 调整阈值
        # 去噪
        denoised_image = binary_image.filter(ImageFilter.MedianFilter(size=3))  # 使用更强的去噪滤波器
        return denoised_image

    def ocr_image(self, image):
        width, height = image.size
        texts = []

        try:
            for i in range(0, height, self.block_height):
                box = (0, i, width, min(i + self.block_height, height))
                block = image.crop(box)
                preprocessed_block = self.preprocess_image(block)  # 使用self.preprocess_image
                
                # 修改Tesseract配置，移除不支持的参数
                # 移除 --preserve-interword-spaces 参数
                custom_config = r'--oem 3 --psm {} -l chi_sim'.format(self.psm)  # 使用 LSTM OCR 引擎和单行文本模式
                
                # 获取OCR文本
                text = pytesseract.image_to_string(
                    preprocessed_block, 
                    lang='chi_sim', 
                    config=custom_config
                )
                texts.append(text)

            # 改进后处理逻辑，更好地处理分段
            processed_text = ""
            for block_text in texts:
                # 分割成行
                lines = block_text.split('\n')
                
                # 处理每一行文本
                current_paragraph = []
                for line in lines:
                    # 如果是空行，则表示是段落分隔符
                    if not line.strip():
                        if current_paragraph:
                            # 将当前段落合并为一个段落，添加到结果中
                            processed_text += ' '.join(current_paragraph) + '\n\n'
                            current_paragraph = []
                        continue
                    
                    # 非空行，加入当前段落
                    current_paragraph.append(line.strip())
                
                # 处理最后一个段落
                if current_paragraph:
                    processed_text += ' '.join(current_paragraph) + '\n\n'
            
            # 删除多余的空行并返回结果
            final_text = '\n'.join([para for para in processed_text.split('\n') if para.strip()])
            return final_text
            
        except Exception as e:
            return f"OCR处理失败: {str(e)}\n请确保已安装tesseract-ocr和中文语言包"

    # 添加一个新方法，用于保留原始行结构的OCR识别
    def ocr_image_preserve_lines(self, image):
        """
        执行OCR识别并尽量保留原始文本的行结构
        
        参数:
            image: PIL图像对象
            
        返回:
            识别的文本，保留原始行结构
        """
        width, height = image.size
        texts = []

        try:
            for i in range(0, height, self.block_height):
                box = (0, i, width, min(i + self.block_height, height))
                block = image.crop(box)
                preprocessed_block = self.preprocess_image(block)
                
                # 使用适合行级识别的PSM模式
                # PSM 6: 假设一个统一的文本块
                # PSM 4: 假设可变大小的单列文本
                line_psm = 4 if self.psm != 4 else 6
                
                # 移除不支持的参数
                custom_config = r'--oem 3 --psm {} -l chi_sim'.format(line_psm)
                text = pytesseract.image_to_string(
                    preprocessed_block,
                    lang='chi_sim', 
                    config=custom_config
                )
                texts.append(text)

            # 简单的后处理：保留原始行结构，但去除连续空行
            result_lines = []
            for block_text in texts:
                lines = block_text.split('\n')
                for line in lines:
                    # 只添加非空行或者前一行非空时的空行（保留段落分隔）
                    if line.strip() or (result_lines and result_lines[-1].strip()):
                        result_lines.append(line)
            
            # 合并为最终文本
            final_text = '\n'.join(result_lines)
            return final_text
            
        except Exception as e:
            return f"OCR处理失败: {str(e)}\n请确保已安装tesseract-ocr和中文语言包"
    
    def deepseek_text_filter(self, text):
        """
        使用DeepSeek API进行文本过滤和重构
        
        参数:
            text (str): OCR识别出的原始文本
            
        返回:
            str: 经过DeepSeek API处理后的文本
        """
        if not self.api_key:
            return text + "\n[Error: DEEPSEEK_API_KEY not found in .env file]"

        # DeepSeek API 的 URL - 正确的端点
        api_url = "https://api.deepseek.com/v1/chat/completions"  # 修复URL
        
        # 记录调试信息
        print(f"使用API URL: {api_url}")
        print(f"API密钥前几位: {self.api_key[:5]}***")
        
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 限制提示文本长度，避免请求过大
        max_text_len = 50000   # 最大文本长度（字符数）
        if len(text) > max_text_len:
            print(f"文本过长 ({len(text)} 字符)，截断至 {max_text_len} 字符...")
            text = text[:max_text_len] + "\n[文本已截断]"

        # 构建提示信息，让DeepSeek重构OCR文本
        prompt = f"""你是一个OCR后处理专家，要求如下：
1. 修复OCR识别的文本中的错误，使其更加通顺、正确；
2. 如果文本有语法错误或者不通顺的情况，请尽可能在尊重原文的基础上进行修改；
3. 如果段落语义混乱无法总结和归纳，可以直接删除；
4. 最后不需要返回修改内容和原文的差异比较；
5. 这本的名字叫《熊逸讲透资治通鉴》；
6. 如果遇到包含不连贯字符、无意义数字符号混合的文本段落，请删除该部分；
7. 如果遇到无法理解的文本段落，请删除该部分；
8. 当检测到以下特征的文本时，请直接删除该部分内容：
    8.1 包含3个以上无关联汉字连续出现
    8.2 存在数字字母非常规混合（如"显划重点"）
    8.3 超过50%内容不符合中文语法结构
    8.4含有异常符号穿插（如< > [ ]）
9. 包含有：微信、朋友圈、用户留言、只看作者、正在加载、写留言、与作者互动等内容的请仔细甄别，如果与全文没有关联的话请删除

以下是OCR文本：
{text}"""

        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }
        
        print(f"请求数据大小: {len(json.dumps(data))} 字节")

        for attempt in range(self.max_retries):
            try:
                print(f"尝试连接DeepSeek API (尝试 {attempt+1}/{self.max_retries})...")
                
                # 增加详细日志
                print("发送请求...")
                start_time = time.time()
                
                # 使用更精细的超时控制
                response = requests.post(
                    api_url, 
                    headers=headers, 
                    data=json.dumps(data).encode('utf-8'),
                    timeout=(60, self.timeout),  # (连接超时, 读取超时)
                    stream=False  # 确保不使用流式处理
                )
                
                print(f"请求耗时: {time.time() - start_time:.2f} 秒")
                print(f"响应状态码: {response.status_code}")
                
                # 检查响应状态
                response.raise_for_status()
                print("收到成功响应，解析数据...")
                
                # 打印完整的响应内容以便调试
                response_data = response.json()
                print(f"API响应内容: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                
                # 验证响应格式是否正确
                if 'choices' not in response_data or not response_data['choices']:
                    error_msg = "API返回格式不正确：找不到'choices'字段"
                    print(error_msg)
                    return text + f"\n[Error: {error_msg}]"
                    
                choice = response_data['choices'][0]
                if 'message' not in choice or 'content' not in choice.get('message', {}):
                    error_msg = "API返回格式不正确：找不到'message.content'字段"
                    print(error_msg)
                    return text + f"\n[Error: {error_msg}]"
                
                polished_text = choice['message']['content']
                
                # 验证返回文本是否有意义
                if not polished_text or len(polished_text) < 10:
                    error_msg = f"API返回文本过短或为空: '{polished_text}'"
                    print(error_msg)
                    return text + f"\n[Error: {error_msg}]"
                
                # 检查是否包含常见错误信息关键词
                error_keywords = ["error", "exception", "invalid", "failed", "unauthorized", "权限不足", "错误", "失败"]
                if any(keyword in polished_text.lower() for keyword in error_keywords):
                    print(f"API可能返回了错误信息: {polished_text}")
                    # 但仍然返回结果，因为可能只是巧合包含了这些词
                
                print("DeepSeek API调用成功")
                return polished_text
                
            except requests.exceptions.Timeout:
                print(f"DeepSeek API请求超时 (尝试 {attempt+1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    # 如果不是最后一次尝试，则等待后重试
                    time.sleep(2)  # 等待2秒后重试
                else:
                    return text + "\n[Error: DeepSeek API request timed out after multiple attempts]"
            except requests.exceptions.ConnectionError as e:
                print(f"DeepSeek API连接错误: {e} (尝试 {attempt+1}/{self.max_retries})")
                # 打印更多网络诊断信息
                print(f"错误详情: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    retry_wait = 5  # 等待时间延长到5秒
                    print(f"等待 {retry_wait} 秒后重试...")
                    time.sleep(retry_wait)
                else:
                    return text + f"\n[Error: Connection to DeepSeek API failed after {self.max_retries} attempts: {e}]"
            except requests.exceptions.RequestException as e:
                print(f"DeepSeek API请求错误: {e}")
                return text + f"\n[Error calling DeepSeek API: {e}]"
            except (KeyError, IndexError) as e:
                print(f"解析DeepSeek API响应错误: {e}")
                return text + f"\n[Error parsing DeepSeek API response: {e}]"
            except Exception as e:
                print(f"与DeepSeek API通信时发生未知错误: {e}")
                return text + f"\n[Unknown error with DeepSeek API: {e}]"

    def deepseek_text_filter_stream(self, text):
        """
        使用DeepSeek API进行文本过滤和重构（流式传输版本）
        
        参数:
            text (str): OCR识别出的原始文本
            
        返回:
            生成器，逐步返回处理后的文本
        """
        if not self.api_key:
            yield "Error: DEEPSEEK_API_KEY not found in .env file"
            return

        # DeepSeek API 的 URL - 流式接口
        api_url = "https://api.deepseek.com/v1/chat/completions"
        #api_url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建提示信息，让DeepSeek重构OCR文本
        prompt = f"""你是一个OCR后处理专家，要求如下：
1. 修复OCR识别的文本中的错误，使其更加通顺、正确；
2. 如果文本有语法错误或者不通顺的情况，请尽可能在尊重原文的基础上进行修改；
3. 如果段落语义混乱无法总结和归纳，可以直接删除；
4. 最后不需要返回修改内容和原文的差异比较；
5. 这本的名字叫《熊逸讲透资治通鉴》；
6. 如果遇到包含不连贯字符、无意义数字符号混合的文本段落，请删除该部分；
7. 如果遇到无法理解的文本段落，请删除该部分；
8. 当检测到以下特征的文本时，请直接删除整段内容：
    8.1 包含3个以上无关联汉字连续出现
    8.2 存在数字字母非常规混合（如"显划重点"）
    8.3 超过50%内容不符合中文语法结构
    8.4含有异常符号穿插（如< > [ ]）


以下是OCR文本：


{text}"""

        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "stream": True  # 启用流式传输
        }

        for attempt in range(self.max_retries):
            try:
                print(f"尝试连接DeepSeek API流式接口 (尝试 {attempt+1}/{self.max_retries})...")
                
                # 使用更精细的超时控制
                session = requests.Session()
                session.mount('https://', requests.adapters.HTTPAdapter(
                    max_retries=3,
                    pool_connections=10,
                    pool_maxsize=10
                ))
                
                # 发送请求并启用流式响应
                with session.post(
                    api_url, 
                    headers=headers, 
                    data=json.dumps(data).encode('utf-8'),
                    timeout=(60, self.timeout),  # (连接超时, 读取超时)
                    stream=True  # 启用流式传输
                ) as response:
                    response.raise_for_status()
                    
                    # 初始化累积的文本
                    accumulated_text = ""
                    
                    # 处理流式响应
                    for line in response.iter_lines():
                        if line:
                            # 解析 SSE 格式的数据
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                json_str = line_text[6:]  # 移除 'data: ' 前缀
                                
                                # 跳过 '[DONE]' 消息
                                if json_str.strip() == '[DONE]':
                                    continue
                                    
                                try:
                                    # 解析 JSON 数据
                                    chunk = json.loads(json_str)
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta_content = chunk['choices'][0].get('delta', {}).get('content', '')
                                        if delta_content:
                                            accumulated_text += delta_content
                                            yield accumulated_text  # 逐步返回累积的文本
                                except json.JSONDecodeError:
                                    print(f"JSON解析错误: {json_str}")
                    
                print("DeepSeek API流式传输完成")
                return  # 生成器结束
                
            except requests.exceptions.Timeout:
                print(f"DeepSeek API请求超时 (尝试 {attempt+1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2)  # 等待2秒后重试
                else:
                    yield text + f"\n[Error: DeepSeek API request timed out after {self.max_retries} attempts]"
            except requests.exceptions.ConnectionError as e:
                print(f"DeepSeek API连接错误: {e} (尝试 {attempt+1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2)  # 等待2秒后重试
                else:
                    yield text + f"\n[Error: Connection to DeepSeek API failed: {e}]"
            except requests.exceptions.RequestException as e:
                print(f"DeepSeek API请求错误: {e}")
                yield text + f"\n[Error calling DeepSeek API: {e}]"
                return
            except Exception as e:
                print(f"与DeepSeek API通信时发生未知错误: {e}")
                yield text + f"\n[Unknown error with DeepSeek API: {e}]"
                return
