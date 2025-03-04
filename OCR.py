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
    def __init__(self, psm=6, block_height=10000, max_retries=3, timeout=600, max_chunk_size=64000):
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
            max_chunk_size (int): API文本处理时的最大块大小，默认64000字符
        """
        self.pdf_path = None
        self.psm = psm
        self.block_height = block_height
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_chunk_size = max_chunk_size  # 添加最大块大小属性
        # 加载环境变量
        load_dotenv()
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

        # 明确设置 tesseract 命令路径
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        
        # 使用用户家目录创建 tessdata 目录
        home_dir = os.path.expanduser("~")
        self.tessdata_dir = os.path.join(home_dir, '.tessdata')
        os.makedirs(self.tessdata_dir, exist_ok=True)
        os.environ['TESSDATA_PREFIX'] = self.tessdata_dir
        
        self.check_and_install_tessdata()
        
    def check_and_install_tessdata(self):
        """
        检查并安装Tesseract中文训练数据到用户目录
        """
        print("检查Tesseract中文语言包...")
        
        # 使用用户家目录下的 tessdata
        chi_sim_path = os.path.join(self.tessdata_dir, 'chi_sim.traineddata')
        
        if not os.path.exists(chi_sim_path):
            print("中文训练数据文件不存在，尝试下载...")
            try:
                # 直接下载训练数据到用户目录
                import requests
                url = "https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata"
                try:
                    print(f"正在下载中文语言数据到 {chi_sim_path}...")
                    response = requests.get(url, stream=True, timeout=600)
                    response.raise_for_status()
                    with open(chi_sim_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"中文训练数据下载成功: {chi_sim_path}")
                except Exception as e:
                    print(f"下载失败: {e}，尝试使用系统默认中文数据...")
                    # 如果下载失败，尝试找到系统安装的中文数据并复制
                    system_chi_sim = "/usr/share/tesseract-ocr/4.00/tessdata/chi_sim.traineddata"
                    if os.path.exists(system_chi_sim):
                        print(f"找到系统中文数据，复制到用户目录...")
                        shutil.copy2(system_chi_sim, chi_sim_path)
                        print(f"复制成功: {chi_sim_path}")
                    else:
                        raise RuntimeError("无法找到或下载中文训练数据")
            except Exception as e:
                print(f"警告: {e}")
                print("继续尝试使用系统默认语言包...")
                
        # 检测语言包是否可用
        try:
            print("测试OCR引擎...")
            test_image = Image.new('RGB', (100, 30), color = (255, 255, 255))
            pytesseract.image_to_string(test_image, lang='chi_sim')
            print("OCR引擎测试成功")
        except Exception as e:
            print(f"OCR引擎测试失败: {e}")
            print("尝试不指定语言包...")
            try:
                pytesseract.image_to_string(test_image)
                print("默认OCR引擎可用")
            except:
                print("OCR引擎完全不可用，请检查Tesseract安装")

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

        # 检查文本长度
        text_length = len(text)
        print(f"原始文本长度: {text_length} 字符")

        # 使用实例的max_chunk_size而不是硬编码的30000
        if text_length > self.max_chunk_size:
            print(f"文本过长，采用分块处理策略 (最大块大小: {self.max_chunk_size})")
            chunks = []
            # 按段落分割文本
            paragraphs = text.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                # 如果添加当前段落会导致块超过限制，则先处理当前块
                if len(current_chunk) + len(para) + 2 > self.max_chunk_size and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                        
            # 添加最后一个块
            if current_chunk:
                chunks.append(current_chunk)
                
            print(f"将文本分为 {len(chunks)} 个块进行处理")
            
            # 处理每个块
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"处理第 {i+1}/{len(chunks)} 个块...")
                chunk_prompt = f"""你是一个OCR后处理专家，要求如下：
1. 修复OCR识别的文本中的错误，使其更加通顺、正确；
2. 如果文本有语法错误或者不通顺的情况，请尽可能在尊重原文的基础上进行修改；
3. 如果段落语义混乱无法总结和归纳，可以直接删除；
4. 最后不需要返回修改内容和原文的差异比较；
5. 请注意：这是一个分块处理的文档，当前是第 {i+1}/{len(chunks)} 块，请确保处理后的文本可以与其他块无缝连接；
6. 输出时直接给出修复后的文本即可，输出格式请使用markdown。

以下是OCR文本块：
{chunk}"""
                
                chunk_result = self._call_deepseek_api(api_url, headers, chunk_prompt)
                processed_chunks.append(chunk_result)
                
            # 合并处理后的块
            return "\n\n".join(processed_chunks)
        else:
            # 对于较短文本使用原有处理逻辑
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
9. 包含有：微信、朋友圈、用户留言、只看作者、正在加载、写留言、与作者互动等内容的请仔细甄别，如果与全文没有关联的话请删除；
10. 如果有可能，保留原本的段落结构和标题，如果有加粗字体，也一并显示；
11. 输出时直接给出修复后的文本即可，输出格式请使用markdown。

以下是OCR文本：
{text}"""
            return self._call_deepseek_api(api_url, headers, prompt)

    def _call_deepseek_api(self, api_url, headers, prompt):
        """
        调用DeepSeek API的辅助方法
        
        参数:
            api_url: API端点
            headers: 请求头
            prompt: 提示文本
            
        返回:
            API返回的处理后文本
        """
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 4000  # 明确指定最大输出token
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
                
                # 解析响应
                response_data = response.json()
                
                # 验证响应格式是否正确
                if 'choices' not in response_data or not response_data['choices']:
                    error_msg = "API返回格式不正确：找不到'choices'字段"
                    print(error_msg)
                    return prompt + f"\n[Error: {error_msg}]"
                    
                choice = response_data['choices'][0]
                if 'message' not in choice or 'content' not in choice.get('message', {}):
                    error_msg = "API返回格式不正确：找不到'message.content'字段"
                    print(error_msg)
                    return prompt + f"\n[Error: {error_msg}]"
                
                polished_text = choice['message']['content']
                
                # 验证返回文本是否有意义
                if not polished_text or len(polished_text) < 10:
                    error_msg = f"API返回文本过短或为空: '{polished_text}'"
                    print(error_msg)
                    return prompt + f"\n[Error: {error_msg}]"
                
                print("DeepSeek API调用成功")
                return polished_text
                
            except requests.exceptions.Timeout:
                print(f"DeepSeek API请求超时 (尝试 {attempt+1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    # 如果不是最后一次尝试，则等待后重试
                    time.sleep(2)  # 等待2秒后重试
                else:
                    return prompt + "\n[Error: DeepSeek API request timed out after multiple attempts]"
            except requests.exceptions.ConnectionError as e:
                print(f"DeepSeek API连接错误: {e} (尝试 {attempt+1}/{self.max_retries})")
                # 打印更多网络诊断信息
                print(f"错误详情: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    retry_wait = 5  # 等待时间延长到5秒
                    print(f"等待 {retry_wait} 秒后重试...")
                    time.sleep(retry_wait)
                else:
                    return prompt + f"\n[Error: Connection to DeepSeek API failed after {self.max_retries} attempts: {e}]"
            except requests.exceptions.RequestException as e:
                print(f"DeepSeek API请求错误: {e}")
                return prompt + f"\n[Error calling DeepSeek API: {e}]"
            except (KeyError, IndexError) as e:
                print(f"解析DeepSeek API响应错误: {e}")
                return prompt + f"\n[Error parsing DeepSeek API response: {e}]"
            except Exception as e:
                print(f"与DeepSeek API通信时发生未知错误: {e}")
                return prompt + f"\n[Unknown error with DeepSeek API: {e}]"

        return prompt + "\n[Error: API调用失败]"

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
8. 当检测到以下特征的文本时，请直接删除该部分内容：
    8.1 包含3个以上无关联汉字连续出现
    8.2 存在数字字母非常规混合（如"显划重点"）
    8.3 超过50%内容不符合中文语法结构
    8.4含有异常符号穿插（如< > [ ]）
9. 包含有：微信、朋友圈、用户留言、只看作者、正在加载、写留言、与作者互动等内容的请仔细甄别，如果与全文没有关联的话请删除；
10. 如果有可能，保留原本的段落结构和标题，如果有加粗字体，也一并显示；
11. 输出时直接给出修复后的文本即可，输出格式请使用markdown。

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
