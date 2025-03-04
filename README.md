# PDF2OCR 项目说明

## 简介
该项目用于对 PDF 文档进行 OCR 识别及 DeepSeek 文本优化处理，支持批量和单个文件的离线处理。用户需将 PDF 文件上传至项目根目录下的 `upload` 文件夹中，处理结果将分别保存到 `upload/ocr` 与 `upload/polish` 目录中。

## 安装

1. 安装 Python 依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 安装 Tesseract OCR：
   - Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim`
   - Windows: 下载 Tesseract 并配置路径（在 OCR.py 中指定路径）

   **故障排除**：
   - 如遇权限问题：
     ```bash
     # 解决权限错误："[Errno 13] Permission denied: '/usr/share/tesseract-ocr'"
     sudo chmod -R a+r /usr/share/tesseract-ocr
     ```
   - 确认 Tesseract 安装位置：
     ```bash
     which tesseract
     tesseract --version
     ```
   - 如果无法访问系统 tessdata 目录，程序会自动尝试：
     1. 在用户家目录下创建 `.tessdata` 文件夹
     2. 下载所需的中文语言数据
     3. 使用该目录进行OCR处理
     
   - 如果上述方法仍然不起作用，可以手动操作：
     ```bash
     # 创建用户级别的tessdata目录
     mkdir -p ~/.tessdata
     
     # 复制中文语言数据
     sudo cp /usr/share/tesseract-ocr/4.00/tessdata/chi_sim.traineddata ~/.tessdata/
     
     # 设置环境变量
     export TESSDATA_PREFIX=~/.tessdata
     ```

3. 配置环境变量：
   - 在项目根目录下创建 `.env` 文件，填写：
     ```
     DEEPSEEK_API_KEY=你的API密钥
     ```
   - 或 复制.env.sample并重命名   

## 使用方法

### 批量处理
将 PDF 文件上传至 `upload` 文件夹，然后运行：
```bash
python batch_process_multi_files.py
```
程序将按文件名顺序处理 PDF，遇到已处理文件将跳过以节省计算资源。

### 单个文件处理
执行以下命令处理单个 PDF 文件：
```bash
python batch_process_single_file.py path/to/your.pdf --output-dir 输出目录
```
可选参数：
- `--psm`：Tesseract 的页面分割模式（默认6）
- `--block-height`：图像分块高度（默认10000）
- `--no-preserve-layout`：不保留原始布局

### 自动定时处理

项目支持通过定时任务在指定时间段自动运行批处理：

1. 默认配置将在每天00:30-08:00之间运行批处理
2. 安装定时任务：
   ```bash
   # 查看当前cron配置
   crontab -l
   
   # 添加项目的cron配置
   crontab -l > current_crontab
   cat crontab.txt >> current_crontab
   crontab current_crontab
   rm current_crontab
   
   # 或直接编辑
   crontab -e
   # 然后添加以下行
   # 30 0 * * * cd /home/george/文档/repos/pdf2ocr && /usr/bin/python3 scheduled_batch_process.py >> /home/george/文档/repos/pdf2ocr/cron.log 2>&1
   ```

3. 定时任务说明：
   - 任务会在00:30自动启动
   - 任务会在08:00自动停止（即使未完成）
   - 日志保存在`scheduled_task.log`和`cron.log`中

## 项目结构
- `OCR.py`：OCR 主模块，包含 PDF 图像提取、OCR 识别和 DeepSeek API 调用。
- `batch_process_multi_files.py`：批量处理多个 PDF 文件。
- `batch_process_single_file.py`：针对单个 PDF 文件的处理脚本。
- `gradio_demo.py`：单个文件的处理流程演示

## 注意事项
- 确保 Tesseract 及其中文语言包正确安装。
- DeepSeek API 请求依赖网络环境，必要时可调整重试次数和超时时间。
- 请确保 `.env` 文件中已正确配置 DEEPSEEK_API_KEY。
- 网络超时默认为60秒，因此大部分情况下DeepSeek API无法及时返回数据从而会导致超时。解决方案采用异步操作，需要手动点击获取最新状态（因为gradio最新版5.20 删除了刷新功能）
- 对于超长文本处理，系统会自动分块处理（最大块大小默认为64K字符），确保完整文本都能得到处理。

## 高级配置
可以通过修改代码中的以下参数优化处理效果：
- OCR引擎参数:
  ```python
  ocr_engine = pdf2ocr(
      psm=6,                # 页面分割模式
      block_height=10000,   # 图像分块高度
      max_retries=3,        # API请求重试次数
      timeout=600,          # API请求超时时间(秒)
      max_chunk_size=64000  # 文本处理最大分块大小(字符)
  )
  ```

如有疑问，请参考代码内详细注释或联系项目维护者。
