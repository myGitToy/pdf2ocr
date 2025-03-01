"""
你需要安装 Tesseract OCR 并确保它在你的系统 PATH 中。你可以使用以下命令在 Linux 上安装 Tesseract：
sudo apt-get update
sudo apt-get install tesseract-ocr

安装完成后，确保 tesseract 命令可以在终端中运行。你可以通过运行以下命令来验证：
tesseract --version

如果安装成功，你应该会看到 Tesseract 的版本信息。
接下来，确保 pytesseract 可以找到 Tesseract 的可执行文件。你可以在代码中显式指定 Tesseract 的路径：
import pytesseract
# 指定 Tesseract 的路径
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

#添加中文识别语言包
sudo apt-get install tesseract-ocr-chi-sim
"""
import pytesseract

# 指定 Tesseract 的路径
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
