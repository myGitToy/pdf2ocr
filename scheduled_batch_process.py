#!/usr/bin/env python3
"""
PDF批处理定时任务脚本
每天00:30-08:00时间段内运行批处理程序
"""

import os
import sys
import time
import datetime
import subprocess
import signal
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scheduled_task.log')
)

logger = logging.getLogger('pdf2ocr_scheduler')

def is_within_time_window():
    """检查当前时间是否在00:30-08:00之间"""
    now = datetime.datetime.now().time()
    start_time = datetime.time(0, 30)  # 00:30
    end_time = datetime.time(8, 0)    # 08:00
    
    return start_time <= now <= end_time

def run_batch_process():
    """运行批处理脚本"""
    logger.info("启动PDF批量处理...")
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    batch_script = os.path.join(script_dir, 'batch_process_multi_files.py')
    
    # 计算截止时间戳 (08:00)
    now = datetime.datetime.now()
    end_time = datetime.datetime(
        now.year, now.month, now.day, 8, 0
    )
    if now > end_time:
        # 如果当前时间已超过今天的8:00，则使用明天的8:00
        end_time += datetime.timedelta(days=1)
    
    end_timestamp = end_time.timestamp()
    
    try:
        # 启动批处理进程
        process = subprocess.Popen(
            [sys.executable, batch_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # 循环检查是否超过结束时间
        while process.poll() is None:
            if time.time() >= end_timestamp:
                logger.warning("已到达结束时间(08:00)，终止批处理进程")
                process.send_signal(signal.SIGTERM)
                time.sleep(5)  # 给进程一些时间来优雅地结束
                if process.poll() is None:  # 如果进程仍然在运行
                    process.kill()  # 强制终止
                break
            time.sleep(10)  # 每10秒检查一次
        
        # 获取进程输出
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info("批处理成功完成")
        else:
            logger.error(f"批处理失败，返回码：{process.returncode}")
            if stderr:
                logger.error(f"错误输出：{stderr.decode('utf-8', errors='replace')}")
    
    except Exception as e:
        logger.error(f"执行批处理时发生错误：{str(e)}")
        return False
    
    return True

def main():
    """主函数"""
    logger.info("定时任务启动，检查时间窗口...")
    
    if not is_within_time_window():
        logger.info("当前时间不在处理窗口(00:30-08:00)内，不执行批处理")
        return
    
    # 在时间窗口内，执行批处理
    success = run_batch_process()
    
    logger.info(f"定时任务结束，结果：{'成功' if success else '失败'}")

if __name__ == "__main__":
    main()
