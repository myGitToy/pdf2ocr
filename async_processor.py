"""
异步处理模块，用于在后台处理长时间运行的任务
"""

import asyncio
import threading
import uuid
import time
import os
from queue import Queue
from threading import Thread
from typing import Dict, Any, Optional, Callable

class AsyncTaskManager:
    def __init__(self, results_dir: str = "./results"):
        """
        初始化异步任务管理器
        
        Args:
            results_dir: 存储结果的目录
        """
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # 启动一个工作线程来处理任务队列
        self.task_queue = Queue()
        self.worker_thread = Thread(target=self._process_tasks, daemon=True)
        self.worker_thread.start()
        
        # 添加任务完成回调映射
        self.completion_callbacks: Dict[str, Callable] = {}
    
    def submit_task(self, function: Callable, args: tuple = (), kwargs: dict = None, on_complete: Callable = None) -> str:
        """
        提交一个任务到队列中异步处理
        
        Args:
            function: 要执行的函数
            args: 函数参数
            kwargs: 函数关键字参数
            on_complete: 任务完成时的回调函数
            
        Returns:
            task_id: 任务ID，用于后续查询结果
        """
        if kwargs is None:
            kwargs = {}
            
        task_id = str(uuid.uuid4())
        task_info = {
            "id": task_id,
            "status": "pending",
            "function": function,
            "args": args,
            "kwargs": kwargs,
            "result": None,
            "error": None,
            "start_time": None,
            "end_time": None,
            "progress": 0,
        }
        
        self.tasks[task_id] = task_info
        self.task_queue.put(task_id)
        
        # 如果提供了回调函数，则保存它
        if on_complete is not None:
            self.completion_callbacks[task_id] = on_complete
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务信息字典
        """
        if task_id not in self.tasks:
            return {"status": "not_found"}
        
        task_info = self.tasks[task_id].copy()
        # 移除不需要返回的字段
        task_info.pop("function", None)
        task_info.pop("args", None)
        task_info.pop("kwargs", None)
        
        return task_info
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务结果或None（如果任务尚未完成）
        """
        if task_id not in self.tasks:
            return None
        
        task_info = self.tasks[task_id]
        if task_info["status"] == "completed":
            return task_info["result"]
            
        return None
    
    def _process_tasks(self):
        """
        任务处理循环，在后台线程中运行
        """
        while True:
            try:
                task_id = self.task_queue.get()
                if task_id is None:
                    break  # 结束信号
                
                task_info = self.tasks[task_id]
                function = task_info["function"]
                args = task_info["args"]
                kwargs = task_info["kwargs"]
                
                # 更新任务状态
                task_info["status"] = "running"
                task_info["start_time"] = time.time()
                
                try:
                    # 执行任务
                    result = function(*args, **kwargs)
                    
                    # 如果结果是字符串且过大，保存到文件
                    if isinstance(result, str) and len(result) > 10000:
                        result_file = os.path.join(self.results_dir, f"{task_id}.txt")
                        with open(result_file, "w", encoding="utf-8") as f:
                            f.write(result)
                        task_info["result"] = f"结果已保存到文件：{result_file}"
                        task_info["result_file"] = result_file
                    else:
                        task_info["result"] = result
                        
                    task_info["status"] = "completed"
                except Exception as e:
                    task_info["status"] = "failed"
                    task_info["error"] = str(e)
                
                task_info["end_time"] = time.time()
                task_info["progress"] = 100
                
                # 调用任务完成回调(如果有)
                if task_id in self.completion_callbacks:
                    try:
                        callback = self.completion_callbacks[task_id]
                        callback(task_id)
                        del self.completion_callbacks[task_id]  # 回调完成后删除
                    except Exception as callback_error:
                        print(f"回调执行错误: {callback_error}")
                
            except Exception as e:
                print(f"任务处理错误: {e}")
            
            finally:
                self.task_queue.task_done()

# 创建全局任务管理器实例
task_manager = AsyncTaskManager()

def deepseek_process(text: str, ocr_engine, max_retries=3, timeout=600) -> str:
    """
    使用DeepSeek处理文本的包装函数
    """
    try:
        print(f"开始DeepSeek处理，文本长度: {len(text)} 字符")
        result = ocr_engine.deepseek_text_filter(text)
        print(f"DeepSeek处理完成，结果长度: {len(result)} 字符")
        return result
    except Exception as e:
        print(f"DeepSeek处理异常: {e}")
        return f"处理失败: {str(e)}"
