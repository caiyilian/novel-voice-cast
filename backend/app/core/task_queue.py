"""
异步任务队列 — 轻量级实现，无需 Redis。

功能：
1. 任务入队/出队
2. 后台 worker 消费任务
3. 回调更新状态
4. 错误重试机制（最多重试 3 次）

注意：这是开发环境的轻量实现。生产环境可替换为 ARQ (Redis)。
"""
import asyncio
import enum
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional


# ─── Task Status ───────────────────────────────────────────────────

class TaskStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    retrying = "retrying"


# ─── Task ──────────────────────────────────────────────────────────

@dataclass
class Task:
    """异步任务"""
    id: str
    func_name: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.pending
    result: Any = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


# ─── Task Queue ────────────────────────────────────────────────────

class TaskQueue:
    """轻量级异步任务队列（无需 Redis）"""

    def __init__(self):
        self._queue: Optional[asyncio.Queue] = None
        self._tasks: Dict[str, Task] = {}
        self._handlers: Dict[str, Callable] = {}
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    def _get_queue(self) -> asyncio.Queue:
        """懒加载队列"""
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    def register_handler(self, name: str, handler: Callable):
        """注册任务处理函数"""
        self._handlers[name] = handler

    async def enqueue(
        self,
        func_name: str,
        args: tuple = (),
        kwargs: dict = None,
        max_retries: int = 3,
    ) -> Task:
        """入队任务"""
        task_id = f"task_{int(time.time() * 1000)}_{len(self._tasks)}"
        task = Task(
            id=task_id,
            func_name=func_name,
            args=args,
            kwargs=kwargs or {},
            max_retries=max_retries,
        )
        self._tasks[task_id] = task
        await self._get_queue().put(task)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务状态"""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        return list(self._tasks.values())

    async def start_worker(self):
        """启动后台 worker"""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop_worker(self):
        """停止后台 worker"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def _worker_loop(self):
        """Worker 主循环"""
        queue = self._get_queue()
        while self._running:
            try:
                task = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            await self._process_task(task)

    async def _process_task(self, task: Task):
        """处理单个任务"""
        handler = self._handlers.get(task.func_name)
        if not handler:
            task.status = TaskStatus.failed
            task.error = f"No handler for {task.func_name}"
            task.completed_at = time.time()
            return

        task.status = TaskStatus.running
        task.started_at = time.time()

        try:
            if asyncio.iscoroutinefunction(handler):
                task.result = await handler(*task.args, **task.kwargs)
            else:
                task.result = handler(*task.args, **task.kwargs)

            task.status = TaskStatus.done
            task.completed_at = time.time()

        except Exception as e:
            task.error = str(e)
            task.retries += 1

            if task.retries < task.max_retries:
                task.status = TaskStatus.retrying
                await asyncio.sleep(1)  # 等待 1 秒后重试
                await self._get_queue().put(task)
            else:
                task.status = TaskStatus.failed
                task.completed_at = time.time()

    async def wait_for_task(self, task_id: str, timeout: float = None) -> Task:
        """等待任务完成"""
        start = time.time()
        while True:
            task = self._tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")

            if task.status in (TaskStatus.done, TaskStatus.failed):
                return task

            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Task {task_id} timed out")

            await asyncio.sleep(0.1)


# ─── Global Queue Instance ─────────────────────────────────────────

_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """获取全局任务队列实例"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue
