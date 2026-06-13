"""Tests for the async task queue."""
import sys
sys.path.insert(0, "..")

import asyncio
from app.core.task_queue import TaskQueue, Task, TaskStatus


# ─── Mock Handlers ─────────────────────────────────────────────────

call_log = []

async def async_handler(x, y=0):
    call_log.append(("async", x, y))
    return x + y

def sync_handler(x, y=0):
    call_log.append(("sync", x, y))
    return x + y

async def failing_handler():
    call_log.append(("fail",))
    raise ValueError("test error")

async def slow_handler():
    await asyncio.sleep(0.1)
    call_log.append(("slow",))
    return "done"


# ─── Tests ─────────────────────────────────────────────────────────

def test_enqueue():
    """入队任务"""
    queue = TaskQueue()
    task = asyncio.run(queue.enqueue("test_func", args=(1, 2)))
    assert task.status == TaskStatus.pending
    assert task.func_name == "test_func"
    assert task.args == (1, 2)
    print("[PASS] Enqueue task")


def test_get_task():
    """获取任务"""
    queue = TaskQueue()
    task = asyncio.run(queue.enqueue("test_func"))
    retrieved = queue.get_task(task.id)
    assert retrieved is task
    assert queue.get_task("nonexistent") is None
    print("[PASS] Get task")


def test_get_all_tasks():
    """获取所有任务"""
    queue = TaskQueue()
    asyncio.run(queue.enqueue("a"))
    asyncio.run(queue.enqueue("b"))
    asyncio.run(queue.enqueue("c"))
    tasks = queue.get_all_tasks()
    assert len(tasks) == 3
    print("[PASS] Get all tasks")


async def test_worker_process():
    """Worker 处理任务"""
    queue = TaskQueue()
    queue.register_handler("add", async_handler)

    task = await queue.enqueue("add", args=(3, 4))
    await queue.start_worker()
    result = await queue.wait_for_task(task.id, timeout=5.0)
    await queue.stop_worker()

    assert result.status == TaskStatus.done
    assert result.result == 7
    print("[PASS] Worker process task")


async def test_sync_handler():
    """同步 handler"""
    queue = TaskQueue()
    queue.register_handler("sync_add", sync_handler)

    task = await queue.enqueue("sync_add", args=(10, 20))
    await queue.start_worker()
    result = await queue.wait_for_task(task.id, timeout=5.0)
    await queue.stop_worker()

    assert result.status == TaskStatus.done
    assert result.result == 30
    print("[PASS] Sync handler")


async def test_retry_on_failure():
    """失败重试"""
    call_log.clear()
    queue = TaskQueue()
    queue.register_handler("fail", failing_handler)

    task = await queue.enqueue("fail", max_retries=2)
    await queue.start_worker()

    # 等待足够时间让重试完成
    await asyncio.sleep(3)
    await queue.stop_worker()

    assert task.status == TaskStatus.failed
    assert task.retries == 2
    assert len(call_log) == 2  # 调用了 2 次
    print("[PASS] Retry on failure")


async def test_wait_timeout():
    """等待超时"""
    queue = TaskQueue()
    queue.register_handler("slow", slow_handler)

    task = await queue.enqueue("slow")
    await queue.start_worker()

    try:
        await queue.wait_for_task(task.id, timeout=0.01)
        print("[FAIL] Should have timed out")
    except TimeoutError:
        print("[PASS] Wait timeout")

    await queue.stop_worker()


async def test_multiple_tasks():
    """多个任务"""
    call_log.clear()
    queue = TaskQueue()
    queue.register_handler("add", async_handler)

    tasks = []
    for i in range(5):
        task = await queue.enqueue("add", args=(i, i))
        tasks.append(task)

    await queue.start_worker()

    results = []
    for task in tasks:
        result = await queue.wait_for_task(task.id, timeout=5.0)
        results.append(result)

    await queue.stop_worker()

    assert all(r.status == TaskStatus.done for r in results)
    assert len(call_log) == 5
    print("[PASS] Multiple tasks")


def run_async_tests():
    """Run all async tests."""
    asyncio.run(test_worker_process())
    asyncio.run(test_sync_handler())
    asyncio.run(test_retry_on_failure())
    asyncio.run(test_wait_timeout())
    asyncio.run(test_multiple_tasks())


if __name__ == "__main__":
    test_enqueue()
    test_get_task()
    test_get_all_tasks()
    run_async_tests()
    print("\nAll task queue tests passed!")
