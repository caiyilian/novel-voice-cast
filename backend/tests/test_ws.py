"""Tests for WebSocket progress pushing."""
import sys, asyncio, json
sys.path.insert(0, "..")

from app.api.ws import ConnectionManager, get_manager


# ─── Tests ─────────────────────────────────────────────────────────

def test_manager_init():
    """管理器初始化"""
    manager = ConnectionManager()
    assert len(manager._connections) == 0
    print("[PASS] Manager init")


def test_get_manager():
    """获取全局管理器"""
    manager = get_manager()
    assert manager is not None
    print("[PASS] Get manager")


async def test_broadcast():
    """广播消息"""
    manager = ConnectionManager()
    # 模拟连接（使用 mock）
    class MockWebSocket:
        def __init__(self):
            self.messages = []
            self.accepted = False

        async def accept(self):
            self.accepted = True

        async def send_json(self, data):
            self.messages.append(data)

    ws = MockWebSocket()
    await manager.connect(ws, project_id=1)
    assert ws.accepted
    assert 1 in manager._connections

    await manager.broadcast(1, {"type": "progress", "current": 50, "total": 100})
    assert len(ws.messages) == 1
    assert ws.messages[0]["type"] == "progress"
    print("[PASS] Broadcast")


async def test_send_progress():
    """发送进度事件"""
    manager = ConnectionManager()

    class MockWebSocket:
        def __init__(self):
            self.messages = []
        async def accept(self):
            pass
        async def send_json(self, data):
            self.messages.append(data)

    ws = MockWebSocket()
    await manager.connect(ws, project_id=1)

    await manager.send_progress(1, 50, 100, "synthesizing")
    assert len(ws.messages) == 1
    msg = ws.messages[0]
    assert msg["type"] == "progress"
    assert msg["current"] == 50
    assert msg["total"] == 100
    assert msg["stage"] == "synthesizing"
    print("[PASS] Send progress")


async def test_send_done():
    """发送完成事件"""
    manager = ConnectionManager()

    class MockWebSocket:
        def __init__(self):
            self.messages = []
        async def accept(self):
            pass
        async def send_json(self, data):
            self.messages.append(data)

    ws = MockWebSocket()
    await manager.connect(ws, project_id=1)

    await manager.send_done(1, "/output/chapter1.wav")
    assert len(ws.messages) == 1
    msg = ws.messages[0]
    assert msg["type"] == "done"
    assert msg["output_path"] == "/output/chapter1.wav"
    print("[PASS] Send done")


async def test_send_error():
    """发送错误事件"""
    manager = ConnectionManager()

    class MockWebSocket:
        def __init__(self):
            self.messages = []
        async def accept(self):
            pass
        async def send_json(self, data):
            self.messages.append(data)

    ws = MockWebSocket()
    await manager.connect(ws, project_id=1)

    await manager.send_error(1, "Synthesis failed")
    assert len(ws.messages) == 1
    msg = ws.messages[0]
    assert msg["type"] == "error"
    assert msg["message"] == "Synthesis failed"
    print("[PASS] Send error")


async def test_disconnect():
    """断开连接"""
    manager = ConnectionManager()

    class MockWebSocket:
        def __init__(self):
            self.messages = []
        async def accept(self):
            pass
        async def send_json(self, data):
            self.messages.append(data)

    ws = MockWebSocket()
    await manager.connect(ws, project_id=1)
    assert 1 in manager._connections

    manager.disconnect(ws, project_id=1)
    assert 1 not in manager._connections
    print("[PASS] Disconnect")


async def test_multiple_connections():
    """多个连接"""
    manager = ConnectionManager()

    class MockWebSocket:
        def __init__(self):
            self.messages = []
        async def accept(self):
            pass
        async def send_json(self, data):
            self.messages.append(data)

    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    await manager.connect(ws1, project_id=1)
    await manager.connect(ws2, project_id=1)

    await manager.broadcast(1, {"type": "test"})
    assert len(ws1.messages) == 1
    assert len(ws2.messages) == 1
    print("[PASS] Multiple connections")


async def test_different_projects():
    """不同项目的连接隔离"""
    manager = ConnectionManager()

    class MockWebSocket:
        def __init__(self):
            self.messages = []
        async def accept(self):
            pass
        async def send_json(self, data):
            self.messages.append(data)

    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    await manager.connect(ws1, project_id=1)
    await manager.connect(ws2, project_id=2)

    await manager.broadcast(1, {"type": "test"})
    assert len(ws1.messages) == 1
    assert len(ws2.messages) == 0  # 不同项目，不应收到消息
    print("[PASS] Different projects isolated")


def run_async_tests():
    """Run all async tests."""
    asyncio.run(test_broadcast())
    asyncio.run(test_send_progress())
    asyncio.run(test_send_done())
    asyncio.run(test_send_error())
    asyncio.run(test_disconnect())
    asyncio.run(test_multiple_connections())
    asyncio.run(test_different_projects())


if __name__ == "__main__":
    test_manager_init()
    test_get_manager()
    run_async_tests()
    print("\nAll WebSocket tests passed!")
