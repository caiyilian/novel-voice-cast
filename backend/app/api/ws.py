"""
WebSocket 进度推送 — 实时推送合成进度到前端。

事件类型：
- progress: 进度更新 { type: "progress", current: 50, total: 200, stage: "synthesizing" }
- done: 合成完成 { type: "done", output_path: "..." }
- error: 合成失败 { type: "error", message: "..." }
"""
import asyncio
import json
from typing import Dict, List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])


# ─── Connection Manager ────────────────────────────────────────────

class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self._connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, project_id: int):
        """接受 WebSocket 连接"""
        await websocket.accept()
        if project_id not in self._connections:
            self._connections[project_id] = set()
        self._connections[project_id].add(websocket)

    def disconnect(self, websocket: WebSocket, project_id: int):
        """断开 WebSocket 连接"""
        if project_id in self._connections:
            self._connections[project_id].discard(websocket)
            if not self._connections[project_id]:
                del self._connections[project_id]

    async def broadcast(self, project_id: int, message: dict):
        """向项目的所有连接广播消息"""
        if project_id not in self._connections:
            return

        dead_connections = set()
        for connection in self._connections[project_id]:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)

        # 清理断开的连接
        for conn in dead_connections:
            self._connections[project_id].discard(conn)

    async def send_progress(self, project_id: int, current: int, total: int, stage: str = "synthesizing"):
        """发送进度事件"""
        await self.broadcast(project_id, {
            "type": "progress",
            "current": current,
            "total": total,
            "stage": stage,
        })

    async def send_done(self, project_id: int, output_path: str = ""):
        """发送完成事件"""
        await self.broadcast(project_id, {
            "type": "done",
            "output_path": output_path,
        })

    async def send_error(self, project_id: int, message: str):
        """发送错误事件"""
        await self.broadcast(project_id, {
            "type": "error",
            "message": message,
        })


# ─── Global Manager ────────────────────────────────────────────────

_manager = ConnectionManager()


def get_manager() -> ConnectionManager:
    """获取全局连接管理器"""
    return _manager


# ─── WebSocket Endpoint ────────────────────────────────────────────

@router.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: int):
    """WebSocket 端点，用于接收实时进度推送"""
    await _manager.connect(websocket, project_id)
    try:
        while True:
            # 保持连接，等待消息
            data = await websocket.receive_text()
            # 可以处理客户端发送的消息（如取消请求）
            try:
                message = json.loads(data)
                if message.get("type") == "cancel":
                    await _manager.send_progress(project_id, 0, 0, "cancelled")
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        _manager.disconnect(websocket, project_id)
