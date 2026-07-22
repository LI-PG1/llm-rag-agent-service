"""
报修工单系统 API 工具
对应成品库② · 报修工单系统 Rest API
"""

import json
import time
import os
from typing import Optional

class RepairOrderTool:
    """报修工单系统（模拟实现）"""

    def __init__(self):
        self.orders_file = "data/repair_orders.json"
        self._init_storage()

    def _init_storage(self):
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.orders_file):
            with open(self.orders_file, "w") as f:
                json.dump([], f)

    def _load_orders(self) -> list:
        with open(self.orders_file, "r") as f:
            return json.load(f)

    def _save_orders(self, orders: list):
        with open(self.orders_file, "w") as f:
            json.dump(orders, f, ensure_ascii=False, indent=2)

    def submit(self, equipment_id: str, fault_desc: str, priority: str = "normal") -> str:
        """提交报修工单"""
        order_id = f"RO-{int(time.time())}-{equipment_id[:4]}"

        order = {
            "order_id": order_id,
            "equipment_id": equipment_id,
            "fault_desc": fault_desc,
            "priority": priority,
            "status": "已提交",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        orders = self._load_orders()
        orders.append(order)
        self._save_orders(orders)

        return f"工单已提交: {order_id}（优先级: {priority}）"

    def check_status(self, order_id: str) -> str:
        """查询工单状态"""
        orders = self._load_orders()
        for order in orders:
            if order["order_id"] == order_id:
                return (
                    f"工单: {order['order_id']}\n"
                    f"设备: {order['equipment_id']}\n"
                    f"故障: {order['fault_desc']}\n"
                    f"状态: {order['status']}\n"
                    f"优先级: {order['priority']}"
                )
        return f"未找到工单: {order_id}"

    def update_status(self, order_id: str, status: str) -> str:
        """更新工单状态（用于模拟流程推进）"""
        orders = self._load_orders()
        for order in orders:
            if order["order_id"] == order_id:
                order["status"] = status
                self._save_orders(orders)
                return f"工单 {order_id} 状态更新为: {status}"
        return f"未找到工单: {order_id}"
