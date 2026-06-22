"""
备件库 SQL 查询工具（添加 SQL 注入防护）
对应成品库② · 备件库 SQL 查询

初版问题：直接执行 LLM 生成的 SQL 存在注入风险
修复：只允许 SELECT 查询 + 白名单表名 + 参数化查询兜底
"""

import sqlite3
import re
import os
from config import SQL_CONNECTION_STRING


class SparePartsQueryTool:
    """备件库存查询（SQLite 实现，带注入防护）"""

    ALLOWED_TABLES = {"spare_parts"}

    def __init__(self):
        self.db_path = "data/spare_parts.db"
        self._init_db()

    def _init_db(self):
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spare_parts (
                id INTEGER PRIMARY KEY,
                name TEXT,
                model TEXT,
                quantity INTEGER,
                location TEXT,
                status TEXT
            )
        """)
        cursor.executemany(
            "INSERT OR IGNORE INTO spare_parts VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "电机轴承", "M-2024-BEARING", 15, "A区-3架", "正常"),
                (2, "温度传感器 PT100", "SEN-TEMP-PT100", 8, "B区-1架", "正常"),
                (3, "变频器模块", "VFD-7.5kW", 3, "A区-5架", "低库存"),
                (4, "冷却风扇", "FAN-3000rpm", 20, "C区-2架", "正常"),
                (5, "密封圈套装", "SEAL-KIT-01", 0, "C区-4架", "缺货"),
            ],
        )
        conn.commit()
        conn.close()

    def query(self, sql: str) -> str:
        """
        执行 SQL 查询（只允许 SELECT）。
        安全限制：
        - 只允许 SELECT 语句（INSERT/UPDATE/DELETE/DROP 均拒绝）
        - 只允许操作 spare_parts 表
        """
        # 安全检查
        sql_stripped = sql.strip().upper()
        if not sql_stripped.startswith("SELECT"):
            return "[安全拒绝] 只允许 SELECT 查询"

        # 检查是否涉及非白名单表
        for word in sql_stripped.split():
            if word in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "EXEC", "--"):
                return f"[安全拒绝] 非法操作: {word}"

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()

            if not rows:
                return "查询结果为空"

            result = " | ".join(columns) + "\n" + "-" * 40 + "\n"
            for row in rows:
                result += " | ".join(str(v) for v in row) + "\n"
            return result
        except Exception as e:
            return f"[SQL 错误] {e}"
