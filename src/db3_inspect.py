"""db3_inspect.py — быстрый осмотр rosbag2 .db3"""
from pathlib import Path
import sqlite3


def inspect(db_path: Path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    print(f"\n=== {db_path.name} ===")
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        print("Таблицы:", tables)

        if "topics" in tables:
            cur.execute("SELECT id, name, type, serialization_format FROM topics")
            for row in cur.fetchall():
                print(f"  topic id={row[0]} name={row[1]} type={row[2]} "
                      f"format={row[3]}")

        if "messages" in tables:
            cur.execute("SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id")
            for row in cur.fetchall():
                print(f"  topic_id={row[0]}: {row[1]} сообщений")
    finally:
        con.close()


if __name__ == "__main__":
    for p in Path(".").rglob("*.db3"):
        inspect(p)