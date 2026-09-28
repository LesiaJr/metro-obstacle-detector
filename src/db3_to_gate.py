"""
db3_to_gate.py

Читает PointCloud2 из rosbag2 .db3, конвертирует в (N,3) numpy,
прогоняет через DynamicGate, печатает аномалии.

Топик определяется автоматически из таблицы topics.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from dynamic_gate import DynamicGate


def find_cloud_topic(db_path: Path) -> str:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name, type FROM topics "
            "WHERE type='sensor_msgs/msg/PointCloud2'"
        )
        rows = cur.fetchall()
        if not rows:
            raise RuntimeError(f"В {db_path} нет PointCloud2 топиков")
        if len(rows) > 1:
            print(f"  Внимание: несколько PointCloud2 топиков: {rows}. "
                  f"Беру первый.")
        return rows[0][0]
    finally:
        con.close()


def make_reader(bag_path: Path, topic: str) -> rosbag2_py.SequentialReader:
    storage = rosbag2_py.StorageOptions(uri=str(bag_path),
                                        storage_id="sqlite3")
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    return reader


def cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    xs, ys, zs = [], [], []
    for p in point_cloud2.read_points(
        msg, field_names=("x", "y", "z"), skip_nans=True
    ):
        if isinstance(p, np.void):
            x, y, z = float(p["x"]), float(p["y"]), float(p["z"])
        else:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            xs.append(x)
            ys.append(y)
            zs.append(z)
    if not xs:
        return np.zeros((0, 3), dtype=np.float32)
    return np.column_stack([xs, ys, zs]).astype(np.float32)


def dump_stats(xyz: np.ndarray) -> None:
    if xyz.size == 0:
        print("  (облако пустое)")
        return
    r = np.linalg.norm(xyz, axis=1)
    print(f"  xyz: shape={xyz.shape} "
          f"x[{xyz[:,0].min():+.2f},{xyz[:,0].max():+.2f}] "
          f"y[{xyz[:,1].min():+.2f},{xyz[:,1].max():+.2f}] "
          f"z[{xyz[:,2].min():+.2f},{xyz[:,2].max():+.2f}]")
    print(f"  range: min={r.min():.3f} "
          f"median={np.median(r):.3f} "
          f"p99={np.percentile(r, 99):.3f} "
          f"max={r.max():.3f}")


def run_bag(bag_path: Path, debug: bool = True,
            max_frames: int | None = None) -> None:
    print(f"\n=== {bag_path} ===")
    topic = find_cloud_topic(bag_path)
    print(f"  топик: {topic}")

    reader = make_reader(bag_path, topic)
    gate = DynamicGate(
        azimuth_bins=7200,
        elevation_bins=128,
        forward_half_angle_deg=45.0,
        min_range=3.0,
        max_range=200.0,
        strong_threshold=2.5,
        weak_threshold_min=1.5,
        n_confirm=4,
        m_confirm=6,
        neighbor_radius=2,
        neighbor_min_count=3,
        motion_default=2.5,
        motion_estimate=True,
        motion_far_min_range=20.0,
        motion_min_samples=200,
        debug=debug,
        debug_every=5,
    )

    i = 0
    total_anom_frames = 0
    while reader.has_next():
        _topic, data, t_ns = reader.read_next()
        msg = deserialize_message(data, PointCloud2)
        xyz = cloud_to_xyz(msg)

        if i == 0:
            dump_stats(xyz)

        mask = gate.process(xyz)
        n_anom = int(mask.sum()) if mask.ndim == 2 else 0
        marker = "!!!" if n_anom else "   "
        if n_anom:
            total_anom_frames += 1

        line = (f"{marker} [{i:04d}] t={t_ns/1e9:9.3f} "
                f"pts={len(xyz):7d} аномалий={n_anom}")
        if n_anom:
            info = gate.anomaly_info(mask)
            if info:
                rs = [r for _, _, r in info]
                azs = [az for _, az, _ in info]
                els = [el for el, _, _ in info]
                line += (f"  range={min(rs):.1f}..{max(rs):.1f}м "
                         f"az={min(azs):+.1f}..{max(azs):+.1f}° "
                         f"el={min(els):+.1f}..{max(els):+.1f}°")
        print(line)

        i += 1
        if max_frames is not None and i >= max_frames:
            print(f"  (остановлено на {max_frames} кадрах)")
            break

    print(f"  Итого кадров: {i}, кадров с аномалиями: {total_anom_frames}")


if __name__ == "__main__":
    db = Path("doubleT_platform/doubleT_platform_0.db3")
    run_bag(db, debug=True, max_frames=50)
