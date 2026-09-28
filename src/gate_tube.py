"""
gate_tube.py

Комбинированный пайплайн:
  1. DynamicGate (створ) находит аномальные ячейки на каждом кадре.
  2. Для кадров с аномалиями строится габарит-труба (bend_tube.py).
  3. Аномалии, попавшие внутрь трубы, помечаются как препятствия.
  4. Fallback-труба (200 м прямая) НЕ используется для фильтрации —
     такие кадры помечаются как FALLBACK и точки "в трубе" не считаются.
  5. Если подряд идёт N кадров (по умолчанию 5), где створ дал аномалию
     И хотя бы часть аномальных точек попала в трубу (OK/SHORT) —
     печатается тревога "ЭТО ПРЕПЯТСТВИЕ".

Запуск:
    python3 gate_tube.py path/to/rosbag.db3 \
        --gate-rotate-deg 90 \
        --max-frames 50

Выход: только консольная статистика и тревоги.
"""

from __future__ import annotations

import argparse
import importlib.util
import sqlite3
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from dynamic_gate import DynamicGate


MODE_OK = 0   # труба построена по стене, длина ≥ SHORT_LEN
MODE_SHORT = 1   # труба построена, но короткая
MODE_FALLBACK = 2   # стены не найдены → прямая заглушка
MODE_NONE = 3   # труба вообще не построена (sections пусто)

MODE_NAMES = {
    MODE_OK:       "OK",
    MODE_SHORT:    "SHORT",
    MODE_FALLBACK: "FALLBACK",
    MODE_NONE:     "NONE",
}

# Пороги классификации длины
LEN_FALLBACK_MIN = 180.0   # > 180 м → считаем fallback-прямой
LEN_SHORT_MAX = 60.0    # < 60 м → short

# Подтверждение препятствия

OBSTACLE_CONFIRM_FRAMES = 5     # сколько кадров подряд = препятствие
OBSTACLE_MATCH_DIST_M = 8.0   # макс. смещение XY-центроида между кадрами


def classify_tube_mode(tube_len_m: float) -> int:
    if tube_len_m <= 0:
        return MODE_NONE
    if tube_len_m > LEN_FALLBACK_MIN:
        return MODE_FALLBACK
    if tube_len_m < LEN_SHORT_MAX:
        return MODE_SHORT
    return MODE_OK


#  Bag / PointCloud2

def find_cloud_topic(db_path: Path) -> str:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM topics "
                    "WHERE type='sensor_msgs/msg/PointCloud2'")
        rows = cur.fetchall()
        if not rows:
            raise RuntimeError(f"В {db_path} нет PointCloud2 топиков")
        if len(rows) > 1:
            print(f"  Внимание: несколько PointCloud2 топиков, беру первый: "
                  f"{rows[0][0]}")
        return rows[0][0]
    finally:
        con.close()


def make_reader(bag_path: Path, topic: str):
    storage = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    return reader


def cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    """PointCloud2 → numpy (N,3) float32. NaN/inf отфильтровываются."""
    xs = []
    ys = []
    zs = []
    for p in point_cloud2.read_points(
        msg, field_names=("x", "y", "z"), skip_nans=True
    ):
        if isinstance(p, np.void):
            x = float(p["x"])
            y = float(p["y"])
            z = float(p["z"])
        else:
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])
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
    print(f"  range: min={r.min():.3f} median={np.median(r):.3f} "
          f"p99={np.percentile(r, 99):.3f} max={r.max():.3f}")


# Загрузка модуля трубы
def load_tube_module(path: Path):
    spec = importlib.util.spec_from_file_location("bend_tube_mod", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не могу загрузить модуль трубы из {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bend_tube_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_tube_args(overrides: dict | None = None) -> Namespace:
    """Дефолты совпадают с argparse-дефолтами bend_tube.py."""
    d = dict(
        input=None, bag=None, topic="/lidar_points",
        frame_start=0, frame_end=None,
        voxel=0.15, voxel_keep_extremes=True, voxel_metric="x_abs",
        no_slice_grids=False, quiet=True,
        half_width=40.0, fwd_min=1.0, fwd_max=None,
        z_min=-1.0, z_max=2.0, tol_x=0.3,
        search_x_range="all",
        y_slice=0.5, y_overlap=0.5, y_step=0.3, y_slice_per_meter=0.03,
        min_points_in_slice=4, y_spread_limit=None,
        cell_per_meter=0.015, min_cell=0.05, max_cell=0.5,
        min_coverage=0.3, min_occupied=2, edge_frac=0.25,
        min_z_span=1.0, z_filter="span",
        min_slices=2, x_merge_tol=0.4, forward_sign=-1,
        fit_line=True, line_source="extreme_in_columns",
        chain_dx=0.8, chain_dy=3.0,
        min_line_length=2.0, min_line_points=3,
        merge_lines=True, merge_method="cluster",
        merge_cluster_eps=3.0, merge_side_check=0.5, merge_cos_min=0.4,
        curve_seg_len=8.0, merge_colinear=True,
        merge_colinear_dx=0.8, merge_colinear_angle_deg=6.0,
        bend_by_segments=True, bend_side="longest",
        no_data_angle=0.0, bend_min_seg_length=5.0,
        bend_smooth_window=3, bend_max_rate_deg_per_m=1.0,
        bend_max_extrapolation=30.0, bend_theta_poly_deg=2,
        bend_max_rms=0.15, bend_max_mean_x=2.0, bend_min_pts=15,
        bend_full_debug=False, bend_debug_file=None,
        bend_temporal_alpha=0.5, bend_temporal_off=True,
        train_half_width=1.05, train_fwd_min=None, train_fwd_max=None,
        safe_zone_thickness=0.30, safe_zone_min_length=10.0,
        safe_zone_max_rms=0.15,
        output_dir=None, no_save_empty=True,
        debug=False, explain_slices=False,
    )
    if overrides:
        d.update(overrides)
    return Namespace(**d)


#  Вспомогательное: поворот облака вокруг Z
def rotate_z(xyz: np.ndarray, deg: float) -> np.ndarray:
    if deg == 0.0 or xyz.size == 0:
        return xyz
    th = np.deg2rad(deg)
    c, s = float(np.cos(th)), float(np.sin(th))
    out = xyz.copy()
    x = xyz[:, 0].astype(np.float64)
    y = xyz[:, 1].astype(np.float64)
    out[:, 0] = (c * x - s * y).astype(xyz.dtype)
    out[:, 1] = (s * x + c * y).astype(xyz.dtype)
    return out


#  Основной цикл
def run(bag_path: Path,
        tube_mod,
        rotate_z_deg: float = 0.0,
        gate_rotate_deg: float = 0.0,
        tube_args_overrides: dict | None = None,
        gate_overrides: dict | None = None,
        max_frames: int | None = None,
        count_fallback: bool = False,
        obstacle_confirm_frames: int = OBSTACLE_CONFIRM_FRAMES,
        obstacle_match_dist_m: float = OBSTACLE_MATCH_DIST_M,
        debug: bool = True) -> None:

    print(f"\n=== {bag_path} ===")
    topic = find_cloud_topic(bag_path)
    print(f"  топик: {topic}")
    if gate_rotate_deg:
        print(f"  gate-rotate-z = {gate_rotate_deg:+.1f}° (перед створом)")
    if rotate_z_deg:
        print(f"  tube-rotate-z = {rotate_z_deg:+.1f}° (перед трубой)")
    print(f"  fallback-кадры: {'учитываются' if count_fallback else 'ИГНОРИРУЮТСЯ'}")
    print(f"  подтверждение препятствия: {obstacle_confirm_frames} кадров подряд, "
          f"match-dist = {obstacle_match_dist_m:.1f} м")

    tube_args = make_tube_args(tube_args_overrides)
    forward_sign = tube_args.forward_sign

    gate_kwargs = dict(
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
    if gate_overrides:
        gate_kwargs.update(gate_overrides)
    gate = DynamicGate(**gate_kwargs)

    reader = make_reader(bag_path, topic)

    # статистика по режимам
    mode_counter = {MODE_OK: 0, MODE_SHORT: 0,
                    MODE_FALLBACK: 0, MODE_NONE: 0}
    anom_in_tube_by_mode = {MODE_OK: 0, MODE_SHORT: 0,
                            MODE_FALLBACK: 0, MODE_NONE: 0}

    prev_bend_path = None

    obstacle_streak = 0        # сколько кадров подряд "в трубе"
    streak_start_frame = None     # кадр начала серии
    prev_obstacle_centroid = None     # (2,) XY центроид прошлого кадра
    active_event = None     # текущее подтверждённое событие
    obstacle_events = []       # список подтверждённых препятствий

    i = 0
    total_anom_frames = 0
    total_tube_ok = 0

    while reader.has_next():
        _topic, data, t_ns = reader.read_next()
        msg = deserialize_message(data, PointCloud2)
        xyz = cloud_to_xyz(msg)

        # флаги "на этом кадре есть препятствие в трубе"
        frame_has_obstacle = False
        frame_n_in = 0
        frame_obstacle_centroid = None

        if i == 0:
            dump_stats(xyz)

        # 1) СТВОР
        xyz_gate = rotate_z(xyz, gate_rotate_deg)
        mask = gate.process(xyz_gate)
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

        # 2) ТРУБА (только на аномальных кадрах)
        if n_anom:
            a_xyz_gate = gate.anomalies_to_xyz(mask)
            a_xyz = (rotate_z(a_xyz_gate, -gate_rotate_deg)
                     if a_xyz_gate.size else a_xyz_gate)

            xyz_t = rotate_z(xyz, rotate_z_deg)
            a_xyz_t = rotate_z(a_xyz, rotate_z_deg) if a_xyz.size else a_xyz

            sections = None
            try:
                tube_res = tube_mod.process_one_frame(
                    xyz_t, tube_args, forward_sign,
                    frame_label=f"[frame {i:06d}]",
                    prev_bend_path=prev_bend_path)
                sections = tube_res.get("bend_sections")
                prev_bend_path = tube_res.get("bend_path")
            except Exception as e:
                print(f"    [warn] труба упала на кадре {i}: {e}")

            # классификация режима
            tube_len_m = -1.0
            mode = MODE_NONE
            if sections and len(sections) >= 2:
                ys_t = np.array([s['center'][1] for s in sections],
                                dtype=np.float64)
                tube_len_m = float(np.abs(ys_t).max() - np.abs(ys_t).min())
                mode = classify_tube_mode(tube_len_m)
                if mode == MODE_OK:
                    total_tube_ok += 1

            mode_counter[mode] += 1

            # фильтр
            if (sections and a_xyz_t.size
                    and (mode in (MODE_OK, MODE_SHORT)
                         or count_fallback)):
                fwd_max_cur = (
                    tube_args.fwd_max
                    if tube_args.fwd_max is not None
                    else float(np.abs(xyz_t[:, 1]).max()))
                train_fwd_min = (
                    tube_args.train_fwd_min
                    if tube_args.train_fwd_min is not None
                    else tube_args.fwd_min)
                train_fwd_max = (
                    tube_args.train_fwd_max
                    if tube_args.train_fwd_max is not None
                    else fwd_max_cur)

                in_tube = tube_mod.points_in_tube(
                    a_xyz_t, sections,
                    train_half_width=tube_args.train_half_width,
                    train_fwd_min=train_fwd_min,
                    train_fwd_max=train_fwd_max,
                    z_min=tube_args.z_min, z_max=tube_args.z_max,
                    forward_sign=forward_sign)
                n_in = int(in_tube.sum())
                anom_in_tube_by_mode[mode] += n_in

                line += (f"  труба={tube_len_m:.0f}м"
                         f"[{MODE_NAMES[mode]}]"
                         f"  в трубе: {n_in}/{a_xyz_t.shape[0]}")
                if n_in:
                    a_in = a_xyz[in_tube]
                    # для подтверждения препятствия
                    frame_has_obstacle = True
                    frame_n_in = n_in
                    frame_obstacle_centroid = a_in[:, :2].mean(axis=0)
            else:
                if mode in (MODE_FALLBACK, MODE_NONE) and not count_fallback:
                    line += (f"  труба={tube_len_m:.0f}м"
                             f"[{MODE_NAMES[mode]}]"
                             f"  в трубе: SKIP")
                elif not sections:
                    line += "  труба не построена"
                else:
                    line += (f"  труба={tube_len_m:.0f}м"
                             f"[{MODE_NAMES[mode]}]  в трубе: —")

        print(line)

        # Подтверждение препятствия: N кадров подряд "в трубе"
        if frame_has_obstacle:
            if prev_obstacle_centroid is not None:
                d = float(np.linalg.norm(
                    frame_obstacle_centroid - prev_obstacle_centroid))
                if d <= obstacle_match_dist_m:
                    obstacle_streak += 1
                else:
                    obstacle_streak = 1
                    streak_start_frame = i
            else:
                obstacle_streak = 1
                streak_start_frame = i

            prev_obstacle_centroid = frame_obstacle_centroid

            # первый раз пересекли порог — создаём событие
            if obstacle_streak == obstacle_confirm_frames:
                active_event = {
                    'start_frame':  streak_start_frame,
                    'first_confirm': i,
                    'last_frame':    i,
                    'n_frames':      obstacle_streak,
                    'centroid':      frame_obstacle_centroid.copy(),
                    'n_points_last': frame_n_in,
                }
                print(f"    ⚠️  ЭТО ПРЕПЯТСТВИЕ!  "
                      f"(подтверждено {obstacle_streak} кадров подряд: "
                      f"{streak_start_frame}..{i}, "
                      f"XY=({frame_obstacle_centroid[0]:+.2f},"
                      f"{frame_obstacle_centroid[1]:+.2f}), "
                      f"точек в кадре={frame_n_in})")

            # препятствие уже подтверждено — продлеваем событие
            elif obstacle_streak > obstacle_confirm_frames:
                if active_event is not None:
                    active_event['last_frame'] = i
                    active_event['n_frames'] = obstacle_streak
                    active_event['n_points_last'] = frame_n_in
                if obstacle_streak % obstacle_confirm_frames == 0:
                    print(f"    ⚠️  ПРЕПЯТСТВИЕ ПРОДОЛЖАЕТСЯ "
                          f"({obstacle_streak} кадров подряд, кадр {i}, "
                          f"точек={frame_n_in})")
        else:
            # серия прервалась
            if (obstacle_streak >= obstacle_confirm_frames
                    and active_event is not None):
                print(f"    ✓ препятствие исчезло "
                      f"(было {obstacle_streak} кадров: "
                      f"{active_event['start_frame']}.."
                      f"{active_event['last_frame']}, "
                      f"центроид XY=({active_event['centroid'][0]:+.2f},"
                      f"{active_event['centroid'][1]:+.2f}))")
                obstacle_events.append(active_event)
                active_event = None
            obstacle_streak = 0
            prev_obstacle_centroid = None
            streak_start_frame = None

        i += 1
        if max_frames is not None and i >= max_frames:
            print(f"  (остановлено на {max_frames} кадрах)")
            break

    print(f"\n  Итого кадров: {i}, "
          f"кадров с аномалиями: {total_anom_frames}, "
          f"из них труба OK: {total_tube_ok}")

    print("  Режимы трубы на аномальных кадрах:")
    total_modes = sum(mode_counter.values())
    for m, name in MODE_NAMES.items():
        cnt = mode_counter[m]
        pct = 100.0 * cnt / max(total_modes, 1)
        print(f"    {name:<9} {cnt:>5} ({pct:5.1f}%)   "
              f"точек в трубе: {anom_in_tube_by_mode[m]}")

    # «дозакрываем» незавершённое событие, если запись оборвалась на нём
    if (obstacle_streak >= obstacle_confirm_frames
            and active_event is not None):
        active_event['last_frame'] = i - 1
        obstacle_events.append(active_event)

    print(f"\n  === ПОДТВЕРЖДЁННЫЕ ПРЕПЯТСТВИЯ "
          f"(порог: {obstacle_confirm_frames} кадров подряд) ===")
    if not obstacle_events:
        print("    нет")
    else:
        for ev in obstacle_events:
            print(f"    кадры {ev['start_frame']:>5}..{ev['last_frame']:<5} "
                  f"({ev['n_frames']} кадров)  "
                  f"XY=({ev['centroid'][0]:+.2f},{ev['centroid'][1]:+.2f})  "
                  f"точек в последнем кадре: {ev['n_points_last']}")


def main():
    ap = argparse.ArgumentParser(
        description="Створ + габарит-труба. Fallback-труба (200м прямая) "
                    "игнорируется при подсчёте 'в трубе'. "
                    "При N кадрах подряд 'створ + в трубе' печатает "
                    "'ЭТО ПРЕПЯТСТВИЕ'.")
    ap.add_argument("bag", type=Path, help="путь к .db3")
    ap.add_argument("--tube-script", type=Path,
                    default=Path(__file__).with_name("bend_tube.py"),
                    help="путь к модулю трубы")
    ap.add_argument("--rotate-z-deg", type=float, default=0.0,
                    help="поворот облака перед трубой (редко нужно)")
    ap.add_argument("--gate-rotate-deg", type=float, default=0.0,
                    help="поворот облака перед створом "
                         "(для forward=-Y: 90)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--count-fallback", action="store_true", default=False,
                    help="если задано — считать 'в трубе' и на FALLBACK-кадрах"
                         "(по умолчанию они игнорируются)")
    ap.add_argument("--obstacle-streak", type=int, default=5,
                    help="сколько кадров подряд 'створ + в трубе' "
                         "считать подтверждённым препятствием "
                         "(по умолчанию 5)")
    ap.add_argument("--obstacle-match-dist", type=float, default=8.0,
                    help="макс. смещение XY-центроида между кадрами (м), "
                         "при котором считаем, что это то же препятствие "
                         "(по умолчанию 8.0)")
    ap.add_argument("--no-gate-debug", dest="debug",
                    action="store_false", default=True)
    args = ap.parse_args()

    if not args.tube_script.exists():
        print(f"ОШИБКА: не найден модуль трубы {args.tube_script}")
        sys.exit(1)

    tube_mod = load_tube_module(args.tube_script)
    run(args.bag, tube_mod,
        rotate_z_deg=args.rotate_z_deg,
        gate_rotate_deg=args.gate_rotate_deg,
        max_frames=args.max_frames,
        count_fallback=args.count_fallback,
        obstacle_confirm_frames=args.obstacle_streak,
        obstacle_match_dist_m=args.obstacle_match_dist,
        debug=args.debug)


if __name__ == "__main__":
    main()
