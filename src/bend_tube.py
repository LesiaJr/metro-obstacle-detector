"""
Габарит-труба + safe-zone + вывод точек препятствий.

Этапы:
  1. Поиск стен (find_columns_in_slice / detect_wall_candidates)
  2. Линии стен (fit_line_pca / build_wall_lines)
  3. Слияние линий (colinear merge + clustering)
  4. Bend path (θ(Y) по стенам)
  5. Extrude в трубу (build_bend_tube / points_in_tube)
  6. Safe-zone вдоль надёжных стен
  7. Obstacle points = tube & ~wall & ~safe

Вход:
  * ROS2 bag с PointCloud2    --bag PATH --topic TOPIC

Выход:
  * По кадру: obstacles -> <output-dir>/frame_XXXXXX.npy
  * В stdout: сводка (frame, pts, tube, wall, safe, obstacles, range)

Дефолты соответствуют команде, которой этот скрипт обычно гоняется.
Только логика и вывод — без визуализации.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    from rosbag2_py import (SequentialReader, StorageOptions,
                            ConverterOptions)
    import rclpy.serialization
    from rosidl_runtime_py.utilities import get_message
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


#  helpers

_DEBUG_FH = None


def log(*a, **kw):
    print(*a, **kw, flush=True)


def dlog(*a, **kw):
    target = _DEBUG_FH if _DEBUG_FH is not None else sys.stdout
    print(*a, **kw, file=target, flush=True)


def adaptive_cell_size(y_center, cell_per_meter, min_cell, max_cell):
    return float(np.clip(cell_per_meter * max(abs(y_center), 1.0),
                         min_cell, max_cell))


def voxel_downsample_np(points, voxel_size):
    if voxel_size <= 0 or len(points) == 0:
        return points
    xyz = points[:, :3]
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0,
                                   return_inverse=True,
                                   return_counts=True)
    summed = np.zeros((len(counts), points.shape[1]), dtype=np.float64)
    np.add.at(summed, inverse, points.astype(np.float64))
    out = (summed / counts[:, None]).astype(points.dtype)
    return out


def voxel_downsample_extremes(points, voxel_size, metric='x_abs'):
    if voxel_size <= 0 or len(points) == 0:
        return points
    xyz = points[:, :3]
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    kx, ky, kz = keys[:, 0], keys[:, 1], keys[:, 2]
    x = points[:, 0]
    y = points[:, 1]
    ax = np.abs(x)
    ay = np.abs(y)
    if metric == 'x_abs':
        m = -ax
    elif metric == 'y_abs':
        m = -ay
    elif metric == 'xy_radial':
        m = -np.sqrt(x * x + y * y)
    elif metric == 'xy_max':
        m = -np.maximum(ax, ay)
    elif metric == 'x_max':
        m = x
    elif metric == 'x_min':
        m = -x
    else:
        m = -ax
    order = np.lexsort((m, kz, ky, kx))
    skx, sky, skz = kx[order], ky[order], kz[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (skx[1:] != skx[:-1]) | \
                (sky[1:] != sky[:-1]) | \
                (skz[1:] != skz[:-1])
    idx = order[first]
    return points[idx]


#  ROS2 bag reader

def pointcloud2_to_numpy(msg):
    field_names = [f.name for f in msg.fields]
    has_intensity = 'intensity' in field_names
    dtype_list = []
    for f in msg.fields:
        np_type = {
            1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
            5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
        }[f.datatype]
        dtype_list.append((f.name, np_type))
    buf = np.frombuffer(msg.data, dtype=np.dtype(dtype_list))
    if buf.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x = buf['x'].astype(np.float32)
    y = buf['y'].astype(np.float32)
    z = buf['z'].astype(np.float32)
    i = (buf['intensity'].astype(np.float32) if has_intensity
         else np.zeros_like(x))
    out = np.stack([x, y, z, i], axis=1)
    mask = np.isfinite(out).all(axis=1)
    return out[mask]


class BagReader:
    def __init__(self, bag_path, topic, quiet=False):
        if not HAS_ROS2:
            raise RuntimeError("ROS2-библиотеки не установлены.")
        self.bag_path = bag_path
        self.topic = topic
        self.quiet = quiet
        t0 = time.perf_counter()
        if not quiet:
            log(f"[bag] opening {bag_path} topic={topic}")
        storage = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter = ConverterOptions(input_serialization_format='cdr',
                                     output_serialization_format='cdr')
        self.reader = SequentialReader()
        self.reader.open(storage, converter)
        type_map = {t.name: t.type
                    for t in self.reader.get_all_topics_and_types()}
        if topic not in type_map:
            raise ValueError(f"Топик {topic} не найден. "
                             f"Доступные: {list(type_map.keys())}")
        self.msg_type = get_message(type_map[topic])
        self._count = None
        if not quiet:
            log(f"[bag] opened in {time.perf_counter()-t0:.2f}s")

    def count(self):
        if self._count is not None:
            return self._count
        if not self.quiet:
            log("[count] start")
        t0 = time.perf_counter()
        r = SequentialReader()
        r.open(StorageOptions(uri=self.bag_path, storage_id='sqlite3'),
               ConverterOptions(input_serialization_format='cdr',
                                output_serialization_format='cdr'))
        n = 0
        total = 0
        while r.has_next():
            topic, _, _ = r.read_next()
            total += 1
            if topic == self.topic:
                n += 1
        self._count = n
        if not self.quiet:
            log(f"[count] done: {n} кадров за {time.perf_counter()-t0:.2f}s")
        return n

    def iter_frames(self, i_start=0, i_end=None):
        if i_end is None:
            i_end = self.count()
        if not self.quiet:
            log(f"[iter] skip до {i_start}, стоп на {i_end}")
        count = 0
        while self.reader.has_next():
            topic, data, _ = self.reader.read_next()
            if topic != self.topic:
                continue
            if count < i_start:
                count += 1
                continue
            if count >= i_end:
                break
            msg = rclpy.serialization.deserialize_message(data, self.msg_type)
            yield count, pointcloud2_to_numpy(msg)
            count += 1

    def close(self):
        pass


#  ЭТАП 1: поиск колонн (кандидатов в стены)
def find_columns_in_slice(pts_slice,
                          side,
                          half_width,
                          tol_x,
                          z_min,
                          z_max,
                          cell_size,
                          min_coverage,
                          min_occupied_bins,
                          edge_frac,
                          search_all_x=True,
                          min_z_span=1.0,
                          z_filter='span'):
    if len(pts_slice) == 0:
        return None
    x = pts_slice[:, 0]
    z = pts_slice[:, 2]
    if search_all_x:
        if side == 'left':
            x_lo, x_hi = -half_width, 0.0
        else:
            x_lo, x_hi = 0.0, half_width
    else:
        if side == 'left':
            x_lo, x_hi = -half_width, -half_width + tol_x
        else:
            x_lo, x_hi = half_width - tol_x, half_width
    x_edges = np.arange(x_lo, x_hi + cell_size, cell_size)
    z_edges = np.arange(z_min, z_max + cell_size, cell_size)
    if len(x_edges) < 2 or len(z_edges) < 2:
        return None
    nx = len(x_edges) - 1
    nz = len(z_edges) - 1
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    zi = np.clip(np.digitize(z, z_edges) - 1, 0, nz - 1)
    occ = np.zeros((nx, nz), dtype=bool)
    occ[xi, zi] = True
    edge_bins = max(1, int(round(edge_frac * nz)))
    columns = []
    for k in range(nx):
        col = occ[k]
        n_occ = int(col.sum())
        if n_occ == 0:
            continue
        coverage = n_occ / nz
        z_idx = np.where(col)[0]
        z_span_bins = int(z_idx.max() - z_idx.min()) + 1
        z_span_m = float(z_span_bins * cell_size)
        has_low = bool(col[:edge_bins].any())
        has_high = bool(col[-edge_bins:].any())
        if z_filter == 'edges':
            passes_z = has_low and has_high
        elif z_filter == 'both':
            passes_z = (z_span_m >= min_z_span) and has_low and has_high
        else:
            passes_z = z_span_m >= min_z_span
        passes = (n_occ >= min_occupied_bins and
                  coverage >= min_coverage and passes_z)
        if not passes:
            continue
        x_center = 0.5 * (x_edges[k] + x_edges[k + 1])
        local_idx = np.where(xi == k)[0]
        columns.append({'x_bin': k, 'x_center': x_center,
                        'coverage': coverage, 'local_idx': local_idx,
                        'n_occ': n_occ,
                        'z_span_m': z_span_m})
    return {'columns': columns}


def detect_wall_candidates(pts,
                           half_width,
                           fwd_min,
                           fwd_max,
                           z_min,
                           z_max,
                           tol_x=0.3,
                           search_all_x=True,
                           y_slice=1.0,
                           y_overlap=0.5,
                           y_step=None,
                           y_slice_per_meter=0.0,
                           min_points_in_slice=4,
                           cell_per_meter=0.015,
                           min_cell=0.05,
                           max_cell=0.3,
                           min_coverage=0.5,
                           min_occupied_bins=3,
                           edge_frac=0.25,
                           min_z_span=1.0,
                           z_filter='span',
                           min_slices_for_wall=2,
                           x_merge_tol=0.4,
                           max_y_spread_in_column=None,
                           quiet=False):
    n = len(pts)
    wall_mask = np.zeros(n, dtype=bool)
    accepted_columns = []
    if y_step is None:
        y_step = max(y_slice - y_overlap, 1e-3)
    y_step = max(y_step, 1e-3)
    y_abs = np.abs(pts[:, 1])
    x = pts[:, 0]
    z = pts[:, 2]
    y_centers = np.arange(fwd_min + 0.5 * y_slice,
                          fwd_max - 0.5 * y_slice + 1e-6, y_step)

    def slice_thickness(yc):
        if y_slice_per_meter <= 0.0:
            return y_slice
        return max(y_slice, y_slice_per_meter * abs(yc))

    for side in ('left', 'right'):
        side_mask = (x < 0) if side == 'left' else (x > 0)
        absx = np.abs(x)
        if search_all_x:
            edge_mask = (absx <= half_width)
        else:
            edge_mask = (absx >= half_width - tol_x) & (absx <= half_width)
        base_mask = (side_mask & edge_mask &
                     (y_abs >= fwd_min) & (y_abs <= fwd_max) &
                     (z >= z_min) & (z <= z_max))
        base_idx = np.where(base_mask)[0]
        if len(base_idx) == 0:
            continue
        slice_columns = []
        for k, yc in enumerate(y_centers):
            ys = slice_thickness(yc)
            y0 = yc - 0.5 * ys
            y1 = yc + 0.5 * ys
            m = (y_abs[base_idx] >= y0) & (y_abs[base_idx] < y1)
            if not m.any():
                slice_columns.append([]); continue
            sub_idx = base_idx[m]
            if len(sub_idx) < min_points_in_slice:
                slice_columns.append([]); continue
            sub_pts = pts[sub_idx]
            cs = adaptive_cell_size(yc, cell_per_meter, min_cell, max_cell)
            result = find_columns_in_slice(
                sub_pts, side, half_width, tol_x, z_min, z_max, cs,
                min_coverage=min_coverage,
                min_occupied_bins=min_occupied_bins,
                edge_frac=edge_frac, search_all_x=search_all_x,
                min_z_span=min_z_span, z_filter=z_filter)
            if result is None:
                slice_columns.append([]); continue
            y_spread_limit = max_y_spread_in_column or ys
            cols = result['columns']
            for c in cols:
                g_idx = sub_idx[c['local_idx']]
                if g_idx.size > 1:
                    yv = np.abs(pts[g_idx, 1])
                    if (yv.max() - yv.min()) > y_spread_limit:
                        continue
                c['global_idx'] = g_idx
                c['y_center'] = yc
                c['slice_k'] = k
                c['side'] = side
                c['cell_size'] = cs
            slice_columns.append([c for c in cols if 'global_idx' in c])
        flat = [c for cols in slice_columns for c in cols]
        if not flat:
            continue
        xcs = np.array([c['x_center'] for c in flat])
        order = np.argsort(xcs)
        flat = [flat[i] for i in order]
        groups = []
        cur = [flat[0]]
        for c in flat[1:]:
            if abs(c['x_center'] - cur[-1]['x_center']) <= x_merge_tol:
                cur.append(c)
            else:
                groups.append(cur); cur = [c]
        groups.append(cur)
        for g in groups:
            slices = {c['slice_k'] for c in g}
            if len(slices) < min_slices_for_wall:
                continue
            for c in g:
                wall_mask[c['global_idx']] = True
                accepted_columns.append(c)
    return {'wall_mask': wall_mask, 'columns': accepted_columns,
            'y_step': y_step}


#  ЭТАП 2: линии стен

def fit_line_pca(pts_xy):
    if len(pts_xy) < 2:
        return None
    pts_xy = np.asarray(pts_xy, dtype=np.float64)
    c = pts_xy.mean(axis=0)
    centered = pts_xy - c
    cov = (centered.T @ centered) / len(centered)
    vals, vecs = np.linalg.eigh(cov)
    direction = vecs[:, -1]
    proj = centered @ direction
    perp = centered - np.outer(proj, direction)
    rms = float(np.sqrt((perp ** 2).sum(axis=1).mean()))
    t_min, t_max = float(proj.min()), float(proj.max())
    p_a = c + t_min * direction
    p_b = c + t_max * direction
    if abs(p_a[1]) <= abs(p_b[1]):
        p_start, p_end = p_a, p_b
    else:
        p_start, p_end = p_b, p_a
        direction = -direction
    angle = float(np.degrees(np.arctan2(direction[0], abs(direction[1]))))
    return {'point': c, 'direction': direction,
            'p_start': p_start, 'p_end': p_end,
            'length': float(np.linalg.norm(p_end - p_start)),
            'rms': rms, 'n_points': int(len(pts_xy)),
            'angle_deg': angle}


def fit_polyline_pca(pts_xy, seg_len=8.0, min_pts=2):
    pts = np.asarray(pts_xy, dtype=np.float64)
    if len(pts) < min_pts:
        return None
    order = np.argsort(pts[:, 1])
    pts = pts[order]
    y_min = float(pts[:, 1].min())
    y_max = float(pts[:, 1].max())
    dy = y_max - y_min
    if dy < seg_len * 1.2:
        line = fit_line_pca(pts)
        if line is None:
            return None
        return [{
            'p_start': np.asarray(line['p_start']),
            'p_end':   np.asarray(line['p_end']),
            'angle_deg': line['angle_deg'],
            'rms': line['rms'], 'n_points': line['n_points'],
            'length': line['length'],
        }]
    n_seg = max(1, int(np.ceil(dy / seg_len)))
    edges = np.linspace(y_min, y_max, n_seg + 1)
    segments = []
    for k in range(n_seg):
        ya, yb = edges[k], edges[k + 1]
        ya_ov = max(y_min, ya - 0.5)
        yb_ov = min(y_max, yb + 0.5)
        m = (pts[:, 1] >= ya_ov) & (pts[:, 1] <= yb_ov)
        sub = pts[m]
        if len(sub) < min_pts:
            continue
        line = fit_line_pca(sub)
        if line is None:
            continue
        segments.append({
            'p_start': np.asarray(line['p_start']),
            'p_end':   np.asarray(line['p_end']),
            'angle_deg': line['angle_deg'],
            'rms': line['rms'], 'n_points': line['n_points'],
            'length': line['length'],
        })
    if not segments:
        return None
    segments.sort(key=lambda s: min(s['p_start'][1], s['p_end'][1]))
    for s in segments:
        if s['p_start'][1] > s['p_end'][1]:
            s['p_start'], s['p_end'] = s['p_end'], s['p_start']
    return segments


def cluster_columns_spatial(columns, chain_dx=0.6, chain_dy=3.0):
    n = len(columns)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    xs = np.array([c['x_center'] for c in columns])
    ys = np.array([c['y_center'] for c in columns])
    for i in range(n):
        for j in range(i + 1, n):
            if abs(ys[i] - ys[j]) > chain_dy:
                continue
            if abs(xs[i] - xs[j]) > chain_dx:
                continue
            union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(columns[i])
    return list(groups.values())


def _select_points_for_cluster(cluster, pts, source):
    centers_xy = np.array([[c['x_center'], c['y_center']] for c in cluster],
                          dtype=np.float64)
    if source == 'centers':
        return centers_xy
    if source == 'points':
        all_idx = np.unique(np.concatenate(
            [c['global_idx'] for c in cluster]))
        if all_idx.size < 2:
            return None
        p = pts[all_idx]
        return np.stack([p[:, 0], np.abs(p[:, 1])], axis=1)
    if source == 'extreme_in_columns':
        by_slice = {}
        for c in cluster:
            by_slice.setdefault(c['slice_k'], []).append(c)
        sel = []
        for k, group in by_slice.items():
            all_idx = np.unique(np.concatenate(
                [c['global_idx'] for c in group]))
            if all_idx.size == 0:
                continue
            p = pts[all_idx]
            j = int(np.argmax(np.abs(p[:, 0])))
            sel.append([p[j, 0], np.abs(p[j, 1])])
        if len(sel) < 2:
            return None
        return np.asarray(sel, dtype=np.float64)
    if source == 'extreme_perp':
        prelim = fit_line_pca(centers_xy)
        if prelim is None:
            return None
        P = prelim['point']; D = prelim['direction']
        perp = np.array([-D[1], D[0]])
        by_slice = {}
        for c in cluster:
            by_slice.setdefault(c['slice_k'], []).append(c)
        sel = []
        for k, group in by_slice.items():
            all_idx = np.unique(np.concatenate(
                [c['global_idx'] for c in group]))
            if all_idx.size == 0:
                continue
            p = pts[all_idx]
            xy = np.stack([p[:, 0], np.abs(p[:, 1])], axis=1)
            rel = xy - P
            s = rel @ perp
            j = int(np.argmax(s))
            sel.append(xy[j])
        if len(sel) < 2:
            return None
        return np.asarray(sel, dtype=np.float64)
    return None


def build_wall_lines(columns,
                     pts,
                     source='extreme_in_columns',
                     chain_dx=0.6,
                     chain_dy=3.0,
                     min_length=2.0,
                     min_points=3):
    clusters = cluster_columns_spatial(columns, chain_dx, chain_dy)
    accepted = {}
    rejected = {}
    for i, cluster in enumerate(clusters):
        if len(cluster) < 2:
            continue
        xy = _select_points_for_cluster(cluster, pts, source)
        if xy is None or len(xy) < 2:
            continue
        line = fit_line_pca(xy)
        if line is None:
            continue
        mean_x = float(np.mean([c['x_center'] for c in cluster]))
        line['cluster_id'] = i
        line['n_columns'] = len(cluster)
        line['source'] = source
        line['mean_x'] = mean_x
        line['fit_xy'] = np.asarray(xy, dtype=np.float64)
        line['label'] = f'wall_{i}_meanX{mean_x:+.2f}'
        rej = None
        if line['length'] < 1e-3:
            rej = 'degenerate'
        elif line['length'] < min_length:
            rej = f'length {line["length"]:.2f}<{min_length}'
        elif line['n_points'] < min_points:
            rej = f'points {line["n_points"]}<{min_points}'
        if rej is not None:
            line['reject_reason'] = rej
            rejected[line['label']] = line
            continue
        accepted[line['label']] = line
    return accepted, rejected


#  МЕРДЖ линий


def _line_params(ln):
    c = np.asarray(ln['point'], dtype=np.float64)
    d_vec = np.asarray(ln['direction'], dtype=np.float64)
    if d_vec[1] < 0:
        d_vec = -d_vec
    theta = float(np.arctan2(d_vec[0], d_vec[1]))
    d_off = float(c[0] * np.cos(theta) - c[1] * np.sin(theta))
    return theta, d_off


def premerge_colinear_segments(wall_lines,
                               side_check=0.5,
                               dx_tol=0.8,
                               angle_tol_deg=6.0,
                               min_points=3,
                               verbose=False):
    items = list(wall_lines.values())
    if not items:
        return wall_lines
    N = len(items)
    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    params = [_line_params(ln) for ln in items]
    angle_tol = float(np.deg2rad(angle_tol_deg))

    for i in range(N):
        ti, di_off = params[i]
        mxi = items[i]['mean_x']
        for j in range(i + 1, N):
            mxj = items[j]['mean_x']
            if side_check > 0:
                if (mxi < -side_check and mxj > side_check) or \
                   (mxi > side_check and mxj < -side_check):
                    continue
            tj, dj_off = params[j]
            dt = abs(ti - tj)
            dt = min(dt, abs(np.pi - dt))
            if dt > angle_tol:
                continue
            dd = abs(di_off - dj_off)
            if dd > dx_tol:
                continue
            union(i, j)

    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(items[i])

    merged = {}
    for k, g in enumerate(groups.values()):
        if len(g) == 1:
            ln = g[0]
            merged[ln['label']] = ln
            continue
        all_xy = []
        for ln in g:
            xy = ln.get('fit_xy')
            if xy is not None and len(xy):
                all_xy.append(np.asarray(xy, dtype=np.float64))
        if not all_xy:
            continue
        all_xy = np.concatenate(all_xy, axis=0)
        if len(all_xy) < min_points:
            continue
        line = fit_line_pca(all_xy)
        if line is None:
            continue
        segments = fit_polyline_pca(all_xy, seg_len=8.0, min_pts=2)
        if not segments:
            segments = [{
                'p_start': np.asarray(line['p_start']),
                'p_end':   np.asarray(line['p_end']),
                'angle_deg': line['angle_deg'],
                'rms': line['rms'], 'n_points': line['n_points'],
                'length': line['length'],
            }]
        total_len = float(sum(float(ln.get('length', 0.0)) for ln in g))
        mean_x = float(np.mean([ln['mean_x'] for ln in g]))
        label = f'wall_col{k}_meanX{mean_x:+.2f}'
        merged[label] = {
            'label': label,
            'source': 'colinear-merge',
            'mean_x': mean_x,
            'p_start': line['p_start'],
            'p_end': line['p_end'],
            'length': total_len,
            'rms': line['rms'],
            'angle_deg': line['angle_deg'],
            'n_points': line['n_points'],
            'n_columns': int(sum(int(ln.get('n_columns', 0)) for ln in g)),
            'merged_from': [ln['label'] for ln in g],
            'n_segments': len(g),
            'segments': segments,
            'fit_xy': all_xy,
            'point': line['point'],
            'direction': line['direction'],
        }
        if verbose:
            log(f"  colinear-merge -> {label}  "
                f"from {len(g)} segments  L={total_len:.1f} м  "
                f"rms={line['rms']:.3f}")
    return merged


def merge_lines_by_clustering(wall_lines, eps=3.0, min_points=3,
                              seg_len=8.0, side_check=0.5,
                              cos_min=0.4, verbose=False):
    items = list(wall_lines.values())
    if not items:
        return wall_lines
    all_pts = []
    labels = []
    mean_xs = []
    dirs = []
    for ln in items:
        xy = ln['fit_xy']
        if xy is None or len(xy) == 0:
            continue
        d = np.asarray(ln['direction'], dtype=np.float64)
        nrm = np.linalg.norm(d)
        if nrm > 1e-9:
            d = d / nrm
        for p in xy:
            all_pts.append([float(p[0]), float(p[1])])
            labels.append(ln['label'])
            mean_xs.append(ln['mean_x'])
            dirs.append(d)
    if len(all_pts) < 2:
        return wall_lines
    all_pts = np.asarray(all_pts, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64)
    N = len(all_pts)
    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(N):
        for j in range(i + 1, N):
            mxi, mxj = mean_xs[i], mean_xs[j]
            if side_check > 0:
                if (mxi < -side_check and mxj > side_check) or \
                   (mxi > side_check and mxj < -side_check):
                    continue
            delta = all_pts[j] - all_pts[i]
            d = float(np.linalg.norm(delta))
            if d > eps:
                continue
            if d > 1e-9:
                u = delta / d
                ca = abs(float(np.dot(u, dirs[i])))
                cb = abs(float(np.dot(u, dirs[j])))
                if ca < cos_min or cb < cos_min:
                    continue
            union(i, j)
    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    merged = {}
    for k, idxs in enumerate(groups.values()):
        if len(idxs) < min_points:
            continue
        xy = all_pts[idxs]
        segments = fit_polyline_pca(xy, seg_len=seg_len, min_pts=2)
        if not segments:
            continue
        line = fit_line_pca(xy)
        if line is None:
            continue
        p_start = segments[0]['p_start']
        p_end = segments[-1]['p_end']
        total_len = float(sum(s['length'] for s in segments))
        rms_mean = float(np.mean([s['rms'] for s in segments]))
        n_pts = int(sum(s['n_points'] for s in segments))
        dx = p_end[0] - p_start[0]
        dy_ = p_end[1] - p_start[1]
        overall_angle = float(np.degrees(np.arctan2(dx, abs(dy_)))) \
            if abs(dy_) > 1e-9 else 0.0
        mean_x = float(np.mean([mean_xs[i] for i in idxs]))
        src = sorted({labels[i] for i in idxs})
        n_cols = 0
        for ln in items:
            if ln['label'] in src:
                n_cols += ln['n_columns']
        label = f'wall_k{k}_meanX{mean_x:+.2f}'
        merged[label] = {
            'label': label, 'source': 'cluster+polyline',
            'mean_x': mean_x,
            'p_start': p_start, 'p_end': p_end,
            'length': total_len, 'rms': rms_mean,
            'angle_deg': overall_angle,
            'n_points': n_pts, 'n_columns': int(n_cols),
            'merged_from': src,
            'segments': segments, 'n_segments': len(segments),
            'fit_xy': xy,
            'point': line['point'], 'direction': line['direction'],
        }
        if verbose:
            log(f"  cluster -> {label}  from {src}  "
                f"сегментов={len(segments)}  L={total_len:.1f} м")
    return merged


#  Safe-zone


def compute_distance_to_lines(pts_xy,
                              wall_lines,
                              min_length=10.0,
                              max_rms=0.15):
    N = len(pts_xy)
    min_dist = np.full(N, np.inf, dtype=np.float64)
    n_reliable = 0
    for key, ln in wall_lines.items():
        if ln.get('length', 0.0) < min_length:
            continue
        if ln.get('rms', 1.0) > max_rms:
            continue
        n_reliable += 1
        segs = ln.get('segments')
        if not segs:
            segs = [{'p_start': ln['p_start'], 'p_end': ln['p_end']}]
        for s in segs:
            p0 = np.asarray(s['p_start'], dtype=np.float64)
            p1 = np.asarray(s['p_end'], dtype=np.float64)
            ab = p1 - p0
            ab2 = float(ab[0] * ab[0] + ab[1] * ab[1])
            if ab2 < 1e-12:
                continue
            ap = pts_xy - p0
            t = (ap[:, 0] * ab[0] + ap[:, 1] * ab[1]) / ab2
            t = np.clip(t, 0.0, 1.0)
            proj_x = p0[0] + t * ab[0]
            proj_y = p0[1] + t * ab[1]
            dx = pts_xy[:, 0] - proj_x
            dy = pts_xy[:, 1] - proj_y
            d = np.sqrt(dx * dx + dy * dy)
            min_dist = np.minimum(min_dist, d)
    return min_dist, n_reliable


def classify_safe_zone(pts_xy, wall_lines,
                       thickness=0.30,
                       min_length=10.0, max_rms=0.15,
                       verbose=False, quiet=False):
    min_dist, n_reliable = compute_distance_to_lines(
        pts_xy, wall_lines,
        min_length=min_length, max_rms=max_rms)
    safe = min_dist <= thickness
    if verbose and not quiet:
        log(f"  safe-zone: надёжных линий={n_reliable} "
            f"(len>={min_length}, rms<={max_rms}), "
            f"ширина={thickness} м")
    return safe


#  Bend path + extrude

def build_bend_path(wall_lines,
                    fwd_min,
                    fwd_max,
                    side_filter='auto',
                    no_data_angle=0.0,
                    min_seg_length=5.0,
                    smooth_window=3,
                    max_rate_deg_per_m=1.0,
                    max_extrapolation=30.0,
                    theta_poly_deg=2,
                    full_debug=False,
                    verbose=False, quiet=False,
                    bend_max_rms=0.15,
                    bend_max_mean_x=2.0,
                    bend_min_pts=15):
    all_segs = []
    filtered_out = []

    if full_debug and not quiet:
        dlog(f"  [debug] все стены на кадре ({len(wall_lines)}):")
        for kk, lnk in wall_lines.items():
            dlog(f"    {kk:<36}  L={lnk.get('length', 0.0):7.2f}  "
                 f"mean_x={lnk['mean_x']:+.2f}  "
                 f"rms={lnk.get('rms', 0.0):.3f}  "
                 f"n_pts={lnk.get('n_points', 0):>4}  "
                 f"n_segs={lnk.get('n_segments', '?')}")

    selected_key = None
    if side_filter == 'longest':
        best_score = -1.0
        best_info = {}

        for key, ln in wall_lines.items():
            L = float(ln.get('length', 0.0))
            rms = float(ln.get('rms', 1.0))
            mean_x = abs(float(ln['mean_x']))
            n_pts = int(ln.get('n_points', 0))

            if L < min_seg_length:
                continue
            if rms > bend_max_rms:
                if full_debug and not quiet:
                    dlog(f"    [reject rms] {key}  rms={rms:.3f} > {bend_max_rms}")
                continue
            if mean_x > bend_max_mean_x:
                if full_debug and not quiet:
                    dlog(f"    [reject mean_x] {key}  |mean_x|={mean_x:.2f} > {bend_max_mean_x}")
                continue
            if n_pts < bend_min_pts:
                if full_debug and not quiet:
                    dlog(f"    [reject n_pts] {key}  n_pts={n_pts} < {bend_min_pts}")
                continue

            rms_range = max(bend_max_rms - 0.05, 0.01)
            rms_penalty = max(0.0, (rms - 0.05) / rms_range)
            x_penalty = max(0.0, abs(mean_x - 1.5) / 1.0)
            x_penalty = min(x_penalty, 1.0)
            score = L * (1.0 - rms_penalty) * (1.0 - x_penalty)

            if full_debug and not quiet:
                dlog(f"    [score] {key}  L={L:.1f}  rms={rms:.3f}  "
                     f"|mx|={mean_x:.2f}  pts={n_pts}  "
                     f"score={score:.1f}")

            if score > best_score:
                best_score = score
                selected_key = key
                best_info = {'L': L, 'rms': rms, 'mean_x': ln['mean_x'],
                             'pts': n_pts, 'score': score}

        if selected_key is None:
            if not quiet:
                log("  bend side (longest): надёжных стен не найдено, "
                    "использую no_data_angle")
            return [(fwd_min, no_data_angle), (fwd_max, no_data_angle)]

        if not quiet:
            log(f"  bend side (reliable_longest): {selected_key}  "
                f"L={best_info['L']:.1f} м  "
                f"mean_x={best_info['mean_x']:+.2f}  "
                f"rms={best_info['rms']:.3f}  "
                f"pts={best_info['pts']}  "
                f"score={best_info['score']:.1f}")

    for key, ln in wall_lines.items():
        if selected_key is not None and key != selected_key:
            continue
        segs = ln.get('segments')
        if not segs:
            segs = [{'p_start': ln['p_start'], 'p_end': ln['p_end'],
                     'angle_deg': ln['angle_deg']}]
        for s in segs:
            y_a = min(float(s['p_start'][1]), float(s['p_end'][1]))
            y_b = max(float(s['p_start'][1]), float(s['p_end'][1]))
            length = y_b - y_a
            if length < 1e-3:
                continue
            if length < min_seg_length:
                filtered_out.append((y_a, y_b,
                                     float(s['angle_deg']),
                                     ln['mean_x']))
                continue
            n_pts_seg = int(s.get('n_points', 0))
            all_segs.append({'y_start': y_a, 'y_end': y_b,
                             'angle': float(s['angle_deg']),
                             'mean_x': ln['mean_x'],
                             'n_points': n_pts_seg})

    if not quiet and filtered_out:
        log(f"  отфильтровано {len(filtered_out)} коротких сегментов "
            f"(<{min_seg_length} м)")

    if not all_segs:
        return [(fwd_min, no_data_angle), (fwd_max, no_data_angle)]

    if side_filter == 'auto':
        left_len = sum(s['y_end'] - s['y_start']
                       for s in all_segs
                       if s['mean_x'] < 0
                       and s.get('n_points', 0) >= bend_min_pts)
        right_len = sum(s['y_end'] - s['y_start']
                        for s in all_segs
                        if s['mean_x'] >= 0
                        and s.get('n_points', 0) >= bend_min_pts)
        side_filter = 'left' if left_len >= right_len else 'right'
        if not quiet:
            log(f"  bend side (auto): {side_filter}")

    if side_filter == 'left':
        all_segs = [s for s in all_segs if s['mean_x'] < 0]
    elif side_filter == 'right':
        all_segs = [s for s in all_segs if s['mean_x'] >= 0]

    if not all_segs:
        return [(fwd_min, no_data_angle), (fwd_max, no_data_angle)]

    all_segs.sort(key=lambda s: s['y_start'])

    if full_debug and not quiet:
        dlog(f"  [debug] сегменты ДО полинома ({len(all_segs)}):")
        for i, s in enumerate(all_segs):
            dlog(f"    [{i:2d}]  Y=[{s['y_start']:7.2f},{s['y_end']:7.2f}]  "
                 f"θ_raw={s['angle']:+7.2f}°  n_pts={s.get('n_points', 0)}")

    if theta_poly_deg > 0 and len(all_segs) >= theta_poly_deg + 1:
        y_mids = np.array([0.5 * (s['y_start'] + s['y_end'])
                           for s in all_segs])
        th_vals = np.array([s['angle'] for s in all_segs])
        w = np.array([max(float(s.get('n_points', 1)), 1.0)
                      for s in all_segs], dtype=np.float64)
        try:
            coeffs = np.polyfit(y_mids, th_vals, theta_poly_deg, w=w)
            poly = np.poly1d(coeffs)
            for s in all_segs:
                y_mid = 0.5 * (s['y_start'] + s['y_end'])
                s['angle'] = float(poly(y_mid))
            if not quiet:
                log(f"  theta-poly-fit deg={theta_poly_deg}  "
                    f"n_segs={len(all_segs)}  w=[{w.min():.0f}..{w.max():.0f}]")
        except Exception as e:
            if not quiet:
                log(f"  theta-poly-fit failed: {e}")

    path = []
    y_first = all_segs[0]['y_start']
    if y_first > fwd_min:
        path.append((fwd_min, no_data_angle))
        path.append((y_first, no_data_angle))

    prev_end = y_first
    prev_angle = no_data_angle
    prev_mean_x = None

    for seg in all_segs:
        y_a = seg['y_start']
        y_b = seg['y_end']
        seg_angle = seg['angle']
        seg_mean_x = seg['mean_x']

        if prev_mean_x is not None and \
           (prev_mean_x < 0) != (seg_mean_x < 0):
            break

        if y_a > prev_end + 1e-3:
            gap = y_a - prev_end
            rate_obs = (seg_angle - prev_angle) / gap if gap > 1e-9 else 0.0
            if gap > 1e-9 and abs(rate_obs) > max_rate_deg_per_m:
                sign = np.sign(rate_obs)
                seg_angle = prev_angle + sign * max_rate_deg_per_m * gap
            path.append((prev_end, prev_angle))
            path.append((y_a, seg_angle))
        else:
            path.append((y_a, seg_angle))

        path.append((y_b, seg_angle))
        prev_end = y_b
        prev_angle = seg_angle
        prev_mean_x = seg_mean_x

    tail_end = min(fwd_max, prev_end + max_extrapolation)
    if tail_end > prev_end + 1e-6:
        path.append((tail_end, prev_angle))

    path.sort(key=lambda p: p[0])
    dedup = []
    for p in path:
        if dedup and abs(dedup[-1][0] - p[0]) < 1e-6:
            dedup[-1] = p
        else:
            dedup.append(p)
    path = dedup

    if smooth_window >= 3 and len(path) >= smooth_window:
        ys_p = np.array([p[0] for p in path])
        th_p = np.array([p[1] for p in path])
        n_uniform = max(len(path) * 5, 50)
        y_uniform = np.linspace(ys_p[0], ys_p[-1], n_uniform)
        th_uniform = np.interp(y_uniform, ys_p, th_p)
        k = smooth_window
        kernel = np.ones(k) / k
        th_smooth = np.convolve(th_uniform, kernel, mode='same')
        half = k // 2
        th_smooth[:half] = th_uniform[:half]
        th_smooth[-half:] = th_uniform[-half:]
        path = list(zip(y_uniform.tolist(), th_smooth.tolist()))

    if not quiet:
        log(f"  bend path: {len(path)} точек")

    return path


def smooth_bend_path_temporal(new_path,
                              prev_path,
                              alpha=0.5,
                              n_grid=200,
                              max_gap=5.0):
    if prev_path is None or len(prev_path) < 2 or len(new_path) < 2:
        return new_path
    y_n = np.array([p[0] for p in new_path], dtype=np.float64)
    th_n = np.array([p[1] for p in new_path], dtype=np.float64)
    y_p = np.array([p[0] for p in prev_path], dtype=np.float64)
    th_p = np.array([p[1] for p in prev_path], dtype=np.float64)
    y0 = max(float(y_n.min()), float(y_p.min()))
    y1 = min(float(y_n.max()), float(y_p.max()))
    if y1 - y0 < max_gap:
        return new_path
    y_grid = np.linspace(y0, y1, n_grid)
    th_new_i = np.interp(y_grid, y_n, th_n)
    th_prev_i = np.interp(y_grid, y_p, th_p)
    th_smooth = alpha * th_new_i + (1.0 - alpha) * th_prev_i
    return list(zip(y_grid.tolist(), th_smooth.tolist()))


def build_bend_tube(bend_path,
                    train_half_width,
                    forward_sign,
                    z_min,
                    z_max):
    sections = []
    if not bend_path:
        return sections
    p = np.array([0.0, forward_sign * float(bend_path[0][0])])
    prev_y = float(bend_path[0][0])
    for i, (y_abs, th_deg) in enumerate(bend_path):
        y_abs = float(y_abs)
        if i > 0:
            L = y_abs - prev_y
            th_mid = 0.5 * (bend_path[i - 1][1] + th_deg)
            th_mid_rad = np.deg2rad(th_mid)
            d_mid = np.array([np.sin(th_mid_rad),
                              forward_sign * np.cos(th_mid_rad)])
            p = p + d_mid * L
            prev_y = y_abs
        th_rad = np.deg2rad(th_deg)
        cos_t = np.cos(th_rad)
        sin_t = np.sin(th_rad)
        d_here = np.array([sin_t, forward_sign * cos_t])
        perp = np.array([cos_t, -forward_sign * sin_t])
        p_l = p + perp * train_half_width
        p_r = p - perp * train_half_width
        sections.append({
            'y_abs': y_abs,
            'theta_deg': float(th_deg),
            'center': p.copy(),
            'direction': d_here.copy(),
            'll': np.array([p_l[0], p_l[1], z_min]),
            'rr': np.array([p_r[0], p_r[1], z_min]),
            'll_u': np.array([p_l[0], p_l[1], z_max]),
            'rr_u': np.array([p_r[0], p_r[1], z_max]),
        })
    return sections


def points_in_tube(pts,
                   sections,
                   train_half_width,
                   train_fwd_min,
                   train_fwd_max,
                   z_min,
                   z_max,
                   forward_sign):
    if not sections or len(sections) < 2:
        return np.zeros(len(pts), dtype=bool)
    xy_pts = pts[:, :2]
    y_abs_pts = np.abs(pts[:, 1])
    in_range = (y_abs_pts >= train_fwd_min) & (y_abs_pts <= train_fwd_max)
    in_z = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    centers = np.array([[s['center'][0], s['center'][1]]
                        for s in sections])
    n = len(xy_pts)
    min_dist = np.full(n, np.inf, dtype=np.float64)
    for i in range(len(centers) - 1):
        a = centers[i]
        b = centers[i + 1]
        ab = b - a
        ab2 = float(ab[0] * ab[0] + ab[1] * ab[1])
        if ab2 < 1e-12:
            continue
        ap = xy_pts - a
        t = (ap[:, 0] * ab[0] + ap[:, 1] * ab[1]) / ab2
        t = np.clip(t, 0.0, 1.0)
        proj_x = a[0] + t * ab[0]
        proj_y = a[1] + t * ab[1]
        dx = xy_pts[:, 0] - proj_x
        dy = xy_pts[:, 1] - proj_y
        d = np.sqrt(dx * dx + dy * dy)
        min_dist = np.minimum(min_dist, d)
    return in_range & in_z & (min_dist <= train_half_width)


#  Обработка одного кадра
def process_one_frame(pts_raw, args, forward_sign, frame_label="",
                      prev_bend_path=None):
    t0 = time.perf_counter()
    quiet = args.quiet

    if args.voxel > 0:
        n_before = len(pts_raw)
        if args.voxel_keep_extremes:
            pts_raw = voxel_downsample_extremes(
                pts_raw, args.voxel, metric=args.voxel_metric)
        else:
            pts_raw = voxel_downsample_np(pts_raw, args.voxel)

    pts = pts_raw[:, :3].astype(np.float64)
    total = len(pts)

    y_abs = np.abs(pts[:, 1])
    x = pts[:, 0]
    z = pts[:, 2]

    fwd_max = args.fwd_max if args.fwd_max is not None else float(y_abs.max())
    train_fwd_min = (args.train_fwd_min
                     if args.train_fwd_min is not None else args.fwd_min)
    train_fwd_max = (args.train_fwd_max
                     if args.train_fwd_max is not None else fwd_max)

    res = detect_wall_candidates(
        pts, half_width=args.half_width,
        fwd_min=args.fwd_min, fwd_max=fwd_max,
        z_min=args.z_min, z_max=args.z_max,
        tol_x=args.tol_x,
        search_all_x=(args.search_x_range == "all"),
        y_slice=args.y_slice, y_overlap=args.y_overlap,
        y_step=args.y_step,
        y_slice_per_meter=args.y_slice_per_meter,
        min_points_in_slice=args.min_points_in_slice,
        cell_per_meter=args.cell_per_meter,
        min_cell=args.min_cell, max_cell=args.max_cell,
        min_coverage=args.min_coverage,
        min_occupied_bins=args.min_occupied,
        edge_frac=args.edge_frac,
        min_z_span=args.min_z_span,
        z_filter=args.z_filter,
        min_slices_for_wall=args.min_slices,
        x_merge_tol=args.x_merge_tol,
        max_y_spread_in_column=args.y_spread_limit,
        quiet=args.quiet)
    wall_mask = res['wall_mask']
    columns = res['columns']

    wall_lines = {}
    wall_lines_rejected = {}
    if args.fit_line:
        wall_lines, wall_lines_rejected = build_wall_lines(
            columns, pts,
            source=args.line_source,
            chain_dx=args.chain_dx,
            chain_dy=args.chain_dy,
            min_length=args.min_line_length,
            min_points=args.min_line_points)

    if args.merge_lines and wall_lines and args.merge_colinear:
        wall_lines = premerge_colinear_segments(
            wall_lines,
            side_check=args.merge_side_check,
            dx_tol=args.merge_colinear_dx,
            angle_tol_deg=args.merge_colinear_angle_deg,
            min_points=args.min_line_points,
            verbose=not args.quiet and args.debug)

    if args.merge_lines and wall_lines:
        if args.merge_method == 'cluster':
            wall_lines = merge_lines_by_clustering(
                wall_lines,
                eps=args.merge_cluster_eps,
                min_points=args.min_line_points,
                seg_len=args.curve_seg_len,
                side_check=args.merge_side_check,
                cos_min=args.merge_cos_min,
                verbose=not args.quiet and args.debug)

    bend_path = None
    bend_sections = None
    inside_bent = np.zeros(total, dtype=bool)

    if args.bend_by_segments and wall_lines:
        if _DEBUG_FH is not None:
            print(f"\n=== frame {frame_label.strip()} ===",
                  file=_DEBUG_FH, flush=True)
        bend_path = build_bend_path(
            wall_lines,
            fwd_min=args.fwd_min, fwd_max=fwd_max,
            side_filter=args.bend_side,
            no_data_angle=args.no_data_angle,
            min_seg_length=args.bend_min_seg_length,
            smooth_window=args.bend_smooth_window,
            max_rate_deg_per_m=args.bend_max_rate_deg_per_m,
            max_extrapolation=args.bend_max_extrapolation,
            theta_poly_deg=args.bend_theta_poly_deg,
            full_debug=args.bend_full_debug or (args.bend_debug_file is not None),
            verbose=not args.quiet,
            quiet=args.quiet,
            bend_max_rms=args.bend_max_rms,
            bend_max_mean_x=args.bend_max_mean_x,
            bend_min_pts=args.bend_min_pts)

        if bend_path is not None and not args.bend_temporal_off:
            bend_path = smooth_bend_path_temporal(
                bend_path, prev_bend_path,
                alpha=args.bend_temporal_alpha)

        bend_sections = build_bend_tube(
            bend_path,
            train_half_width=args.train_half_width,
            forward_sign=forward_sign,
            z_min=args.z_min, z_max=args.z_max)
        inside_bent = points_in_tube(
            pts, bend_sections,
            train_half_width=args.train_half_width,
            train_fwd_min=train_fwd_min,
            train_fwd_max=train_fwd_max,
            z_min=args.z_min, z_max=args.z_max,
            forward_sign=forward_sign)
    else:
        m_y = (y_abs >= args.fwd_min) & (y_abs <= fwd_max)
        m_x = np.abs(x) <= args.half_width
        m_z = (z >= args.z_min) & (z <= args.z_max)
        inside_bent = m_y & m_x & m_z

    safe_mask = classify_safe_zone(
        pts[:, :2], wall_lines,
        thickness=args.safe_zone_thickness,
        min_length=args.safe_zone_min_length,
        max_rms=args.safe_zone_max_rms,
        verbose=args.debug, quiet=args.quiet)

    obstacle_mask = inside_bent & ~wall_mask & ~safe_mask

    elapsed = time.perf_counter() - t0

    return {
        'pts': pts,
        'obstacles': pts[obstacle_mask],
        'safe': pts[safe_mask],
        'wall': pts[wall_mask],
        'inside_bent': inside_bent,
        'wall_mask': wall_mask,
        'safe_mask': safe_mask,
        'obstacle_mask': obstacle_mask,
        'bend_sections': bend_sections,
        'bend_path': bend_path,
        'wall_lines': wall_lines,
        'wall_lines_rejected': wall_lines_rejected,
        'columns': columns,
        'fwd_max': fwd_max,
        'elapsed_ms': elapsed * 1000.0,
    }


#  Вывод результата
def format_frame_summary(i, result, total_pts):
    n_obs = len(result['obstacles'])
    n_tube = int(result['inside_bent'].sum())
    n_wall = int(result['wall_mask'].sum())
    n_safe = int(result['safe_mask'].sum())
    line = (f"[frame {i:06d}] pts={total_pts:>7}  "
            f"tube={n_tube:>6}  wall={n_wall:>6}  safe={n_safe:>6}  "
            f"obstacles={n_obs:>5}")
    if n_obs > 0:
        pts_obs = result['obstacles']
        r = np.linalg.norm(pts_obs[:, :2], axis=1)
        line += (f"  |Y|=[{np.abs(pts_obs[:,1]).min():.1f},"
                 f"{np.abs(pts_obs[:,1]).max():.1f}]  "
                 f"X=[{pts_obs[:,0].min():+.2f},{pts_obs[:,0].max():+.2f}]")
    return line


def save_obstacles_npy(result, out_dir, frame_idx, quiet=False):
    if len(result['obstacles']) == 0:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_{frame_idx:06d}.npy"
    np.save(path, result['obstacles'])
    if not quiet:
        log(f"  obstacles -> {path}  ({len(result['obstacles'])})")
    return path


#  main
def main():
    global _DEBUG_FH

    ap = argparse.ArgumentParser(
        description="Габарит-труба + safe-zone → точки препятствий. "
                    "Только ROS2 bag на входе.")

    # вход
    ap.add_argument("--bag", type=str, required=True,
                    help="путь к .db3 rosbag2")
    ap.add_argument("--topic", type=str, default="/lidar_points")
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-end", type=int, default=None)

    # вокселизация
    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--voxel-keep-extremes", action="store_true",
                    default=True)
    ap.add_argument("--voxel-metric",
                    choices=["x_abs", "y_abs", "xy_radial", "xy_max",
                             "x_max", "x_min"],
                    default="x_abs")
    ap.add_argument("--quiet", action="store_true")

    # геометрия
    ap.add_argument("--half-width", type=float, default=40.0)
    ap.add_argument("--fwd-min", type=float, default=1.0)
    ap.add_argument("--fwd-max", type=float, default=None)
    ap.add_argument("--z-min", type=float, default=-1.0)
    ap.add_argument("--z-max", type=float, default=2.0)
    ap.add_argument("--tol-x", type=float, default=0.3)
    ap.add_argument("--search-x-range", choices=["all", "edges"],
                    default="all")
    ap.add_argument("--y-slice", type=float, default=0.5)
    ap.add_argument("--y-overlap", type=float, default=0.5)
    ap.add_argument("--y-step", type=float, default=0.3)
    ap.add_argument("--y-slice-per-meter", type=float, default=0.03)
    ap.add_argument("--min-points-in-slice", type=int, default=4)
    ap.add_argument("--y-spread-limit", type=float, default=None)
    ap.add_argument("--cell-per-meter", type=float, default=0.015)
    ap.add_argument("--min-cell", type=float, default=0.05)
    ap.add_argument("--max-cell", type=float, default=0.5)

    # критерии колонн
    ap.add_argument("--min-coverage", type=float, default=0.3)
    ap.add_argument("--min-occupied", type=int, default=2)
    ap.add_argument("--edge-frac", type=float, default=0.25)
    ap.add_argument("--min-z-span", type=float, default=1.0)
    ap.add_argument("--z-filter", choices=["span", "edges", "both"],
                    default="span")

    ap.add_argument("--min-slices", type=int, default=2)
    ap.add_argument("--x-merge-tol", type=float, default=0.4)
    ap.add_argument("--forward-sign", type=int, choices=[1, -1], default=-1)

    # линии стен
    ap.add_argument("--fit-line", action="store_true", default=True)
    ap.add_argument("--line-source",
                    choices=["extreme_in_columns", "extreme_perp",
                             "centers", "points"],
                    default="extreme_in_columns")
    ap.add_argument("--chain-dx", type=float, default=0.8)
    ap.add_argument("--chain-dy", type=float, default=3.0)
    ap.add_argument("--min-line-length", type=float, default=2.0)
    ap.add_argument("--min-line-points", type=int, default=3)

    # мердж
    ap.add_argument("--merge-lines", action="store_true", default=True)
    ap.add_argument("--merge-method", choices=["cluster", "endpoints"],
                    default="cluster")
    ap.add_argument("--merge-cluster-eps", type=float, default=3.0)
    ap.add_argument("--merge-side-check", type=float, default=0.5)
    ap.add_argument("--merge-cos-min", type=float, default=0.4)
    ap.add_argument("--curve-seg-len", type=float, default=8.0)

    ap.add_argument("--no-merge-colinear", dest="merge_colinear",
                    action="store_false", default=True)
    ap.add_argument("--merge-colinear-dx", type=float, default=0.8)
    ap.add_argument("--merge-colinear-angle-deg", type=float, default=6.0)

    # bend
    ap.add_argument("--bend-by-segments", action="store_true", default=True)
    ap.add_argument("--bend-side",
                    choices=["auto", "left", "right", "longest"],
                    default="longest")
    ap.add_argument("--no-data-angle", type=float, default=0.0)
    ap.add_argument("--bend-min-seg-length", type=float, default=5.0)
    ap.add_argument("--bend-smooth-window", type=int, default=3)
    ap.add_argument("--bend-max-rate-deg-per-m", type=float, default=1.0)
    ap.add_argument("--bend-max-extrapolation", type=float, default=30.0)
    ap.add_argument("--bend-theta-poly-deg", type=int, default=2)

    ap.add_argument("--bend-max-rms", type=float, default=0.15)
    ap.add_argument("--bend-max-mean-x", type=float, default=2.0)
    ap.add_argument("--bend-min-pts", type=int, default=15)

    ap.add_argument("--bend-full-debug", action="store_true")
    ap.add_argument("--bend-debug-file", type=str, default=None)

    ap.add_argument("--bend-temporal-alpha", type=float, default=0.5)
    ap.add_argument("--bend-temporal-off", action="store_true",
                    default=True)

    # труба и safe-zone
    ap.add_argument("--train-half-width", type=float, default=1.05)
    ap.add_argument("--train-fwd-min", type=float, default=None)
    ap.add_argument("--train-fwd-max", type=float, default=None)
    ap.add_argument("--safe-zone-thickness", type=float, default=0.30)
    ap.add_argument("--safe-zone-min-length", type=float, default=10.0)
    ap.add_argument("--safe-zone-max-rms", type=float, default=0.15)

    # вывод
    ap.add_argument("--output-dir", type=str, default=None,
                    help="куда сохранять obstacle-точки по кадрам "
                         "(frame_XXXXXX.npy). Если не задан — только сводка.")
    ap.add_argument("--no-save-empty", action="store_true", default=True,
                    help="не сохранять npy для кадров без препятствий")

    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    if not HAS_ROS2:
        log("ОШИБКА: ROS2-библиотеки не установлены.")
        sys.exit(1)

    if args.bend_debug_file:
        try:
            _DEBUG_FH = open(args.bend_debug_file, "a",
                             encoding="utf-8", buffering=1)
            log(f"[main] debug-лог пишется в {args.bend_debug_file}")
            args.bend_full_debug = True
        except Exception as e:
            log(f"[warn] не могу открыть debug-файл "
                f"{args.bend_debug_file}: {e}")
            _DEBUG_FH = None

    if args.min_slices < 2 and not args.quiet:
        log("[warn] --min-slices=1 — режим отладки.")
    if args.bend_temporal_off and not args.quiet:
        log("[warn] bend-temporal-off: θ(Y) не сглаживается между кадрами.")

    output_dir = Path(args.output_dir) if args.output_dir else None

    reader = BagReader(args.bag, args.topic, quiet=args.quiet)
    i_start = args.frame_start
    total_frames = reader.count()
    i_end = args.frame_end if args.frame_end is not None else total_frames

    forward_sign = args.forward_sign
    if forward_sign is None:
        if not args.quiet:
            log("[main] probe: определяю forward_sign")
        probe = BagReader(args.bag, args.topic, quiet=args.quiet)
        forward_sign = -1
        fwd_max_val = (args.fwd_max if args.fwd_max is not None
                       else 1e9)
        for _, pts_raw in probe.iter_frames(
                i_start, min(i_start + 1, i_end)):
            y_abs = np.abs(pts_raw[:, 1])
            in_g = (y_abs >= args.fwd_min) & \
                   (y_abs <= fwd_max_val) & \
                   (np.abs(pts_raw[:, 0]) <= args.half_width)
            if in_g.any():
                mean_y = float(np.mean(pts_raw[in_g, 1]))
                forward_sign = 1 if mean_y >= 0 else -1
            break
        probe.close()

    log(f"[main] Bag:    {args.bag}")
    log(f"[main] Топик:  {args.topic}")
    log(f"[main] forward_sign = {forward_sign:+d}")
    log(f"[main] Кадры:  [{i_start}, {i_end}) из {total_frames}")
    if output_dir:
        log(f"[main] output-dir = {output_dir}")

    t_batch = time.perf_counter()
    n_done = 0
    n_frames_with_obstacles = 0
    total_obstacle_points = 0
    prev_bend_path = None

    for i, pts_raw in reader.iter_frames(i_start, i_end):
        result = process_one_frame(
            pts_raw, args, forward_sign,
            frame_label=f"[frame {i:06d}] ",
            prev_bend_path=prev_bend_path)
        prev_bend_path = result.get('bend_path')

        if not args.quiet:
            log(format_frame_summary(i, result, len(pts_raw)))
        else:
            n_obs = len(result['obstacles'])
            if n_obs > 0:
                log(f"[frame {i:06d}] obstacles={n_obs}")

        if output_dir is not None and len(result['obstacles']) > 0:
            save_obstacles_npy(result, output_dir, i, quiet=args.quiet)

        if len(result['obstacles']) > 0:
            n_frames_with_obstacles += 1
            total_obstacle_points += len(result['obstacles'])

        n_done += 1

    reader.close()
    dt = time.perf_counter() - t_batch
    log("")
    log(f"=== Итого ===")
    log(f"  кадров обработано:    {n_done}")
    log(f"  кадров с препятствием: {n_frames_with_obstacles}")
    log(f"  всего точек-препятствий: {total_obstacle_points}")
    log(f"  время:                 {dt:.2f} с "
        f"({dt/max(n_done,1)*1000:.1f} мс/кадр)")

    if _DEBUG_FH is not None:
        _DEBUG_FH.close()
        _DEBUG_FH = None


if __name__ == "__main__":
    main()
