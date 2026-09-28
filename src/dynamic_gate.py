"""
Динамический створ для детектирования препятствий в тоннеле метро.

Ключевые идеи:
  1. Работаем только в переднем секторе по азимуту, суженном на
     sector_margin_deg внутрь от forward_half_angle_deg.
  2. Работаем только в допустимом по элевации секторе.
  3. Работаем только там, где фактическая высота точки над лидаром
     попадает в [z_min, z_max]: z = r * sin(el).
  4. Скорость оценивается по данным. Если данных мало —
     компенсация = 0, а НЕ default.
  5. Компенсация движения учитывает проекцию вектора скорости на
     направление луча (cos(az)*cos(el)).

Двухуровневая схема:
  - СИЛЬНАЯ аномалия: падение > strong_threshold.
  - СЛАБАЯ аномалия: падение в [weak_min, strong_threshold].
    Требует пространственного подтверждения + N/M.

Baseline (prev_range) обновляется по всему range image.

Вход:  облако точек (N,3+) или range image (H,W).
Выход: булева маска (H,W), True = подтверждённая аномалия.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class DynamicGate:
    def __init__(
        self,
        # геометрия
        azimuth_bins: int = 7200,
        elevation_bins: int = 128,
        forward_half_angle_deg: float = 50.0,
        sector_margin_deg: float = 5.0,
        elev_min_deg: float = -10.0,
        elev_max_deg: float = 15.0,
        elev_margin_deg: float = 1.5,
        # высота z над лидаром
        z_min: float = -1.0,
        z_max: float = 2.0,
        # дистанции
        min_range: float = 3.0,
        max_range: float = 200.0,
        # пороги
        strong_threshold: float = 2.5,
        weak_threshold_min: float = 1.5,
        # подтверждение
        n_confirm: int = 4,
        m_confirm: int = 6,
        neighbor_radius: int = 2,
        neighbor_min_count: int = 4,
        # движение
        motion_default: float = 0.0,
        motion_estimate: bool = True,
        motion_far_min_range: float = 10.0,
        motion_min_samples: int = 100,
        # baseline
        max_freeze_frames: int = 3,
        # отладка
        debug: bool = True,
        debug_every: int = 5,
    ) -> None:
        self.azimuth_bins = int(azimuth_bins)
        self.elevation_bins = int(elevation_bins)
        self.forward_half_angle_deg = float(forward_half_angle_deg)
        self.sector_margin_deg = float(sector_margin_deg)
        self.elev_min_deg = float(elev_min_deg)
        self.elev_max_deg = float(elev_max_deg)
        self.elev_margin_deg = float(elev_margin_deg)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.strong_thr = float(strong_threshold)
        self.weak_min = float(weak_threshold_min)
        self.n_confirm = int(n_confirm)
        self.m_confirm = int(m_confirm)
        self.neighbor_radius = int(neighbor_radius)
        self.neighbor_min_count = int(neighbor_min_count)
        self.motion_default = float(motion_default)
        self.motion_estimate = bool(motion_estimate)
        self.motion_far_min_range = float(motion_far_min_range)
        self.motion_min_samples = int(motion_min_samples)
        self.max_freeze_frames = int(max_freeze_frames)
        self.debug = bool(debug)
        self.debug_every = int(debug_every)

        # 1D маски по азимуту и элевации с внутренними отступами
        az_deg = (
            np.arange(self.azimuth_bins) / self.azimuth_bins * 360.0 - 180.0
        )
        el_deg = (
            np.arange(self.elevation_bins) / self.elevation_bins * 180.0 - 90.0
        )

        az_limit = self.forward_half_angle_deg - self.sector_margin_deg
        el_lo = self.elev_min_deg + self.elev_margin_deg
        el_hi = self.elev_max_deg - self.elev_margin_deg

        self.forward_mask_1d = np.abs(az_deg) <= az_limit
        self.elev_mask_1d = (el_deg >= el_lo) & (el_deg <= el_hi)
        self.sin_el_1d = np.sin(np.deg2rad(el_deg)).astype(np.float32)

        # карта проекции движения на направление луча
        az_rad = np.deg2rad(az_deg)
        el_rad = np.deg2rad(el_deg)
        AZ, EL = np.meshgrid(az_rad, el_rad)
        self.cos_proj = (np.cos(AZ) * np.cos(EL)).astype(np.float32)

        self.prev_range: np.ndarray | None = None
        self.confirm_buffer: deque[np.ndarray] = deque(maxlen=self.m_confirm)
        self.freeze_counter: np.ndarray | None = None
        self.frame_idx = 0
        self.est_motion = self.motion_default

    def _to_range_image(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2 and frame.shape[1] >= 3:
            return self._points_to_range_image(frame[:, :3])
        if frame.ndim == 2:
            return frame.astype(np.float32, copy=False)
        raise ValueError(f"Неизвестная форма кадра: {frame.shape}")

    def _points_to_range_image(self, xyz: np.ndarray) -> np.ndarray:
        H, W = self.elevation_bins, self.azimuth_bins
        if xyz.size == 0:
            return np.full((H, W), self.max_range, dtype=np.float32)
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        if xyz.size == 0:
            return np.full((H, W), self.max_range, dtype=np.float32)
        r = np.linalg.norm(xyz, axis=1)
        az = np.arctan2(xyz[:, 1], xyz[:, 0])
        el = np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1))
        az_idx = np.clip(
            np.floor((az + np.pi) / (2 * np.pi) * W).astype(np.int32),
            0, W - 1,
        )
        el_idx = np.clip(
            np.floor((el + np.pi / 2) / np.pi * H).astype(np.int32),
            0, H - 1,
        )
        img = np.full((H, W), self.max_range, dtype=np.float32)
        np.minimum.at(img, (el_idx, az_idx), r.astype(np.float32))
        return img

    def _spatial_confirm(self, mask: np.ndarray) -> np.ndarray:
        r = self.neighbor_radius
        count = np.zeros_like(mask, dtype=np.int32)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx == 0:
                    continue
                count += np.roll(
                    np.roll(mask, dy, axis=0), dx, axis=1
                ).astype(np.int32)
        return count >= self.neighbor_min_count

    def process(self, frame: np.ndarray) -> np.ndarray:
        range_img = self._to_range_image(frame)
        H, W = range_img.shape
        valid = (range_img > self.min_range) & (range_img < self.max_range)

        if self.prev_range is None or self.prev_range.shape != range_img.shape:
            self.prev_range = range_img.copy()
            self.freeze_counter = np.zeros_like(valid, dtype=np.int32)
            self.confirm_buffer.append(np.zeros_like(valid))
            self.frame_idx += 1
            return np.zeros_like(valid)

        valid_prev = (self.prev_range > self.min_range) & (
            self.prev_range < self.max_range
        )
        check_all = valid & valid_prev

        # сектор: forward × elevation
        sector_2d = (
            self.forward_mask_1d[np.newaxis, :]
            & self.elev_mask_1d[:, np.newaxis]
        )

        # z-фильтр по фактической дальности
        z_img = range_img * self.sin_el_1d[:, np.newaxis]
        z_ok = (z_img >= self.z_min) & (z_img <= self.z_max)

        check_fwd = check_all & sector_2d & z_ok

        # ---- оценка движения ----
        if self.motion_estimate:
            far = check_fwd & (self.prev_range > self.motion_far_min_range)
            if int(far.sum()) >= self.motion_min_samples:
                v_obs = float(np.median(self.prev_range[far] - range_img[far]))
                self.est_motion = float(np.clip(v_obs, -1.0, 5.0))
            else:
                self.est_motion = 0.0
        else:
            self.est_motion = self.motion_default

        delta = self.est_motion * self.cos_proj
        predicted = self.prev_range - delta
        drop = predicted - range_img
        drop = np.where(check_fwd, drop, -1e6)

        if self.debug and (self.frame_idx % self.debug_every == 0):
            d = drop[check_fwd]
            if d.size:
                print(
                    f"  [debug] f={self.frame_idx:3d} "
                    f"v_est={self.est_motion:+.3f} "
                    f"drop mean={d.mean():+.3f} "
                    f"median={np.median(d):+.3f} "
                    f"p95={np.percentile(d, 95):+.3f} "
                    f"p99={np.percentile(d, 99):+.3f} "
                    f"max={d.max():+.3f} "
                    f"n_fwd={d.size}"
                )

        strong = (drop > self.strong_thr) & check_fwd
        weak = (drop > self.weak_min) & (drop <= self.strong_thr) & check_fwd
        weak_confirmed = weak & self._spatial_confirm(weak)
        candidates = strong | weak_confirmed

        # N/M подтверждение
        self.confirm_buffer.append(candidates)
        stacked = np.stack(list(self.confirm_buffer), axis=0)
        confirmed = stacked.sum(axis=0) >= self.n_confirm

        # обновление baseline
        self.freeze_counter = np.where(candidates, self.freeze_counter + 1, 0)
        still_frozen = candidates & (
            self.freeze_counter < self.max_freeze_frames
        )
        self.prev_range = np.where(still_frozen, self.prev_range, range_img)

        self.frame_idx += 1
        return confirmed

    def anomaly_info(
        self, mask: np.ndarray
    ) -> list[tuple[float, float, float]]:
        """
        Список (elevation_deg, azimuth_deg, range_m) для аномальных ячеек.
        """
        if self.prev_range is None:
            return []
        ys, xs = np.where(mask)
        out = []
        for y, x in zip(ys, xs):
            r = float(self.prev_range[y, x])
            az_deg = (x / self.azimuth_bins) * 360.0 - 180.0
            el_deg = (y / self.elevation_bins) * 180.0 - 90.0
            out.append((el_deg, az_deg, r))
        return out

    def anomalies_to_xyz(self, mask: np.ndarray) -> np.ndarray:
        """
        Возвращает (M, 3) float32 — координаты x, y, z для подтверждённых
        аномальных ячеек. Дальность берётся из prev_range (актуальная),
        азимут и элевация — из индексов ячейки.
        """
        if self.prev_range is None:
            return np.zeros((0, 3), dtype=np.float32)

        ys, xs = np.where(mask)
        if ys.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        H, W = mask.shape

        az_deg = (xs / W) * 360.0 - 180.0
        el_deg = (ys / H) * 180.0 - 90.0
        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)

        r = self.prev_range[ys, xs].astype(np.float32)

        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        return np.column_stack([x, y, z]).astype(np.float32)
