"""
perspective_fixed.py — 修正版 (2026-11-12, 原版は無変更)

元ファイル:
  - perspective_projection.py        (単純式 f*x/(z+d))
  - efficient_perspective_methods.py (EfficientPerspective / CameraSystem)
  ※ _2 は _1 とバイト同一だったため対象外

修正点 (検証バッテリ perspective_battery_v1 の不合格項目に対応):
  1. z の向きの約束を全モジュールで統一: 「カメラ前方 = +z」。
     原版 EfficientPerspective は行列法 (OpenGL式, 前方=-z) と
     ベクトル化法 (前方=+z) が同一クラス内で符号反転していた。
     本修正では行列を +z 前方用に組み直し、両手法を一致させた。
  2. CameraSystem のビュー変換を修正。
     原版は並進成分が変換に効かず、原点カメラ以外で
     カメラ移動量そのままの誤差 (実測 max 5.0) が出ていた。
     本修正は基底ベクトルとの内積で直接計算する
     (回転の転置ミスが構造的に入り込まない形)。
  3. fail-closed ガード: クリップ範囲外・特異点は黙って歪めず例外にする。
     原版ベクトル化法は z を [near, far] へ clip して黙って歪めていた
     (z=200 で出力2倍化、背後 z=-5 で 17.3)。
     原版単純式は z=-d で inf を返していた。
  4. import してもデモが走らない (原版 perspective_projection は
     import 時に描画+表示が実行された)。

約束 (全関数共通):
  - 入力点はカメラ座標で、前方 +z、near <= z <= far が有効域。
  - 有効域外の点が1つでもあれば ValueError (fail-closed)。
"""
import numpy as np


# ============================================================
# 単純式 (原 perspective_projection.py の修正版)
# ============================================================

def perspective_projection(points_3d, focal_length=5, screen_distance=1):
    """x2d = f*x/(z+d), y2d = f*y/(z+d)。分母 <= 0 は例外。"""
    points_3d = np.asarray(points_3d, dtype=float)
    x_3d, y_3d, z_3d = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    denominator = z_3d + screen_distance
    if np.any(denominator <= 0):
        raise ValueError(
            "perspective_projection: point at/behind the screen plane "
            f"(z + screen_distance <= 0), min denominator = {denominator.min()}"
        )
    x_2d = (focal_length * x_3d) / denominator
    y_2d = (focal_length * y_3d) / denominator
    return np.column_stack([x_2d, y_2d])


def orthographic_projection(points_3d):
    points_3d = np.asarray(points_3d, dtype=float)
    return points_3d[:, :2]


# ============================================================
# 行列式 (原 EfficientPerspective の修正版, +z 前方で統一)
# ============================================================

class EfficientPerspective:
    """透視投影。行列法とベクトル化法は同一約束 (+z 前方) で一致する。"""

    def __init__(self, fov=60, aspect=1.0, near=0.1, far=100.0):
        self.fov = np.radians(fov)
        self.aspect = aspect
        self.near = near
        self.far = far
        self.update_projection_matrix()

    def update_projection_matrix(self):
        """+z 前方用の透視投影行列。
        clip: (f/aspect*x, f*y, a*z + b, z)
          a = (far+near)/(far-near), b = -2*far*near/(far-near)
        z=near -> NDC z=-1, z=far -> NDC z=+1, w=z>0 (前方)。
        """
        f = 1.0 / np.tan(self.fov / 2.0)
        a = (self.far + self.near) / (self.far - self.near)
        b = -2.0 * self.far * self.near / (self.far - self.near)
        self.projection_matrix = np.array([
            [f / self.aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, a, b],
            [0, 0, 1, 0],
        ])

    def _as_points(self, points_3d):
        p = np.asarray(points_3d, dtype=float)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        return p

    def _check_z(self, z):
        if np.any(z < self.near) or np.any(z > self.far):
            raise ValueError(
                "EfficientPerspective: z outside [near, far] "
                f"(near={self.near}, far={self.far}, "
                f"z range = [{z.min()}, {z.max()}]) — fail-closed, no clipping"
            )

    def project_points_matrix(self, points_3d):
        p = self._as_points(points_3d)
        self._check_z(p[:, 2])
        n = p.shape[0]
        homogeneous = np.column_stack([p, np.ones(n)])
        projected = homogeneous @ self.projection_matrix.T
        w = projected[:, 3]
        ndc = projected[:, :3] / w.reshape(-1, 1)
        return ndc[:, :2]

    def project_points_vectorized(self, points_3d):
        p = self._as_points(points_3d)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        self._check_z(z)
        f = 1.0 / np.tan(self.fov / 2.0)
        x_proj = (f / self.aspect) * x / z
        y_proj = f * y / z
        return np.column_stack([x_proj, y_proj])


# ============================================================
# カメラ (原 CameraSystem の修正版)
# ============================================================

class CameraSystem:
    """look-at カメラ。出力カメラ座標は 前方 +z (投影側と同約束)。"""

    def __init__(self, position=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.up = np.array(up, dtype=float)
        self.update_view_matrix()

    def update_view_matrix(self):
        """基底を内積で直接組む (転置ミスの入らない形)。
        cam_x = right . d, cam_y = up2 . d, cam_z = forward . d
        (d = point - position)。 前方が +z になる。
        """
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        up2 = np.cross(right, forward)
        self.forward, self.right, self.up2 = forward, right, up2
        # 参考用に行列表現も保持する (行 = 基底, 第4列 = -基底.位置)
        self.view_matrix = np.array([
            [right[0], right[1], right[2], -np.dot(right, self.position)],
            [up2[0], up2[1], up2[2], -np.dot(up2, self.position)],
            [forward[0], forward[1], forward[2], -np.dot(forward, self.position)],
            [0, 0, 0, 1],
        ])

    def transform_points(self, points_3d):
        p = np.asarray(points_3d, dtype=float)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        d = p - self.position
        return np.column_stack([
            d @ self.right,
            d @ self.up2,
            d @ self.forward,
        ])
