"""
Detector geométrico de skill checks do DBD.

O que existe na imagem real (calibrado com frames capturados):
  - Anel do skill check: raio ~60-70px no frame 224x224
  - Zona (normal+ótima): arco BRANCO (baixa saturação, alto valor)
  - Ponteiro: arco VERMELHO brilhante que rotaciona no sentido horário

Abordagem preditiva:
  1. Detecta o arco branco no anel (r=60-70) → posição da zona
  2. Detecta o ponteiro vermelho (r=35-85) → posição atual
  3. Mede velocidade angular com filtro mediano
  4. Dispara quando: distância_até_zona / velocidade <= pipeline_ms
"""

import cv2
import numpy as np
from collections import deque
from time import time


class GeometricSkillCheckDetector:
    FRAME_SIZE = 224
    CX, CY = 112, 112

    # Anel calibrado com imagens reais: r=60-70px
    _R_INNER = 60
    _R_OUTER = 70

    # Ponteiro vermelho — banda mais larga (ele atravessa o anel em diagonal)
    _N_INNER = 35
    _N_OUTER = 85

    # HSV vermelho: dois intervalos (vermelho cruza 0°/180° no hue)
    _RED_LO1 = np.array([0,   130, 100], dtype=np.uint8)
    _RED_HI1 = np.array([12,  255, 255], dtype=np.uint8)
    _RED_LO2 = np.array([168, 130, 100], dtype=np.uint8)
    _RED_HI2 = np.array([180, 255, 255], dtype=np.uint8)

    # HSV branco (zona normal+ótima): saturação baixa, valor alto
    _WHITE_LO = np.array([0,   0, 155], dtype=np.uint8)
    _WHITE_HI = np.array([180, 65, 255], dtype=np.uint8)

    _N = 1080  # resolução angular (1/3° por amostra)

    _STABILITY_FRAMES = 2  # frames consecutivos antes de confiar na zona

    def __init__(self, pipeline_ms: float = 40.0):
        """
        pipeline_ms: latência total captura→tecla chegar no jogo.
        Observação do usuário: ponteiro percorre a zona inteira durante o pipeline
        (~15° a 200dps = ~75ms). Padrão 40ms para disparar antes de entrar na zona.
        """
        self.pipeline_s = pipeline_ms / 1000.0

        self._angles = np.linspace(0, 360, self._N, endpoint=False, dtype=np.float32)
        rad   = np.radians(self._angles)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        # Coordenadas do anel (detecção de zona branca)
        radii_ring = np.arange(self._R_INNER, self._R_OUTER + 1)
        self._ring_xs = np.clip(
            (self.CX + radii_ring[None, :] * cos_a[:, None]).astype(np.int32), 0, 223)
        self._ring_ys = np.clip(
            (self.CY + radii_ring[None, :] * sin_a[:, None]).astype(np.int32), 0, 223)

        # Coordenadas para ponteiro vermelho (banda maior)
        radii_needle = np.arange(self._N_INNER, self._N_OUTER + 1)
        self._needle_xs = np.clip(
            (self.CX + radii_needle[None, :] * cos_a[:, None]).astype(np.int32), 0, 223)
        self._needle_ys = np.clip(
            (self.CY + radii_needle[None, :] * sin_a[:, None]).astype(np.int32), 0, 223)

        self._vel_hist       = deque(maxlen=8)
        self.vel_dps         = 0.0
        self._prev_angle     = None
        self._prev_time      = None
        self._vel_sample_count = 0

        self._white_zone    = None
        self._stable_count  = 0
        self._no_zone_count = 0
        self._diag_frame    = 0

    def reset(self):
        self._vel_hist.clear()
        self.vel_dps           = 0.0
        self._prev_angle       = None
        self._prev_time        = None
        self._vel_sample_count = 0
        self._white_zone       = None
        self._stable_count     = 0
        self._no_zone_count    = 0

    def process(self, frame_rgb: np.ndarray) -> tuple[bool, dict]:
        self._diag_frame += 1
        now = time()
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        # 1. Detectar zona branca (skill check ativo)
        wz = self._detect_white_zone(hsv)
        if wz is not None:
            self._no_zone_count = 0
            self._stable_count += 1
            self._white_zone = wz
        else:
            self._no_zone_count += 1
            self._stable_count = 0
            if self._no_zone_count > 25:
                self.reset()
                return False, {"white_zone": None, "needle": None, "vel": 0.0}
            wz = self._white_zone

        # 2. Detectar ponteiro SEMPRE (para o gate funcionar mesmo durante warm-up)
        needle_angle = self._detect_red_needle(hsv)

        # 3. Atualizar velocidade angular SEMPRE
        if needle_angle is not None and self._prev_angle is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if 0 < dt < 0.08:
                delta = self._cw(self._prev_angle, needle_angle)
                if 0.05 <= delta <= 12.0:
                    self._vel_hist.append(delta / dt)
                    self.vel_dps = float(np.median(self._vel_hist))
                    self._vel_sample_count += 1

        if needle_angle is not None:
            self._prev_angle = needle_angle
            self._prev_time  = now

        debug = {"white_zone": wz, "needle": needle_angle, "vel": self.vel_dps}

        if wz is None:
            return False, debug

        if self._stable_count < self._STABILITY_FRAMES:
            return False, debug

        if needle_angle is None:
            return False, debug

        if self.vel_dps < 20 or self._vel_sample_count < 3:
            return False, debug

        # 4. Decidir se pressiona
        return self._should_press(needle_angle, wz), debug

    def needle_is_moving(self, min_vel_dps: float = 25.0) -> bool:
        """Gate p/ insta-click: True se ponteiro tem velocidade confiável."""
        return self._vel_sample_count >= 1 and self.vel_dps >= min_vel_dps

    # ------------------------------------------------------------------

    def _detect_white_zone(self, hsv: np.ndarray) -> tuple | None:
        mask    = cv2.inRange(hsv, self._WHITE_LO, self._WHITE_HI)
        vals    = mask[self._ring_ys, self._ring_xs]   # (N, n_radii)
        has_hit = np.any(vals > 0, axis=1)             # (N,) bool

        white_angles = self._angles[has_hit]
        if len(white_angles) < 10:
            return None

        arc = self._arc_range(white_angles)
        if arc is None:
            return None

        start, end = arc
        span = self._cw(start, end)
        if span < 15 or span > 200:
            return None

        return start, end

    def _detect_red_needle(self, hsv: np.ndarray) -> float | None:
        mask1 = cv2.inRange(hsv, self._RED_LO1, self._RED_HI1)
        mask2 = cv2.inRange(hsv, self._RED_LO2, self._RED_HI2)
        mask  = cv2.bitwise_or(mask1, mask2)

        # Média de pixels vermelhos por ângulo
        vals    = mask[self._needle_ys, self._needle_xs].astype(np.float32)
        red_cnt = vals.mean(axis=1)  # (N,)

        # Suavização circular
        k   = 7
        pad = np.concatenate([red_cnt[-k:], red_cnt, red_cnt[:k]])
        smooth = np.convolve(pad, np.ones(k) / k, mode='valid')[:self._N]

        peak_idx = int(np.argmax(smooth))
        # threshold: >5 = ao menos 5% dos pixels naquele ângulo são vermelhos
        if smooth[peak_idx] < 5:
            return None

        return float(self._angles[peak_idx])

    # ------------------------------------------------------------------

    def _should_press(self, needle: float, wz: tuple) -> bool:
        """
        Dispara para que o press chegue no FIM do arco branco (zona ótima/great).
        No DBD, great fica no FINAL do arco (sentido horário do ponteiro).
        Compensa todo o delay do sistema via self.pipeline_s.
        """
        start, end = wz

        # Distância clockwise do ponteiro até o FIM do arco (great)
        dist_to_great = self._cw(needle, end)

        # Já passou da great (dist > 180) ou ainda muito longe (> 90) → não dispara
        if dist_to_great > 90:
            return False

        # Tempo estimado para ponteiro chegar na great
        t = dist_to_great / self.vel_dps
        return t <= self.pipeline_s

    def _cw(self, a: float, b: float) -> float:
        return float((b - a) % 360)

    def _in_arc(self, angle: float, start: float, end: float) -> bool:
        return self._cw(start, angle) <= self._cw(start, end)

    def _arc_range(self, angles: np.ndarray) -> tuple | None:
        s    = np.sort(angles)
        gaps = np.concatenate([np.diff(s), [s[0] + 360 - s[-1]]])
        gi   = int(np.argmax(gaps))

        if gaps[gi] < 1:
            return None

        if gi == len(s) - 1:
            return float(s[0]), float(s[-1])
        else:
            return float(s[gi + 1]), float(s[gi])
