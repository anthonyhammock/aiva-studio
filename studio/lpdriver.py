"""Drives LivePortrait (MIT, github.com/KwaiVGI/LivePortrait) frame by frame on
the CPU from explicit motion parameters: head pitch/yaw/roll, eye openness, mouth
openness and expression offsets. No driving video is involved.

LivePortrait finds faces with InsightFace, whose models are for non-commercial
research only. LivePortrait's licence asks commercial users to replace it, so this
driver finds the face with OpenCV's bundled Haar cascade and then refines it with
LivePortrait's own MIT-licensed 203-point landmark model.
"""
import os
import pickle
import sys

import cv2
import numpy as np
import torch

LP_DIR = os.environ.get("LP_DIR", "LivePortrait")
sys.path.insert(0, LP_DIR)

from src.config.inference_config import InferenceConfig  # noqa: E402
from src.live_portrait_wrapper import LivePortraitWrapper  # noqa: E402
from src.utils.camera import get_rotation_matrix  # noqa: E402
from src.utils.crop import crop_image, paste_back, prepare_paste_back  # noqa: E402
from src.utils.human_landmark_runner import LandmarkRunner  # noqa: E402
from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio  # noqa: E402

torch.set_grad_enabled(False)


# ------------------------------------------------------------------ templates

class _NumpyOnly(pickle.Unpickler):
    """LivePortrait's motion templates are pickles of numpy arrays. Load them
    without letting the file name any other Python object."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"),
               ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"template refers to {module}.{name}; refusing to load it")


def load_template(name):
    """Expression track of a bundled motion template (wink, laugh, talking ...),
    as offsets from its first frame: (frames, 21, 3) plus its fps."""
    path = os.path.join(LP_DIR, "assets", "examples", "driving", f"{name}.pkl")
    with open(path, "rb") as f:
        d = _NumpyOnly(f).load()
    exp = np.stack([np.asarray(m["exp"], dtype=np.float32).reshape(21, 3) for m in d["motion"]])
    return exp - exp[0:1], float(d.get("output_fps", 25))


# ------------------------------------------------------------------ face

def find_face(img_rgb, landmarks):
    """203 facial landmarks in image coordinates."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6, minSize=(gray.shape[1] // 8,) * 2)
    if len(faces) == 0:
        raise RuntimeError("no face found in the portrait")
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # five points from typical face proportions inside a Haar box, then refine
    pt5 = np.array([[x + 0.30 * w, y + 0.40 * h], [x + 0.70 * w, y + 0.40 * h], [x + 0.50 * w, y + 0.62 * h],
                    [x + 0.36 * w, y + 0.80 * h], [x + 0.64 * w, y + 0.80 * h]], dtype=np.float32)
    lmk = landmarks.run(img_rgb, pt5)
    for _ in range(2):
        lmk = landmarks.run(img_rgb, lmk)
    return lmk


# ------------------------------------------------------------------ driver

class Driver:
    def __init__(self, portrait_rgb):
        cfg = InferenceConfig(flag_force_cpu=True, flag_use_half_precision=False, flag_do_torch_compile=False)
        self.lp = LivePortraitWrapper(inference_cfg=cfg)
        self.cfg = cfg
        landmarks = LandmarkRunner(ckpt_path=os.path.join(LP_DIR, "pretrained_weights/liveportrait/landmark.onnx"),
                                   onnx_provider="cpu")
        self.img = portrait_rgb
        self.lmk = find_face(portrait_rgb, landmarks)
        crop = crop_image(portrait_rgb, self.lmk, dsize=512, scale=2.3, vx_ratio=0, vy_ratio=-0.125, flag_do_rot=True)
        self.M_c2o = crop["M_c2o"]
        self.mask = prepare_paste_back(cfg.mask_crop, self.M_c2o, dsize=(portrait_rgb.shape[1], portrait_rgb.shape[0]))
        img256 = cv2.resize(crop["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)

        I_s = self.lp.prepare_source(img256)
        self.info = self.lp.get_kp_info(I_s)
        self.f_s = self.lp.extract_feature_3d(I_s)
        self.x_s = self.lp.transform_keypoint(self.info)
        self.eyes_s = calc_eye_close_ratio(self.lmk[None])                    # (1, 2)
        self.lip_s = calc_lip_close_ratio(self.lmk[None])                     # (1, 1)
        self._eye0 = self._eye_delta(float(self.eyes_s.mean()))
        # A portrait with parted lips or a toothy smile gets its mouth closed for
        # the resting pose (LivePortrait's own lip normalisation), so talking
        # clips open from closed lips. Closed-lip portraits are left as they are.
        self.close_lips = float(self.lip_s[0, 0]) > cfg.lip_normalize_threshold
        self._lip_rest = 0.0 if self.close_lips else float(self.lip_s[0, 0])
        self._lip0 = None if self.close_lips else self._lip_delta(self._lip_rest)
        self._lip_closed = self._lip_delta(0.0) if self.close_lips else None

    # where the face sits in the portrait: centre and size, as fractions
    def face_box(self):
        x0, y0 = self.lmk.min(0)
        x1, y1 = self.lmk.max(0)
        h, w = self.img.shape[:2]
        return dict(cx=float((x0 + x1) / 2 / w), cy=float((y0 + y1) / 2 / h), w=float((x1 - x0) / w), h=float((y1 - y0) / h))

    def _eye_delta(self, ratio):
        t = torch.from_numpy(np.concatenate([self.eyes_s, [[ratio]]], axis=1)).float()
        return self.lp.retarget_eye(self.x_s, t)

    def _lip_delta(self, ratio):
        t = torch.from_numpy(np.concatenate([self.lip_s, [[ratio]]], axis=1)).float()
        return self.lp.retarget_lip(self.x_s, t)

    def render(self, pitch=0.0, yaw=0.0, roll=0.0, eyes_closed=0.0, mouth_open=0.0, exp=None):
        """One full-size frame. Angles in degrees, relative to the portrait;
        eyes_closed and mouth_open from 0 to 1; exp a (21, 3) expression offset."""
        info = self.info
        R = get_rotation_matrix(info["pitch"] + pitch, info["yaw"] + yaw, info["roll"] + roll)
        delta = info["exp"].clone()
        if exp is not None:
            delta = delta + torch.from_numpy(np.asarray(exp, dtype=np.float32))[None]
        t = info["t"].clone()
        t[..., 2] = 0
        x_d = info["scale"] * (info["kp"] @ R + delta) + t
        if eyes_closed:
            x_d = x_d + (self._eye_delta(float(self.eyes_s.mean()) * (1 - eyes_closed)) - self._eye0)
        if mouth_open:
            d = self._lip_delta(self._lip_rest + 0.32 * mouth_open)
            x_d = x_d + (d if self.close_lips else d - self._lip0)
        elif self.close_lips:
            x_d = x_d + self._lip_closed
        x_d = self.lp.stitching(self.x_s, x_d)
        out = self.lp.warp_decode(self.f_s, self.x_s, x_d)
        face = self.lp.parse_output(out["out"])[0]
        return paste_back(face, self.M_c2o, self.img, self.mask)
