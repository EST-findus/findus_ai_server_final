"""2단계: 정렬.

- ffhq_align: SAM/StyleGAN 입력용. NVlabs/ffhq-dataset 의 recreate_aligned_images 를 5점 기준으로 옮긴 것.
  68점에서 뽑은 5점(lm68_to_lm5)을 넣으면 FFHQ 원본과 같은 결과가 나온다.
- arcface_align: 얼굴 인식 평가용 112x112.
"""
import cv2
import numpy as np
import PIL.Image
import scipy.ndimage

ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# FFHQ 정렬은 얼굴이 사진 밖으로 나가면 반사 패딩 + 블러로 채워서 1024x1024로 만든다.
def ffhq_align(img_rgb: np.ndarray, lm5: np.ndarray, output_size=1024, transform_size=4096,
               enable_padding=True) -> np.ndarray:
    eye_left, eye_right, _, mouth_left, mouth_right = np.asarray(lm5, dtype=np.float64)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # 기울기·크기를 반영한 크롭 사각형(quad)
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = PIL.Image.fromarray(img_rgb)

    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(img.size[0] / shrink)), int(np.rint(img.size[1] / shrink)))
        img = img.resize(rsize, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # 얼굴이 사진 밖으로 나가면 반사 패딩 + 블러로 채운다
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        arr = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        h, w, _ = arr.shape
        yy, xx, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(xx) / pad[0], np.float32(w - 1 - xx) / pad[2]),
                          1.0 - np.minimum(np.float32(yy) / pad[1], np.float32(h - 1 - yy) / pad[3]))
        blur = qsize * 0.02
        arr += (scipy.ndimage.gaussian_filter(arr, [blur, blur, 0]) - arr) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        arr += (np.median(arr, axis=(0, 1)) - arr) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(arr), 0, 255)), "RGB")
        quad += pad[:2]

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                        PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)
    return np.asarray(img)

# ArcFace 정렬은 얼굴이 사진 밖으로 나가면 검은색으로 채워서 112x112로 만든다.
def arcface_align(img_rgb: np.ndarray, lm5: np.ndarray, size=112) -> np.ndarray:
    template = ARCFACE_TEMPLATE * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(np.asarray(lm5, np.float32), template, method=cv2.LMEDS)
    return cv2.warpAffine(img_rgb, M, (size, size), borderValue=0)
