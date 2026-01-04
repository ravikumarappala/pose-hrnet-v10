from __future__ import absolute_import, division, print_function

import time
import cv2
import numpy as np
from PIL import Image  # (not strictly needed but kept from original)
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision
import os

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect

import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

# ------------- Your original constants -------------

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
              [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255],
              [0, 170, 255], [0, 85, 255], [0, 0, 255],
              [85, 0, 255], [170, 0, 255], [255, 0, 255],
              [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ------------- Drawing helpers -------------

def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: shape [17, 2]
    :params img: BGR image
    """
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def draw_bbox(box, img):
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)

# ------------- Detection & pose helpers -------------

def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []

    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model_input = transform(model_input).unsqueeze(0)
    pose_model.eval()
    with torch.no_grad():
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio

    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

# ------------- Model loading (once) -------------

box_model = None
pose_model = None

def init_models():
    global box_model, pose_model

    if box_model is not None and pose_model is not None:
        return  # already initialized

    # CUDNN config
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Mimic your original parse_args defaults
    class DummyArgs:
        def __init__(self):
            self.cfg = 'inference-config.yaml'
            self.opts = []
            self.modelDir = ''
            self.logDir = ''
            self.dataDir = ''
            self.prevModelDir = ''

    args = DummyArgs()
    update_config(cfg, args)

    # 1) Person detector
    print("=> Loading Faster R-CNN person detector...")
    box_model_local = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model_local.to(CTX)
    box_model_local.eval()

    # 2) Pose model
    print("=> Loading pose model...")
    pose_model_local = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        # Build path relative to repo root (parent of demo/)
        base_dir = os.path.dirname(os.path.abspath(__file__))   # .../HRnet_0.1/demo
        repo_root = os.path.dirname(base_dir)                   # .../HRnet_0.1

        # cfg.TEST.MODEL_FILE is currently "models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth"
        model_rel = cfg.TEST.MODEL_FILE.lstrip("/")             # avoid accidental leading '/'
        model_path = os.path.join(repo_root, model_rel)         # .../HRnet_0.1/models/pytorch/...

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pose model file not found at: {model_path}")

        print('=> loading model from {}'.format(model_path))
        pose_model_local.load_state_dict(torch.load(model_path, map_location=CTX), strict=False)
    else:
        print('Expected pose model path in cfg.TEST.MODEL_FILE')


    pose_model_local = torch.nn.DataParallel(pose_model_local, device_ids=cfg.GPUS)
    pose_model_local.to(CTX)
    pose_model_local.eval()

    box_model = box_model_local
    pose_model = pose_model_local
    print("=> Models loaded and ready.")

# ------------- Core processing function -------------

def process_frame(frame_bgr: np.ndarray, show_fps: bool = False) -> np.ndarray:
    """
    Take a single BGR frame, run:
      - person detection
      - pose estimation
      - draw skeleton
    and return the processed BGR frame.
    """
    global box_model, pose_model
    assert box_model is not None and pose_model is not None, "Models not initialized; call init_models() first."

    image_bgr = frame_bgr.copy()
    last_time = time.time()

    # For pose function that uses RGB/BGR selection based on cfg
    image = image_bgr[:, :, [2, 1, 0]]  # BGR -> RGB

    # Prepare input for person detector (Faster R-CNN)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb / 255.).permute(2, 0, 1).float().to(CTX)
    input_list = [img_tensor]

    # Person detection
    pred_boxes = get_person_detection_boxes(box_model, input_list, threshold=0.9)

    # Pose estimation per person
    if len(pred_boxes) >= 1:
        for box in pred_boxes:
            center, scale = box_to_center_scale(
                box,
                cfg.MODEL.IMAGE_SIZE[0],
                cfg.MODEL.IMAGE_SIZE[1]
            )
            image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
            pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
            if len(pose_preds) >= 1:
                for kpt in pose_preds:
                    draw_pose(kpt, image_bgr)

    if show_fps:
        fps = 1 / (time.time() - last_time)
        cv2.putText(
            image_bgr,
            'fps: ' + "%.2f" % (fps),
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

    return image_bgr

# ------------- FastAPI app -------------

app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_models()

# HTTP endpoint: JPEG in → JPEG out
@app.post("/process-frame")
async def process_frame_api(request: Request):
    body = await request.body()
    if not body:
        return Response(status_code=400, content=b"No image data")

    nparr = np.frombuffer(body, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return Response(status_code=400, content=b"Invalid image data")

    processed_bgr = process_frame(frame_bgr, show_fps=False)

    ok, jpeg = cv2.imencode(".jpg", processed_bgr)
    if not ok:
        return Response(status_code=500, content=b"Failed to encode image")

    return Response(content=jpeg.tobytes(), media_type="image/jpeg")

# WebSocket: binary JPEG frames in → binary JPEG frames out
@app.websocket("/ws-stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()

            nparr = np.frombuffer(data, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                # could send an error message, but just skip
                continue

            processed_bgr = process_frame(frame_bgr, show_fps=False)

            ok, jpeg = cv2.imencode(".jpg", processed_bgr)
            if not ok:
                continue

            await ws.send_bytes(jpeg.tobytes())

    except WebSocketDisconnect:
        print("WebSocket client disconnected")

# If you want to run directly: python api_server.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

