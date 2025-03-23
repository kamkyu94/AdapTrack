import os
import cv2
import torch
import random
import pickle
import numpy as np
import torch.backends.cudnn as cudnn
import logging

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess

# Configure logging for detect
detect_logger = logging.getLogger('detect')

def detect(
    exp_file: str,
    ckpt_file: str,
    output_file: str,
    frame_paths: list,
    test_conf: float = 0.5,
    nmsthre: float = 0.45,
    test_size: tuple = (896, 1600),
    fuse: bool = True,
    fp16: bool = True,
    seed: int = None,
    local_rank: int = 0,
    device: str = None
):
    """
    Perform object detection on a list of frames using a YOLOX model.

    Args:
        exp_file (str): Path to the experiment configuration file.
        ckpt_file (str): Path to the model checkpoint file.
        output_file (str): Path to save the detection results.
        frame_paths (list): List of paths to input image frames.
        test_conf (float): Confidence threshold for detections (default: 0.5).
        nmsthre (float): Non-maximum suppression threshold (default: 0.45).
        test_size (tuple): Target size for input images (height, width) (default: (896, 1600)).
        fuse (bool): Whether to fuse model layers for optimization (default: True).
        fp16 (bool): Whether to use half-precision (FP16) inference (default: True).
        seed (int): Random seed for reproducibility (default: None).
        local_rank (int): Local rank for multi-GPU setup (default: 0).
        device (str): Device to run inference on (default: None, auto-selects 'cuda' or 'cpu').

    Returns:
        list: List of NumPy arrays, where each array contains detections for a frame in the format
              [x1, y1, x2, y2, score, class_id].
    """
    # Set device (CUDA if available, else CPU)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    detect_logger.info(f"Using device: {device}")

    # Log input details
    detect_logger.info(f"Checkpoint exists: {os.path.exists(ckpt_file)}")
    detect_logger.info(f"Checkpoint size (bytes): {os.path.getsize(ckpt_file)}")
    detect_logger.info(f"Exp file exists: {os.path.exists(exp_file)}")
    detect_logger.info(f"Number of frames: {len(frame_paths)}")

    # Seed setup for reproducibility
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.benchmark = True

    # Load experiment configuration and set parameters
    exp = get_exp(exp_file)
    exp.test_conf = test_conf
    exp.nmsthre = nmsthre
    exp.test_size = test_size
    exp.num_classes = 1  # Assuming single-class detection (e.g., pedestrians)

    # Load and prepare the model
    model = exp.get_model().cuda(local_rank).eval()
    torch.cuda.set_device(local_rank)
    ckpt = torch.load(ckpt_file, map_location=f"cuda:{local_rank}", weights_only=True)
    model.load_state_dict(ckpt["model"])
    detect_logger.info("YOLOX model loaded successfully")

    # Apply model optimizations
    if fuse:
        model = fuse_model(model)
    if fp16:
        model = model.half()

    # Define normalization parameters (standard ImageNet values)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Initialize list to store detections
    detections_list = []

    # Process each frame
    for frame_idx, frame_path in enumerate(frame_paths):
        # Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
            detect_logger.warning(f"Failed to load frame: {frame_path}")
            detections_list.append(np.array([]))  # Empty array for failed frames
            continue

        # Log first frame size
        if frame_idx == 0:
            print(f"First frame size: {frame.shape[1]}x{frame.shape[0]} (width x height)")

        # Preprocessing: Convert to RGB, resize, pad, and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame_rgb.shape[:2]
        scale = min(test_size[0] / float(img_h), test_size[1] / float(img_w))
        new_h, new_w = int(img_h * scale), int(img_w * scale)
        img = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to test_size with gray value (114)
        padded_img = np.ones((test_size[0], test_size[1], 3), dtype=np.float32) * 114.0
        padded_img[:new_h, :new_w, :] = img

        # Normalize with mean and std
        padded_img = padded_img / 255.0
        padded_img -= mean
        padded_img /= std

        # Convert to CHW format and create tensor
        padded_img = padded_img.transpose(2, 0, 1)  # HWC to CHW
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        img = torch.from_numpy(padded_img).unsqueeze(0).to(device)
        if fp16:
            img = img.half()
        else:
            img = img.float()

        # Perform model inference
        with torch.no_grad():
            outputs = model(img)
            predictions = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)[0]

        # Handle case with no detections
        if predictions is None or len(predictions) == 0:
            detect_logger.info(f"Frame {frame_idx + 1}: No detections")
            detections_list.append(np.array([]))
            continue

        # Process detections: Combine confidence scores and filter columns
        det = predictions
        det[:, 4] *= det[:, 5]  # Multiply obj_conf by class_conf
        det[:, 5] = det[:, 6]   # Assign class_pred to class_id column
        det = det[:, :6]        # Keep [x1, y1, x2, y2, score, class_id]

        # Scale detections back to original image coordinates
        det = det.cpu().numpy()
        det[:, :4] /= scale

        # Filter out invalid boxes (negative or zero-area boxes)
        det = det[(np.minimum(det[:, 2], img_w - 1) - np.maximum(det[:, 0], 0)) > 0]
        det = det[(np.minimum(det[:, 3], img_h - 1) - np.maximum(det[:, 1], 0)) > 0]

        # Filter for specific class (e.g., class_id == 0 for pedestrians)
        mask = det[:, 5] == 0
        det = det[mask]

        # Store detections (empty array if no valid detections)
        detections_list.append(det if len(det) > 0 else np.array([]))
        detect_logger.info(f"Frame {frame_idx + 1}: Detected {len(det)} objects")

    # Save detections to file
    detections_dict = {i + 1: det for i, det in enumerate(detections_list)}
    with open(output_file, 'wb') as f:
        pickle.dump(detections_list, f, protocol=pickle.HIGHEST_PROTOCOL) 
    detect_logger.info(f"Detections saved to {output_file}")

    return detections_list