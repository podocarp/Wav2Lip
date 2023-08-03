import numpy as np
import cv2
import os
from tqdm import tqdm
import torch
import subprocess
from Wav2Lip import audio
import Wav2Lip.face_detection as face_detection
from Wav2Lip.models import Wav2Lip

IMG_SIZE = 96
MEL_STEP_SIZE = 16
FFMPEG = "ffmpeg -hide_banner -loglevel error"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} for inference.".format(device))


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i:i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, batch_size, pads=(0, 0, 10, 0), no_smooth=True):
    """pads: (y1, y2, x1, x2)"""
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=device
    )

    predictions = []
    for i in tqdm(range(0, len(images), batch_size)):
        predictions.extend(
            detector.get_detections_for_batch(np.array(images[i:i + batch_size]))
        )

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite(
                "temp/faulty_frame.jpg", image
            )  # check this frame where the face was not detected.
            raise ValueError(
                "Face not detected! Ensure the video contains a face in all the frames."
            )

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not no_smooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [
        [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
        for image, (x1, y1, x2, y2) in zip(images, boxes)
    ]

    del detector
    return results


def datagen(frames, mels, box, is_static, batch_size):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if box is not None and len(box) == 4:
        print("Using the specified bounding box instead of face detection...")
        y1, y2, x1, x2 = box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
    else:
        face_det_results = face_detect([frames[0]], batch_size) \
            if is_static else face_detect(frames, batch_size)
        
    for i, m in enumerate(mels):
        idx = 0 if is_static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, IMG_SIZE // 2 :] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, IMG_SIZE // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        yield img_batch, mel_batch, frame_batch, coords_batch



def load_video(video_path: str, fps=24):
    if not os.path.isfile(video_path):
        raise ValueError("--face argument must be a valid path to video/image file")

    elif video_path.split(".")[1] in ["jpg", "png", "jpeg"]:
        full_frames = [cv2.imread(video_path)]

    else:
        video_stream = cv2.VideoCapture(video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print("Reading video frames...")

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            full_frames.append(frame)

    print("Number of frames available for inference: " + str(len(full_frames)))
    return fps, full_frames


def load_audio(audio_path, fps):
    if not audio_path.endswith(".wav"):
        print("Extracting raw audio...")
        temp_path = "temp/__temp.wav"
        command = f"{FFMPEG} -y -i {audio_path} -strict -2 {temp_path}"
        subprocess.call(command, shell=True)
        audio_path = temp_path

    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
        )

    mel_chunks = []
    mel_idx_multiplier = 80.0 / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE:])
            break
        mel_chunks.append(mel[:, start_idx:start_idx + MEL_STEP_SIZE])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))
    return mel_chunks



class Inference():
    def load_model(self, checkpoint_path):
        self.model = Wav2Lip()
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace("module.", "")] = v
        self.model.load_state_dict(new_s)

        self.model = self.model.to(device)
        print("Model loaded")
        return self.model.eval()

    def __init__(self, checkpoint_path):
        self.load_model(checkpoint_path)

    def generate(
        self,
        video_path: str,
        audio_path: str,
        out_path: str,
        batch_size=16,
        no_smooth=False,
        is_static=False,
        box=[],
        fps=24,
    ):
        fps, full_frames = load_video(video_path, fps)
        mel_chunks = load_audio(audio_path, fps)
        full_frames = full_frames[:len(mel_chunks)]
        gen = datagen(full_frames.copy(), mel_chunks, box, is_static, batch_size)

        result_path = "temp/__temp.avi"
        frame_h, frame_w = full_frames[0].shape[:-1]
        out = cv2.VideoWriter(
            result_path,
            cv2.VideoWriter_fourcc(*"DIVX"),
            fps,
            (frame_w, frame_h),
        )

        for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))
        ):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()

        command = f"{FFMPEG} -y -i {audio_path} -i {result_path} -strict -2 -q:v 1 {out_path}"
        subprocess.call(command, shell=True)
