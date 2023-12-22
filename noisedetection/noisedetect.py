import cv2
from matplotlib import pyplot as plt
import numpy as np

def calcNoisePerc(
        video_path: str,
        target_fps: int = None,
        resize: tuple = None
    ):
    """
    Calculate the noise amount by using the saturation channel from HSV space

    If the percentage exceeds a threshold, let's say 0.5, then at least fifty percent of all pixels have a saturation of at least 0.05, so this frame seems to be a valid frame.

    Source: https://stackoverflow.com/questions/58924276/detecting-noise-frames
    """

    vid = cv2.VideoCapture(video_path)
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    
    if target_fps is None:
        target_fps = vid_fps

    frames_to_skip = round((vid_fps - target_fps) / target_fps)
    assert target_fps <= vid_fps
    skipped_frames = 0

    frames = list()
    while True:
        ret = vid.grab()
        if ret is True:
            if skipped_frames != frames_to_skip:
                skipped_frames += 1
            else:
                assert skipped_frames == frames_to_skip
                _, frame = vid.retrieve()
                # Convert image to HSV color space
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                if resize is not None:
                    frame = cv2.resize(frame, resize)
                frames.append(frame)
                skipped_frames = 0
        else:
            break

    out_frames = list()
    frame_id = 0
    g_perc = 0

    for frame in frames:
        # Calculate histogram of saturation channel
        s = cv2.calcHist([frame], [1], None, [256], [0, 256])
        
        # Calculate percentage of pixels with saturation >= p
        p = 0.05
        s_perc = np.sum(s[int(p * 255):-1]) / np.prod(frame.shape[0:2])

        out_frames.append({
            "idx": frame_id,
            "s_perc": s_perc
        })

        g_perc = g_perc + s_perc
        frame_id = frame_id + 1

    g_perc = g_perc / len(out_frames)

    return g_perc, out_frames