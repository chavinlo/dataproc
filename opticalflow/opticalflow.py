# Sources:
# https://stackoverflow.com/questions/76165060/how-to-find-a-threshold-number-in-optical-flow-in-python-opencv
# https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
# https://static1.squarespace.com/static/6213c340453c3f502425776e/t/655ce779b9d47d342a93c890/1700587395994/stable_video_diffusion.pdf (3.1 Data processing & annotation)

import cv2
import numpy as np

def getOpticalMagnitude(
        video_path: str,
        target_fps: int = None,
        resize: tuple = None
    ):
    """
    Function to get optical magnitude as a float.

    video_path (str): Path to the video file
    target_fps (int): process at given FPS, if None then keep at original. Default is None
    resize (tuple): resize video to given size, if None then keep at original. Default is None
    
    returns a dict that contains the mean and median of the magnitude, a flattened tensor containing the magnitude of each frame, and raw magnitude and time arrays (for plotting).
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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if resize is not None:
                    frame = cv2.resize(frame, resize)
                frames.append(frame)
                skipped_frames = 0
        else:
            break
        
    prev_frame = frames[0]
    mags = list()

    for idx in range(1, len(frames)):
        prev_frame = frames[idx - 1]
        curr_frame = frames[idx]

        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_mean = cv2.mean(mag)
        mags.append(mag_mean)

    mags_arr = np.array(mags)
    time_arr = np.arange(len(mags)) / target_fps

    flattened = mags_arr.flatten()
    flattened = flattened[flattened != 0]
    
    return {
        "mean": np.mean(flattened),
        "median": np.median(flattened),
        "flat": flattened,
        "mags": mags_arr,
        "time": time_arr,
    }