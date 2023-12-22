from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from PIL import Image
import cv2
import cv2
import numpy as np
import groundingdino.datasets.transforms as T

class GroundingDinoVideo():
    def __init__(self, model: GroundingDINO) -> None:
        self.model = model
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def run(
            self,
            video_path: str,
            target_fps: int = 2,
            text_prompt: str = "Person",
            box_treshold: float = 0.4,
            text_treshold: float = 0.25
    ):
        # Open the video
        print("Video Opened")
        vid = cv2.VideoCapture(video_path)

        # Calculate the amount of frames to drop
        vid_fps = vid.get(cv2.CAP_PROP_FPS)
        frames_to_skip = round((vid_fps - target_fps) / target_fps)
        assert target_fps <= vid_fps, print("TARGET:", target_fps, "VIDFPS:", vid_fps)
        skipped_frames = 0

        # Drop the frames and break on end of video
        frames = list()
        while True:
            ret = vid.grab()
            if ret is True:
                if skipped_frames != frames_to_skip:
                    skipped_frames += 1
                else:
                    assert skipped_frames == frames_to_skip
                    _, frame = vid.retrieve()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    skipped_frames = 0
            else:
                break
        print("Frames Dropped")

        # Transform the frames
        trans_frames = list()
        for frame in frames:
            framed_trans, _ = self.transform(
                Image.fromarray(frame).convert("RGB"),
                None
            )
            trans_frames.append({
                "trans": framed_trans,
                "og": frame
            })
        print("Frames Transformed")

        pred_frames = list()

        for frame_dict in trans_frames:
            boxes, logits, phrases = predict(
                model=self.model,
                image=frame_dict['trans'],
                caption=text_prompt,
                box_threshold=box_treshold,
                text_threshold=text_treshold
            )

            pred_frames.append(
                {
                    "trans": frame_dict['trans'],
                    "og": frame_dict['og'],
                    "boxes": boxes,
                    "logits": logits,
                    "phrases": phrases
                }
            )
        print("Predicted Frames")

        total_frames = len(trans_frames)

        # Split the string by ' .' and strip whitespace from each element
        classes = [element.strip() for element in text_prompt.split(' .') if element.strip()] # test later

        stats = dict()

        for obj in classes:
            stats[obj] = {
                "average_amount": 0, # Average amount of X in the video
                "precense": 0 # % of the video where class is present
            }

        # Total human boxes across all frames / total frames
        for entry in pred_frames:
            detected_once = list()
            for obj in entry['phrases']:
                if obj not in detected_once:
                    stats[obj]["precense"] = stats[obj]["precense"] + 1
                    detected_once.append(obj)
                stats[obj]["average_amount"] = stats[obj]["average_amount"] + 1

        for obj in classes:
            stats[obj]['average_amount'] = stats[obj]['average_amount'] / total_frames
            stats[obj]["precense"] = stats[obj]["precense"] / total_frames

        return stats

        # Uncomment if doing video annotation
        # annotated_frames = list()
        # for entry in pred_frames:
        #     annotated_frames.append(
        #         annotate(
        #             image_source=entry['og'],
        #             boxes = entry['boxes'],
        #             logits = entry['logits'],
        #             phrases = entry['phrases']
        #         )
        #     )
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter('/home/ubuntu/practice/second/output.mp4', fourcc, 8, (640, 480))

        # for frame in annotated_frames:
        #     resized_frame = cv2.resize(frame, (640, 480))
        #     x = out.write(resized_frame)
        #     print(x)

        # out.release()