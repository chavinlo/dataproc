from imutils.object_detection import non_max_suppression
import numpy as np
import cv2

class EastTextDetector():
    def __init__(self, model_path: str) -> None:
        """
        EAST Text Detection Model Wrapper
        """
        self.model = cv2.dnn.readNet(model_path)
        self.layerNames = [
    	"feature_fusion/Conv_7/Sigmoid",
    	"feature_fusion/concat_3"]

    def run(
            self,
            image: np.array,
            treshold: float = 0.5,
            auto_resize: bool = True,
        ):
        """
        Function to run inference

        image (np.array): Image to analyze
        treshold (float): Confidence treshold, default to 0.5
        auto_resize (bool): Automatically resize to multiple of 32, default to True
        """
        layerNames = self.layerNames
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        (H, W) = image.shape[:2]

        if H % 32 != 0 or W % 32 != 0:
            if auto_resize:
                newH = (round(H / 32) * 32)
                newW = (round(W / 32) * 32)
                image = cv2.resize(image, (newW, newH))
            else:
                raise Exception("Dimensions not multiple of 32")
        
        (H, W) = image.shape[:2]
        
        net = self.model
        
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        
        net.setInput(blob)
        
        (scores, geometry) = net.forward(layerNames)
        
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
        
            for x in range(0, numCols):
                if scoresData[x] < treshold:
                    continue # if lower than treshold ignore

                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
                            
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        return boxes
    
    def calc_coverage(
        self,
        boxes: np.ndarray,
        im_w: int,
        im_h: int
        ):
        """
        Calculate the % of the boxes area and the total image area

        boxes (np.ndarray): List of arrays returned by `run` function
        im_w (int): Width of the image
        im_h (int): Height of the image
        """
        total_boxes_area = 0
        for box in boxes:
            startX, startY, endX, endY = box
            box_width = endX - startX
            box_height = endY - startY
            total_boxes_area += box_width * box_height

        total_image_area = im_w * im_h

        coverage_percentage = (total_boxes_area / total_image_area) * 100
        return coverage_percentage