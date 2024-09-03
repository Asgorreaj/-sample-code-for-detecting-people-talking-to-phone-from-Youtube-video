import cv2
import time
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(20)

class Detector:
    def __init__(self, output_dir='./output'):
        self.cell_phone_index = None  # Placeholder for the index of 'cell phone'
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.frame_counter = 0  # Counter for naming saved images

    def readClasses(self, classesFilePath):
        with open(classesFilePath, "r") as f:
            self.classesList = f.read().splitlines()
        # Find index of 'cell phone'
        if 'cell phone' in self.classesList:
            self.cell_phone_index = self.classesList.index('cell phone')
        else:
            raise ValueError("Class 'cell phone' not found in the classes file.")
        # colors list
        self.colorList = np.random.uniform(
            low=0, high=255, size=(len(self.classesList), 3)
        )

    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[: fileName.index(".")]
        self.cacheDir = os.path.join(os.path.dirname(__file__), "../pretrained_models")

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(
            fname=fileName,
            origin=modelURL,
            cache_dir=self.cacheDir,
            cache_subdir="checkpoints",
            extract=True,
        )

    def loadModel(self):
        print("Loading Model " + self.modelName)

        tf.compat.v1.reset_default_graph()
        self.model = tf.saved_model.load(
            os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model")
        )

        print("Model " + self.modelName + " loaded successfully...")

    def createBoundingBox(self, image, threshold=0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)
        bboxs = detections["detection_boxes"][0].numpy()
        classIndexes = detections["detection_classes"][0].numpy().astype(np.int32)
        classScores = detections["detection_scores"][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(
            bboxs,
            classScores,
            max_output_size=50,
            iou_threshold=threshold,
            score_threshold=threshold,
        )

        bboxIdx = bboxIdx.numpy()  # Convert to a numpy array

        saved_frame = False

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                if classIndexes[i] == self.cell_phone_index:  # Check if the class is 'cell phone'
                    bbox = tuple(bboxs[i].tolist())
                    classConfidence = round(100 * classScores[i])
                    classIndex = classIndexes[i]

                    classLabelText = self.classesList[classIndex].upper()
                    classColor = self.colorList[classIndex]

                    displayText = "{}: {}%".format(classLabelText, classConfidence)

                    ymin, xmin, ymax, xmax = bbox
                    xmin, xmax, ymin, ymax = (
                        xmin * imW,
                        xmax * imW,
                        ymin * imH,
                        ymax * imH,
                    )
                    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                    cv2.rectangle(
                        image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1
                    )
                    cv2.putText(
                        image,
                        displayText,
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        classColor,
                        2,
                    )

                    # Save the frame with detected cell phone
                    self.frame_counter += 1
                    frame_filename = os.path.join(self.output_dir, f"frame_{self.frame_counter:04d}.jpg")
                    cv2.imwrite(frame_filename, image)
                    saved_frame = True

        return image, saved_frame

    def predictVideo(self, videoPath, threshold=0.5):
        cap = cv2.VideoCapture(videoPath)

        if not cap.isOpened():
            print("Error opening file...")
            return

        success, image = cap.read()
        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            bboxImage, saved_frame = self.createBoundingBox(image, threshold)
            cv2.putText(
                bboxImage,
                "FPS: " + str(int(fps)),
                (20, 70),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Result", bboxImage)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            success, image = cap.read()
        cap.release()
        cv2.destroyAllWindows()
