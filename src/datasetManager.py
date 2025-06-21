import cv2

class DatasetManager:
    def __init__(self, video_path):
        self.all_videos = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Impossibile aprire il video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None  # Fine del video
        return frame

    def has_next(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()