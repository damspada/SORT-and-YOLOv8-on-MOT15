import torch
import cv2
import os
import random
import time
from typing import List

class Visualizer:
    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
      """
      Trasform a matrix (N,4) [cx,cy,w,h] in a matrix (N,4) [x1,y1,x2,y2]
      """
      XY_1 = boxes[:, :2] - (boxes[:, 2:] / 2)
      XY_2 = boxes[:, :2] + (boxes[:, 2:] / 2)
      return torch.cat((XY_1, XY_2), dim=1) 

    def __init__(self, save_video: bool = False, output_path: str = "output.mp4", frame_size=None, fps=30):
        """
        save_video: salva il video nel path indicato
        output_path: dove salvare il file mp4 se save_video=True
        frame_size: (width, height). Se None, verrà determinato dal primo frame
        fps: frame rate desiderato per la visualizzazione e salvataggio
        """
        self.save_video = save_video
        self.output_path = output_path
        self.frame_size = frame_size  # può essere None, verrà impostato al primo frame
        self.fps = fps
        self.frame_interval = 1.0 / fps  # in secondi
        self.writer = None
        self.colors = {}
        self.window_name = "SORT Tracker"
        self._last_time = time.time()

    def _get_color(self, track_id: int):
        if track_id not in self.colors:
            self.colors[track_id] = (
                random.randint(64, 255),
                random.randint(64, 255),
                random.randint(64, 255)
            )
        return self.colors[track_id]

    def draw(self, frame, tracks: List):
        """
        frame: immagine numpy (BGR)
        tracks: lista di oggetti con attributi .id (int) e .box (Tensor [x1,y1,x2,y2])
        """
        # Disegna i box
        for track in tracks:
            box = track.X_hat[0,:4].int().tolist()  # Assumi che box sia torch.Tensor (4,) in formato xyxy
            track_id = track.id
            color = self._get_color(track_id)

            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Salva su file video
        if self.save_video:
            if self.writer is None:
                if self.frame_size is None:
                    h, w = frame.shape[:2]
                    self.frame_size = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
            self.writer.write(frame)

        # Mostra a schermo
        cv2.imshow(self.window_name, frame)

        # Timing per simulare il frame rate corretto
        now = time.time()
        elapsed = now - self._last_time
        delay = self.frame_interval - elapsed
        if delay > 0:
            key = cv2.waitKey(int(delay * 1000))
        else:
            key = cv2.waitKey(1)
        self._last_time = time.time()

        if key == ord('q'):
            exit(0)

    def close(self):
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
