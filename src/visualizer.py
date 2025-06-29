import cv2
import numpy as np
import torch
import time
import threading
from queue import Queue
from typing import Optional, Tuple
import colorsys

class Visualizer:
    def __init__(self, save_video: bool = False, output_path: Optional[str] = None, dt: float = 0.033):
        """
        Inizializza il visualizer per SORT
        
        Args:
            save_video (bool): Se True, salva il video su file
            output_path (str): Percorso del file di output (obbligatorio se save_video=True)
            dt (float): Tempo di visualizzazione per frame in secondi
        """
        # Inizializza tutti gli attributi prima di qualsiasi validazione
        self.save_video = save_video
        self.dt = dt
        self.output_path = output_path
        self.video_writer = None
        self.frame_size = None
        self.frame_buffer = Queue(maxsize=2)
        self.last_display_time = 0
        self.display_thread = None
        self.should_stop = False
        self.colors = {}
        self.color_generator = self._generate_colors()
        
        # Validazione parametri dopo aver inizializzato gli attributi
        if save_video and output_path is None:
            raise ValueError("output_path è obbligatorio quando save_video=True")
        
        # Avvio thread di visualizzazione solo se tutto è OK
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
    
    def _generate_colors(self):
        """Genera colori distinti per gli ID usando HSV"""
        hue_step = 0.618033988749895  # Golden ratio per distribuzione uniforme
        hue = 0
        while True:
            # Converti HSV in RGB
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            # Converti in formato BGR per OpenCV
            bgr = tuple(int(c * 255) for c in reversed(rgb))
            yield bgr
            hue = (hue + hue_step) % 1.0
    
    def _get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """Ottiene un colore consistente per un ID specifico"""
        if track_id not in self.colors:
            self.colors[track_id] = next(self.color_generator)
        return self.colors[track_id]
    
    def _setup_video_writer(self, frame_shape: Tuple[int, int]):
        """Configura il video writer"""
        if self.save_video and self.video_writer is None:
            self.frame_size = (frame_shape[1], frame_shape[0])  # (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 1.0 / self.dt
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, fps, self.frame_size
            )
    
    def draw(self, frame: np.ndarray, boxes: torch.Tensor):
        """
        Disegna i bounding box sul frame e gestisce la visualizzazione
        
        Args:
            frame (np.ndarray): Frame di input
            boxes (torch.Tensor): Tensor (N, 5) con [id, x1, y1, x2, y2]
        """
        # Copia il frame per non modificare l'originale
        display_frame = frame.copy()
        
        # Setup video writer se necessario
        self._setup_video_writer(frame.shape)
        
        # Converti boxes in numpy se necessario
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        # Disegna ogni bounding box
        for box in boxes:
            if len(box) >= 5:
                track_id = int(box[0])
                x1, y1, x2, y2 = map(int, box[1:5])
                
                # Ottieni colore per questo ID
                color = self._get_color_for_id(track_id)
                
                # Disegna il bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepara il testo con l'ID
                text = f"ID: {track_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                
                # Calcola dimensioni del testo
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )
                
                # Disegna sfondo per il testo
                text_x = x1
                text_y = y1 - 10
                
                # Assicurati che il testo sia visibile
                if text_y < text_height:
                    text_y = y1 + text_height + 10
                
                cv2.rectangle(
                    display_frame,
                    (text_x, text_y - text_height - 5),
                    (text_x + text_width + 5, text_y + 5),
                    color,
                    -1
                )
                
                # Disegna il testo
                cv2.putText(
                    display_frame, text, (text_x + 2, text_y),
                    font, font_scale, (255, 255, 255), font_thickness
                )
        
        # Aggiungi al buffer per la visualizzazione temporizzata
        current_time = time.time()
        if not self.frame_buffer.full():
            self.frame_buffer.put((display_frame.copy(), current_time))
        else:
            # Se il buffer è pieno, sostituisci l'ultimo frame
            try:
                self.frame_buffer.get_nowait()
            except:
                pass
            self.frame_buffer.put((display_frame.copy(), current_time))
    
    def _display_loop(self):
        """Loop principale per la visualizzazione temporizzata"""
        while not self.should_stop:
            try:
                if not self.frame_buffer.empty():
                    frame, frame_time = self.frame_buffer.get(timeout=0.1)
                    
                    # Calcola quanto tempo aspettare
                    current_time = time.time()
                    elapsed = current_time - self.last_display_time
                    
                    if elapsed < self.dt:
                        # Aspetta il tempo rimanente
                        time.sleep(self.dt - elapsed)
                    
                    # Mostra il frame
                    cv2.imshow('SORT Tracking', frame)
                    
                    # Salva il frame se richiesto
                    if self.save_video and self.video_writer is not None:
                        self.video_writer.write(frame)
                    
                    self.last_display_time = time.time()
                    
                    # Controlla se l'utente vuole uscire
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.should_stop = True
                        break
                else:
                    time.sleep(0.01)  # Breve pausa se il buffer è vuoto
            except:
                time.sleep(0.01)
    
    def close(self):
        """Chiude il visualizer e rilascia le risorse"""
        self.should_stop = True
        
        # Aspetta che il thread di visualizzazione finisca
        if hasattr(self, 'display_thread') and self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
        
        # Chiudi video writer
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
        
        # Chiudi finestre OpenCV
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Distruttore per assicurarsi che le risorse vengano rilasciate"""
        if hasattr(self, 'should_stop'):  # Controlla se l'oggetto è stato inizializzato
            self.close()