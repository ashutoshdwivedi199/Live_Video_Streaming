import os
import cv2
import subprocess
import time
import json
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
from ultralytics import YOLO

def load_config(config_path='stream_config.json'):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load config: {e}")
        return {
            "bitrate": 2048,
            "speed_preset": "ultrafast",
            "target_duration": 5,                               ## Gstreamer aim to create new segments of length target_duration.
            "max_files": 30,                                    ## Provides bigger playback with tradeoff on disk size
            "segment_location": "./hls/%05d.ts",
            "playlist_location": "./hls/test.m3u8",
            "playlist_root": "http://localhost:8554/hls/",
            "frames_interval": 15,                              ## Tradeoff between viewing fluidity and latency
            "detection_conf": 0.75,                             ## Thresholding for model detection => due to varied luminiousity and other factors
            "obj_detection_interval": 10                        ## Tradeoff between detection and latency
        }

http_proc = None
def launch_http_server():
    global http_proc
    if http_proc is None or http_proc.poll() is not None:
        http_proc = subprocess.Popen(
            ["python", "-m", "http.server", "8554"],
            cwd=os.getcwd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ HTTP server started on port 8554.")

class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.rects = OrderedDict()

    def register(self, centroid, rect):
        self.objects[self.next_object_id] = centroid
        self.rects[self.next_object_id] = rect
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.rects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.rects

        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for centroid, rect in zip(input_centroids, rects):
                self.register(centroid, rect)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), np.array(input_centroids))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.rects[object_id] = rects[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], rects[col])

        return self.rects

def start_object_detection_stream():
    print("******************     LIVE VIDEO STREAMING WITH OBJECT DETECTION     ******************")
    os.makedirs('./hls', exist_ok=True)

    model = YOLO('yolov5n.pt')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    config = load_config()

    gst_command = [
        r'gst-launch-1.0.exe',
        'fdsrc', '!',
        'rawvideoparse', 'format=bgr', 'width=640', 'height=480', 'framerate=15/1',
        '!', 'videoconvert',
        '!', 'x264enc', f'tune=zerolatency', f'bitrate={config["bitrate"]}', f'speed-preset={config["speed_preset"]}',
        '!', 'mpegtsmux',
        '!', 'hlssink',
        f'location={config["segment_location"]}',
        f'playlist-location={config["playlist_location"]}',
        f'playlist-root={config["playlist_root"]}',
        f'target-duration={config["target_duration"]}',
        f'max-files={config["max_files"]}'
    ]
    gst_process = subprocess.Popen(gst_command, stdin=subprocess.PIPE)

    frame_interval = 1 / config["frames_interval"]
    count = 0
    tracker = CentroidTracker()
    boxes = []
    launch_http_server()
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break

            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480))

            if count % config["obj_detection_interval"] == 0:
                results = model.predict(frame, stream=False, verbose=False)[0]
                boxes = []
                for box in results.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box[:4]
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
                tracked = tracker.update(boxes)
            else:
                tracked = tracker.update(boxes)

            for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                          results.boxes.cls.cpu().numpy(),
                          results.boxes.conf.cpu().numpy()):
                if conf >= config["detection_conf"]:
                    x1, y1, x2, y2 = map(int, box[:4])
                    label = model.names[int(cls)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            annotated = frame

            try:
                gst_process.stdin.write(annotated.tobytes())
            except Exception as e:
                print(f"GStreamer write error: {e}")
                break

            count += 1
            elapsed = time.time() - start_time
            #print(frame_interval, elapsed)
            time.sleep(max(0, frame_interval - elapsed))
    finally:
        cap.release()
        if gst_process.stdin:
            gst_process.stdin.close()
        gst_process.wait()
        cv2.destroyAllWindows()

        # 🧹 Delete all files in ./hls/
        hls_dir = './hls'
        try:
            for filename in os.listdir(hls_dir):
                file_path = os.path.join(hls_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("Cleaned up HLS segment files.")
        except Exception as e:
            print(f"Error cleaning up HLS files: {e}")

if __name__ == "__main__":
    start_object_detection_stream()