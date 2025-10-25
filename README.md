Live Video Streaming Solution using GStreamer and H.264

Overview:
This project captures live video from a webcam using GStreamer, encodes it with H.264, and streams it via UDP. A simple HTML5 player connects to the stream.

Prerequisites:
- Python 3.7+
- GStreamer with plugins:
  sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly

Setup:
1. Clone the repo:
   git clone https://github.com/yourusername/video_streaming_solution.git
   cd video_streaming_solution

2. Install Python dependencies:
   pip install -r requirements.txt

3. Start the stream:
   python app/streamer.py

4. Run a local server to serve the web player:
   cd app
   python -m http.server 8080

5. Open your browser:
   http://localhost:8080/static/index.html

Notes:
- The stream uses UDP on port 5000.
- You can adapt the pipeline to stream to Amazon Kinesis or other endpoints.
- For object detection, integrate OpenCV or TensorFlow in a separate thread.

Troubleshooting:
- Ensure your webcam is accessible via /dev/video0.
- Use `gst-inspect-1.0` to verify plugin availability.