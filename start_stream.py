import subprocess
import os
import shutil

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
        
def start_stream():
    gst_command = (
        'gst-launch-1.0 ksvideosrc ! '
        'videoconvert ! '
        'x264enc tune=zerolatency bitrate=512 speed-preset=superfast ! '
        'mpegtsmux ! '
        'hlssink location=./hls/segment_%05d.ts '
        'playlist-location=./hls/test.m3u8 '
        'playlist-root=http://localhost:8554/hls/ '
        'target-duration=5 max-files=5'
    )
    launch_http_server()
    try:
        print("Starting GStreamer pipeline...")
        subprocess.run(gst_command, shell=True)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        hls_dir = './hls'
        if os.path.exists(hls_dir):
            print("Cleaning up HLS directory...")
            for filename in os.listdir(hls_dir):
                file_path = os.path.join(hls_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as cleanup_error:
                    print(f"Failed to delete {file_path}: {cleanup_error}")
            print("Cleanup complete.")

if __name__ == "__main__":
    start_stream()