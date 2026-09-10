"""Check the real entry point on the port used by managed Gradio Spaces."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        SPACE_ID="local/startup-check",
        GRADIO_SERVER_PORT="8000",
        GOOGLE_API_KEY="",
        GRADIO_ANALYTICS_ENABLED="False",
        PYTHONUNBUFFERED="1",
    )
    # An empty API key keeps this check independent of Gemini and private PDFs.
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(
            [sys.executable, "app.py"], cwd=root, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"App exited with code {process.returncode}")
                log.seek(0)
                started = "Running on local URL:" in log.read()
                if started:
                    try:
                        with urlopen("http://127.0.0.1:7860/", timeout=2) as response:
                            assert response.status == 200
                            assert b"gradio" in response.read().lower()
                        with urlopen("http://127.0.0.1:7860/config", timeout=2) as response:
                            config = json.load(response)
                            assert any(c["type"] == "chatbot" for c in config["components"])
                        print("PASS: app.py serves the UI and chatbot config on port 7860 without an API key.")
                        return
                    except (URLError, TimeoutError):
                        pass
                time.sleep(0.5)
            raise RuntimeError("App did not become healthy on port 7860 within 60 seconds")
        except Exception:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
