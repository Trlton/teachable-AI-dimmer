import subprocess

subprocess.run(
    ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
    stderr=subprocess.STDOUT
)
