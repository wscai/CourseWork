
from pynput.keyboard import Key,Controller

import socket
keyboard = Controller()
HOST = "192.168.0.61"  # Standard loopback interface address (localhost)
PORT = 9092
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    down = True
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if data==b"close":
                break
            if data:
                data = float(data.decode("utf-8"))
                if data<0:
                    for i in range(0, -int(data*5)):
                        keyboard.press(Key.media_volume_up)
                        keyboard.release(Key.media_volume_up)
                keyboard.press(Key.media_volume_down)
                keyboard.release(Key.media_volume_down)
                print(data)
