import time
import pyautogui
import pyperclip

"""
采用pyautogui.position() 获取要发送消息的窗口位置。
首先把鼠标指针放到聊天窗口，然后运行代码。
"""

pos = pyautogui.position()
x, y = pos.x, pos.y

time.sleep(1)

text = "are you kidding me ?"

for i in range(100):
    pyautogui.click(x, y)
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")
    pyautogui.typewrite("\n")
    time.sleep(0.5)
