import pyautogui
import time
import sys
from datetime import datetime
import random as rd

pyautogui.FAILSAFE = False

print('---- running ----')


seconds = 0

while True:
    time.sleep(5)
    seconds += 5
    # move mouse by 50 pixels to the right
    pyautogui.moveRel(50, 0, duration=0.25)
    # move mouse by 50 pixels left
    pyautogui.moveRel(-50, 0, duration=0.25)
    # every minute write message to console
    # click where the mouse is
    pyautogui.click()
    if seconds % 60 == 0:
        print(" Movement made at {}".format(datetime.now().time()))
