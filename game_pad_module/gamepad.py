import time
import vgamepad as vg

class GamePad:

    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
        time.sleep(2)

    def joystick(self, x_pos = 0, y_pos = 0):
        self.gamepad.left_joystick(x_value=x_pos*-1, y_value=y_pos*-1)
        self.gamepad.update()