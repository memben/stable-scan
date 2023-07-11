import math
from pathlib import Path
from typing import List

import moderngl
import moderngl_window
from moderngl_window import WindowConfig, resources
from moderngl_window.conf import settings
from moderngl_window.meta import ProgramDescription
from moderngl_window.scene.camera import KeyboardCamera
from moderngl_window.timers.clock import Timer


class CustomSetup():
    """
    Custom setup for moderngl_window, based on the custom example setup and WindowConfig
    """
    def __init__(self):
        settings.WINDOW['title'] = self.title or "Custom Viewer"
        settings.WINDOW['size'] = self.window_size or (1024, 1024)
        settings.WINDOW['aspect_ratio'] = self.aspect_ratio or 1.0

        if self.resource_dir:
            resources.register_dir(Path(self.resource_dir).resolve())

        self.wnd = moderngl_window.create_window_from_settings()
        self.ctx = self.wnd.ctx

        # register event methods
        self.wnd.resize_func = self.resize
        self.wnd.iconify_func = self.iconify
        self.wnd.key_event_func = self.key_event
        self.wnd.mouse_position_event_func = self.mouse_position_event
        self.wnd.mouse_drag_event_func = self.mouse_drag_event
        self.wnd.mouse_scroll_event_func = self.mouse_scroll_event
        self.wnd.mouse_press_event_func = self.mouse_press_event
        self.wnd.mouse_release_event_func = self.mouse_release_event
        self.wnd.unicode_char_entered_func = self.unicode_char_entered
        self.wnd.close_func = self.close

    def render(self, time, frame_time):
        self.ctx.clear(
            (math.sin(time) + 1.0) / 2,
            (math.sin(time + 2) + 1.0) / 2,
            (math.sin(time + 3) + 1.0) / 2,
        )

    def run(self):
        timer = Timer()
        timer.start()

        while not self.wnd.is_closing:
            self.wnd.clear()
            time, frame_time = timer.next_frame()
            self.render(time, frame_time)
            self.wnd.swap_buffers()

        self.wnd.destroy()

    def resize(self, width: int, height: int):
        print("Window was resized. buffer size is {} x {}".format(width, height))

    def iconify(self, iconify: bool):
        """Window hide/minimize and restore"""
        print("Window was iconified:", iconify)

    def key_event(self, key, action, modifiers):
        print("Key:", key, "action:", action, "modifiers:", modifiers)
            # toggle cursor
            # if key == keys.C:
            #     self.wnd.cursor = not self.wnd.cursor

            # # Shuffle window tittle
            # if key == keys.T:
            #     title = list(self.wnd.title)
            #     random.shuffle(title)
            #     self.wnd.title = ''.join(title)

            # # Toggle mouse exclusivity
            # if key == keys.M:
            #     self.wnd.mouse_exclusivity = not self.wnd.mouse_exclusivity

    def mouse_position_event(self, x, y, dx, dy):
        print("Mouse position pos={} {} delta={} {}".format(x, y, dx, dy))

    def mouse_drag_event(self, x, y, dx, dy):
        print("Mouse drag pos={} {} delta={} {}".format(x, y, dx, dy))

    def mouse_scroll_event(self, x_offset, y_offset):
        print("mouse_scroll_event", x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        print("Mouse button {} pressed at {}, {}".format(button, x, y))
        print("Mouse states:", self.wnd.mouse_states)

    def mouse_release_event(self, x: int, y: int, button: int):
        print("Mouse button {} released at {}, {}".format(button, x, y))
        print("Mouse states:", self.wnd.mouse_states)

    def unicode_char_entered(self, char):
        print("unicode_char_entered:", char)

    def close(self):
        print("Window was closed")

    def load_program(
        self,
        path=None,
        vertex_shader=None,
        geometry_shader=None,
        fragment_shader=None,
        tess_control_shader=None,
        tess_evaluation_shader=None,
        defines: dict = None,
        varyings: List[str] = None,
    ) -> moderngl.Program:
        """Equals WindowConfig.load_program"""
        return resources.programs.load(
            ProgramDescription(
                path=path,
                vertex_shader=vertex_shader,
                geometry_shader=geometry_shader,
                fragment_shader=fragment_shader,
                tess_control_shader=tess_control_shader,
                tess_evaluation_shader=tess_evaluation_shader,
                defines=defines,
                varyings=varyings,
            )
        )


class CameraWindow(CustomSetup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = KeyboardCamera(
            self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio, near=0.01, far=10.0
        )
        self.camera.velocity = 2.0
        self.camera_enabled = True

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()

    def mouse_position_event(self, x: int, y: int, dx, dy):
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)

if __name__ == '__main__':
    app = CameraWindow()
    app.run()