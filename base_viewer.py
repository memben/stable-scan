import weakref
import moderngl_window as mglw
from moderngl_window import activate_context, get_local_window_cls
from moderngl_window.scene.camera import KeyboardCamera
from moderngl_window.timers.clock import Timer


class CameraWindow(mglw.WindowConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio, near=0.01, far=10.0)
        self.camera.velocity = 2.0
        self.camera_enabled = True
        return
        # inspired by run_window_config
        # custom init code
        mglw.setup_basic_logging(self.log_level)
        parser = mglw.create_parser()
        self.add_arguments(parser)
        values = mglw.parse_args(args=kwargs, parser=parser)
        self.argv = values
        window_cls = get_local_window_cls(values.window)

        # Calculate window size
        size = values.size or self.window_size
        size = int(size[0] * values.size_mult), int(size[1] * values.size_mult)

        # Resolve cursor
        show_cursor = values.cursor
        if show_cursor is None:
            show_cursor = self.cursor

        window = window_cls(
            title=self.title,
            size=size,
            fullscreen=self.fullscreen or values.fullscreen,
            resizable=values.resizable
            if values.resizable is not None
            else self.resizable,
            gl_version=self.gl_version,
            aspect_ratio=self.aspect_ratio,
            vsync=values.vsync if values.vsync is not None else self.vsync,
            samples=values.samples if values.samples is not None else self.samples,
            cursor=show_cursor if show_cursor is not None else True,
            backend=values.backend,
        )
        window.print_context_info()
        activate_context(window=window)
        self.timer = Timer() # TODO(memben): Figure out: self.timer or Timer()
        config = self(ctx=window.ctx, wnd=window, timer=self.timer)
        # Avoid the event assigning in the property setter for now
        # We want the even assigning to happen in WindowConfig.__init__
        # so users are free to assign them in their own __init__.
        window._config = weakref.ref(config)

        # Swap buffers once before staring the main loop.
        # This can trigged additional resize events reporting
        # a more accurate buffer size
        window.swap_buffers()
        window.set_default_viewport()

        self.timer.start()

    # Custom run code for now
    def run(self):
        return ValueError("Note implemented yet")
        window = mglw.window()
        self.timer = window.timer
        while not window.is_closing:
            current_time, delta = self.timer.next_frame()

            # if config.clear_color is not None:
            #     window.clear(*config.clear_color)

            # Always bind the window framebuffer before calling render
            window.use()

            window.render(current_time, delta)
            if not window.is_closing:
                window.swap_buffers()

        # _, duration = timer.stop()
        # window.destroy()
        # if duration > 0:
            # logger.info(
            #     "Duration: {0:.2f}s @ {1:.2f} FPS".format(
            #         duration, window.frames / duration
            #     )
            # )

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