import sys
import numpy as np
from vispy import app, gloo

# Vertex shader
vertex = """
attribute vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Fragment shader
fragment = """
void main() {
    float height = 0.5 * gl_FragCoord.y / 600.0;  // Scale height to window size
    gl_FragColor = vec4(0.0, height, 0.0, 1.0);   // Green bars
}
"""

import sys
import numpy as np
from vispy import app, gloo

# Vertex shader
vertex = """
attribute vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Fragment shader
fragment = """
void main() {
    float height = 0.5 * gl_FragCoord.y / 600.0;  // Scale height to window size
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);   // Black bars
}
"""

# Create a Canvas
class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, title='Moving Bar Graph', size=(800, 600), keys='interactive')

        # Create shader program
        self.program = gloo.Program(vertex, fragment)

        # Initialize data
        self.data = np.zeros((100, 2), dtype=np.float32)
        self.data[:, 0] = np.linspace(-2.0, 2.0, 100)  # Increase x-axis range

        # Set attribute
        self.program['position'] = self.data

        # Timer for updating data
        self.timer = app.Timer(0.1, connect=self.on_timer, start=True)  # Update every 0.1 seconds

        # Counter for moving bars
        self.counter = 0

        self.show()

    def on_draw(self, event):
        gloo.clear(color=True)
        self.program.draw('line_strip')

    def on_timer(self, event):
        # Generate new random data
        new_data = np.random.rand(100, 2).astype(np.float32)
        new_data[:, 0] = np.linspace(-2.0, 2.0, 100)  # Keep x-axis range consistent

        # Update data
        self.data[:, 1] = new_data[:, 1]

        # Shift bars horizontally
        self.data[:, 0] -= 0.01
        self.counter += 1

        # Reset bars when they move out of the screen
        if self.counter >= 100:
            self.data[:, 0] = np.linspace(-2.0, 2.0, 100)
            self.counter = 0

        # Update attribute
        self.program['position'] = self.data

        # Trigger redraw
        self.update()

if __name__ == '__main__':
    canvas = Canvas()
    sys.exit(app.run())
