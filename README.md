**Gesture-Controlled Interactive Periodic Table**

A real-time which turns a webcam into a visual interactive periodic table.
You can open the periodic table with a two-hand gesture, select elements by hovering their thumb, and explore a simplified animated 3D Bohr-model representation using pinch-to-zoom and two-finger rotation.

The project is built with Python, OpenCV, MediaPipe, and NumPy.

**My Specs**

- Python: 3.12.10
- Virtual environment - .venv
- MediaPipe: 0.10.13
- OpenCV: opencv-python
- NumPy: numpy

**How to Set it Up**

- Open the folder where you save all files
- Right click and open Windows Powershell
- Run these commands (in order):
1) *py -3.12 -m venv .venv*

2) *.\.venv\Scripts\Activate.ps1* (If you get an error here run this first: *Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser*, confirm that you see this: *(.venv) PS C:\...\your-project>* after executing)

3) *python -m pip install --upgrade pip*

4) *python -m pip install opencv-python mediapipe numpy*



**Features:**

- Two-hand rectangle gesture to open/close the periodic table
- Interactive periodic table containing all 118 elements
- Thumb-hover selection with dwell-based confirmation
- Animated 3D Bohr-model visualization
- Pinch-to-zoom using thumb/index-finger distance
- Two-finger manual rotation using the index and middle fingers
- Automatic model rotation when manual rotation is disabled
- Previous/next element navigation
- Back button to return to the periodic table
- Threaded camera capture to reduce latency
- Live FPS counter
- Element-category color coding
- Real-time hand tracking using MediaPipe

**Currently Working On**

- Thinking of adding a custom builder which allows you to input n/p/e to build custom elements (and maybe display reasons why a given element cannot be built)
- Maybe adding isotopes for all elements
- Sizes of orbits, neutrons, electrons, protons
