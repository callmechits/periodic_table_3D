# **Gesture-Controlled Interactive Periodic Table**

haunt.gg/chits (Visit and leave a comment, I just want to increase my viewcount lol)

A real-time program which turns a webcam into a visual interactive periodic table.
You can open the periodic table with a two-hand gesture, select elements by hovering their thumb, and explore a simplified animated 3D Bohr-model representation using pinch-to-zoom and two-finger rotation.

The project is built with Python, OpenCV, MediaPipe, and NumPy.

### **My Specs**

- Python: 3.12.10
- Virtual environment - .venv
- MediaPipe: 0.10.13
- OpenCV: opencv-python
- NumPy: numpy

### **How to Set it Up**

- Open the folder where you save all files
- Right click and open Windows Powershell
- Run these commands (in order):
1) **py -3.12 -m venv .venv**

2) ****put in a dot and backslash here**\.venv\Scripts\Activate.ps1** (If you get an error here run this first: **Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser**, confirm that you see this: *(.venv) PS C:\...\your-project>* after executing)

3) **python -m pip install --upgrade pip**

4) **python -m pip install opencv-python mediapipe numpy**

### **How to Use**
(Let's say you've downloaded it and it works)
- Draw a rectangle with both index tips first touching each other then ending the rectangle while touching each other
- This should open up a periodic table (if it does not, just do it slowly or faster and with more accuracy, also first start apart and then bring the indexes closer) Example below: <img width="801" height="628" alt="Screenshot 2026-10-08 042616" src="https://github.com/user-attachments/assets/a71c97fa-bb72-4b23-b5a9-c7854d104b82" />
- Now, use your thumb to select any element that you wanna see, there is a visualizer attached to the thumb to make it easier to spot (colored yellow)
- It should open up the element spinning around and rotating on its own as well. Example below: <img width="796" height="630" alt="Screenshot 2026-10-08 042644" src="https://github.com/user-attachments/assets/fe87b2ed-2f28-4d07-b804-ed8de64a4ac3" />
- There's a display for Element properties (top left), Zoom (top left) and FPS Counter (top right) to look at optimization and useful information about the elements
- A legend (bottom left) for n/p/e and a forward/backward button to go visit the next/previous element respectively
- A button to switch between letting it spin on its own or spinning it manually (top right)
- To spin it manually, you just bring the index and middle fingers close (close to touching) and then you can rotate it around to a certain threshold in all directions. (Threshold because it can glitch out otherwise)
- A button to go back to the periodic table and select another element
- If you make a rectangle again, the table will close.
VOILA!!! Just keep practicing your rectangles LOL.

### **Features:**

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

### **Currently Working On**

- Thinking of adding a custom builder which allows you to input n/p/e to build custom elements (and maybe display reasons why a given element cannot be built)
- Maybe adding isotopes for all elements
- Sizes of orbits, neutrons, electrons, protons
