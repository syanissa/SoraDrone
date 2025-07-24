FlySight is a Python-based GUI application that allows keyboard control of a DJI Tello drone while also leveraging real-time face recognition and MediaPipe face detection for tracking and intelligent movement decisions.


⚙️ Features

- Real-time face recognition using face_recognition library
- Face distance estimation using known face width and focal length
- Head position tracking with MediaPipe
- Intelligent movement control (e.g., auto-alignment, forward/backward)
- Keyboard control for manual drone movement via Tkinter GUI
- Takeoff/Land toggle button
- Dropdown to select the recognized face to track


🖼️ Add Your Face

To enable recognition and tracking:

Place a clear JPG or PNG image of your face inside the faces/ folder.
Name the file as you want to be recognized (e.g., Rio.jpg or Joean.png).
Restart the app. Your name will appear in the dropdown selector.


🧰 Requirements




🧑‍💻 How to Run

Make sure your DJI Tello is turned on and connected to your Wi-Fi, and then run the python file.

Use the GUI window to:

Select the face to track
Take off or land
Use W/A/S/D or arrow keys to fly manually

🛠️ Controls

Key	Action
W	Forward
A	Left
S	Backward
D	Right
↑ (Up arrow)	Ascend
↓ (Down arrow)	Descend
Q	Rotate left
E	Rotate right



📸 Tips for Best Performance

- Use high-resolution, well-lit face images
- Avoid sunglasses or obstructions in photos
- Fly in an open area with good lighting
- Make sure your PC and drone are on the same Wi-Fi network
