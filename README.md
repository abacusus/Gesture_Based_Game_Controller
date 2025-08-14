# Gesture-Based Game Controller

Play Subway Surfers (via BlueStacks) and Chrome Dino using just your fingers and a webcam.  
This project uses OpenCV, Mediapipe, and PyAutoGUI to track your hand gestures in real time and translate them into in-game actions.

## Features
- Real-time hand tracking using Mediapipe
- Swipe gesture detection mapped to arrow key inputs
- Works with Subway Surfers, Chrome Dino. (rest of the programs are all experimental and won't work)
- Fast response time with gesture cooldown to prevent accidental moves
- Compatible with BlueStacks for Android game control

## Tech Stack
- Python 3
- OpenCV – Computer Vision library
- Mediapipe – Hand landmark detection
- PyAutoGUI – Simulating keyboard inputs

## Installation

1. Clone the repository
   git clone https://github.com/your-username/gesture-game-controller.git
   cd gesture-game-controller

2. Install dependencies
   pip install opencv-python mediapipe pyautogui

3. Ensure you have BlueStacks installed
   Download from: https://www.bluestacks.com
   Install Subway Surfers inside BlueStacks (or use Chrome Dino in your browser).

3. Run the script
   python gesture_controller.py

**How to Play**

Open your game (Subway Surfers in BlueStacks or Chrome Dino in Chrome).
Make sure the game window is active and focused.
Use your index finger for gestures:
**Swipe Right → Move right
Swipe Left → Move left
Swipe Up → Jump
Swipe Down → Roll/Duck**
