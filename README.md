# Air Draw

Draw in the air using just your hand and a webcam. No stylus, no touchscreen, nothing extra. Just python, a camera and your fingers.

---

## What is this

Air Draw is a python script that tracks your hand through your webcam and lets you paint on screen by moving your index finger around. It uses MediaPipe under the hood to figure out where your hand and fingers are, and OpenCV to show the camera feed and draw the lines.

Theres also a glowing skeleton that shows up over your hand which looks pretty cool honestly.

---

## Requirements

You need python 3.8 or above. Tested on 3.10 personally.

Install the dependencies:

```
pip install opencv-python mediapipe numpy
```

The script will download a model file called `hand_landmarker.task` automatically the first time you run it (about 8MB). Make sure you're connected to the internet for that first run.

---

## How to run

Just run the script:

```
python air_draw.py
```

A window called "Air Canvas" will open showing your webcam. Hold your hand up in front of the camera and start drawing.

Press `q` to quit.

---

## Controls

**Drawing**
Raise your index finger only. Move it around and it will draw on screen. Thats it.

**Lift the brush (stop drawing without changing color)**
While your index finger is up, bring your thumb close to your index fingertip (like a pinch). When they get close enough the brush lifts and you stop drawing. A little "Lifted" label shows up so you know it worked. Unpinch to draw again.

**Pick a color / Clear the canvas**
Raise both your index finger and your middle finger at the same time. You'll see a colored rectangle appear between your fingers. Move your hand into the colored bar at the top of the screen to select that color. The colors from left to right are:

- Blue (0 to 100px)
- Green (100 to 200px)
- Red (200 to 300px)
- Yellow (300 to 400px)
- Clear (400 to 500px) -- this wipes the whole canvas

**Two hands**
The program supports two hands at the same time. Each hand can draw independently which is fun to play around with.

---

## Stuff worth knowing

- There's a smoothing system on the landmark positions so the lines don't jitter too much when your hand trembles a bit.
- The FPS counter shows up in the top right corner in purple.
- If tracking briefly drops (like your hand goes out of frame for a split second) the program holds the last known hand position for a few frames before giving up.
- Resolution is set to 1280x720 by default. If your webcam doesn't support that it'll fall back to whatever your camera can do.

---

## Known issues

- Sometimes when you switch from drawing mode to selection mode quickly the brush leaves a small stray line before resetting. Havent fixed this yet.
- The color picker only works in the top ~65 pixels of the screen so if your camera feed is small it might be tricky to reach.
- In low light the hand detection gets pretty unreliable. Good lighting makes a huge difference.

---

## Files

```
air_draw.py           -- the main script
hand_landmarker.task  -- downloaded automatically on first run
```

---
