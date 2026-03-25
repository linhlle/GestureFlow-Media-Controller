# 🖐️ GestureFlow: AI-Powered macOS Media Controller

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/Computer_Vision-MediaPipe-green.svg)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GestureFlow** is a real-time computer vision application that allows users to control macOS system functions through intuitive hand gestures. By leveraging **Google MediaPipe** for landmark detection and a **Random Forest Classifier**, the system achieves high-precision control over cursor movement, system navigation, and volume levels.

---

## Key Features

* **Adaptive Mouse Tracking:** Smooth cursor control using index-finger positioning.
* **System Hotkeys:** Custom-trained gestures for:
    * **Spotlight Search** (`Cmd + Space`)
    * **Mission Control** (`Ctrl + Up`)
    * **App Switcher** (`Cmd + Tab`)
* **Geometric Volume Control:** Intuitive vertical hand-sliding logic to adjust macOS system volume via **AppleScript** (`osascript`).
* **Stability Engine:** Implements temporal voting (consensus logic) to filter out camera noise and prevent accidental triggers.

---

## Tech Stack

* **Language:** Python 3.9+
* **Vision & ML:** OpenCV, MediaPipe, Scikit-Learn
* **Automation:** PyAutoGUI, Subprocess (AppleScript)
* **Data Science:** Pandas, NumPy, Matplotlib, Seaborn

---

## Project Structure

| File | Purpose |
| :--- | :--- |
| `utils.py` | Centralized math for coordinate normalization. |
| `data_logger.py` | Tool for capturing and labeling normalized hand coordinates. |
| `train_model.py` | Trains the Random Forest and generates evaluation metrics. |
| `main.py` | The real-time execution engine and system automation. |
| `gesture_classifier.pkl` | The trained serialized model "brain." |

---

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/linhlle/GestureFlow-Media-Controller.git 
    cd GestureFlow-Media-Controller
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Grant Accessibility Permissions:**
    * Navigate to **System Settings > Privacy & Security > Accessibility**.
    * Enable your **Terminal** or **IDE** (e.g., VS Code) to allow the script to execute system hotkeys.

---

## Usage

* **Start the Engine:** Run `python main.py`.
* **Pointer Mode:** Relax your hand to move the cursor.
* **Command Mode:** Perform the **L-Shape**, **High-Five**, or **2-Finger Point** gestures to trigger macOS shortcuts.
* **Volume Mode:** Extend your thumb and move your hand vertically to adjust system audio.