import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


# it is default config batukh, change variable named config at bottom ,there are many visuals and overlay but i have commented them to make it clean
@dataclass
class GestureConfig:
    """Configuration for gesture detection"""
    max_hands: int = 1
    gesture_threshold: float = 40.0
    cooldown_time: float = 0.8
    smoothing_factor: float = 0.7
    confidence_threshold: float = 0.4
    zone_dwell_time: float = 0.1 # Time to stay in zone before triggering

class GestureController:
    def __init__(self, config: GestureConfig = None):
        self.config = config or GestureConfig()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_hands,
            min_detection_confidence=self.config.confidence_threshold,
            min_tracking_confidence=self.config.confidence_threshold
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Tracking variables
        self.prev_positions = []
        self.last_gesture_time = 0
        self.gesture_history = []
        self.current_zone = None
        self.zone_entry_time = 0
        self.zone_triggered = False
        
        # Smoothing buffer
        self.position_buffer = []
        self.buffer_size = 5
        
        # Safety settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
    def smooth_position(self, x: int, y: int) -> Tuple[int, int]:
        """Apply smoothing to reduce jitter"""
        self.position_buffer.append((x, y))
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)
            
        
        weights = np.linspace(0.5, 1.0, len(self.position_buffer))
        weights /= weights.sum()
        
        smooth_x = sum(pos[0] * w for pos, w in zip(self.position_buffer, weights))
        smooth_y = sum(pos[1] * w for pos, w in zip(self.position_buffer, weights))
        
        return int(smooth_x), int(smooth_y)
    
    def detect_zone_gesture(self, x: int, y: int, w: int, h: int) -> Optional[str]:
        """Detect gesture based on which screen zone the finger is in"""
        current_time = time.time()
        new_zone = None
        
        #  zone boundaries
        if y < h//3:
            new_zone = "swipe_up"
        elif y > 2*h//3:
            new_zone = "swipe_down"
        elif x < w//3:
            new_zone = "swipe_left"
        elif x > 2*w//3:
            new_zone = "swipe_right"
        else:
            new_zone = "center"
        
        # Checkzone change
        if new_zone != self.current_zone:
            self.current_zone = new_zone
            self.zone_entry_time = current_time
            self.zone_triggered = False
            return None
        
       
        if (new_zone != "center" and 
            not self.zone_triggered and 
            (current_time - self.zone_entry_time) > self.config.zone_dwell_time and
            (current_time - self.last_gesture_time) > self.config.cooldown_time):
            
            self.zone_triggered = True
            self.last_gesture_time = current_time
            self.gesture_history.append((new_zone, current_time))
            
            
            self.gesture_history = [(g, t) for g, t in self.gesture_history 
                                  if current_time - t < 10]
            
            return new_zone
        
        return None
    
    def execute_gesture_action(self, gesture: str) -> None:
        """Execute the corresponding action for detected gesture"""
        actions = {
            "swipe_right": lambda: pyautogui.press('d'),
            "swipe_left": lambda: pyautogui.press('a'),
            "swipe_up": lambda: pyautogui.press('w'),
            "swipe_down": lambda: pyautogui.press('s')
        }
        
        try:
            if gesture in actions:
                actions[gesture]()
                print(f"✓ Executed: {gesture.replace('_', ' ').title()}")
        except Exception as e:
            print(f"✗ Error executing gesture {gesture}: {e}")
    
    def draw_grid_overlay(self, img: np.ndarray) -> np.ndarray:
        """Draw the grid zones on the image"""
        h, w = img.shape[:2]
        
        # Draw grid lines
        cv2.line(img, (w//3, 0), (w//3, h), (200, 200, 200), 2)
        cv2.line(img, (2*w//3, 0), (2*w//3, h), (200, 200, 200), 2)
        cv2.line(img, (0, h//3), (w, h//3), (200, 200, 200), 2)
        cv2.line(img, (0, 2*h//3), (w, 2*h//3), (200, 200, 200), 2)
        
        #   zonesLabel
        cv2.putText(img, "UP", (w//2 - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(img, "DOWN", (w//2 - 40, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(img, "LEFT", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(img, "RIGHT", (w - 80, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        # Highlight 
        if self.current_zone and self.current_zone != "center":
            zone_color = (0, 255, 255)  # Yellow 
            alpha = 0.3  
            
            overlay = img.copy()
            if self.current_zone == "swipe_up":
                cv2.rectangle(overlay, (0, 0), (w, h//3), zone_color, -1)
            elif self.current_zone == "swipe_down":
                cv2.rectangle(overlay, (0, 2*h//3), (w, h), zone_color, -1)
            elif self.current_zone == "swipe_left":
                cv2.rectangle(overlay, (0, 0), (w//3, h), zone_color, -1)
            elif self.current_zone == "swipe_right":
                cv2.rectangle(overlay, (2*w//3, 0), (w, h), zone_color, -1)
            
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        return img
    
    def draw_info_overlay(self, img: np.ndarray, gesture: Optional[str] = None) -> np.ndarray:
        """Draw information overlay on the image"""
        h, w = img.shape[:2]
        
        #  text
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        
        status_text = "Grid Zone Control Active"
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        #  current zone
        zone_text = f"Zone: {self.current_zone.replace('_', ' ').title() if self.current_zone else 'None'}"
        cv2.putText(img, zone_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 0), 2)
        
        
        if gesture:
            gesture_text = f"Action: {gesture.replace('_', ' ').title()}"
            cv2.putText(img, gesture_text, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 255), 2)
        
        
        history_text = f"Actions: {len(self.gesture_history)}"
        cv2.putText(img, history_text, (w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "ESC: Quit",
            "R: Reset tracking",
            "Space: Pause/Resume",
            "Move finger to edge zones to trigger actions"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(img, instruction, (10, h - 10 - i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return img
    
    def reset_tracking(self) -> None:
        """Reset all tracking variables"""
        self.prev_positions.clear()
        self.position_buffer.clear()
        self.gesture_history.clear()
        self.current_zone = None
        self.zone_entry_time = 0
        self.zone_triggered = False
        print(" Tracking reset")
    
    def process_frame(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Process a single frame and detect gestures"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        h, w = img.shape[:2]
        detected_gesture = None
        
        #  grid overlay
        img = self.draw_grid_overlay(img)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                #  index finger tip (landmark 8)
                landmark = hand_landmarks.landmark[8]
                x, y = int(landmark.x * w), int(landmark.y * h)
                
                #  smoothing
                smooth_x, smooth_y = self.smooth_position(x, y)
                
                #  hand landmarks
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                #  finger tip with  visual
                #innerbluecircl--cv2.circle(img, (smooth_x, smooth_y), 10, (255, 0, 0), -1)
               #innercircle cv2.circle(img, (smooth_x, smooth_y), 15, (255, 255, 255), 2)
                
                # Detect gesture  zone
                detected_gesture = self.detect_zone_gesture(smooth_x, smooth_y, w, h)
                
                
                self.prev_positions.append((smooth_x, smooth_y))
                if len(self.prev_positions) > 10:
                    self.prev_positions.pop(0)
                
                #  movement trail
                if len(self.prev_positions) > 1:
                    for i in range(1, len(self.prev_positions)):
                        cv2.line(img, self.prev_positions[i-1], self.prev_positions[i], 
                                (0, 255, 255), 2)
        
        return img, detected_gesture

def main():
    """Main execution function"""
    print(" Starting Grid Zone Gesture Control System")
    print(" Instructions:")
    print("   - Move your index finger to the edge zones to trigger actions")
    print("   - Stay in a zone  to activate")
    print("   - ESC: Quit")
    print("   - R: Reset tracking")
    print("   - Space: Pause/Resume")
    print("-" * 50)
    
    # Initialize controller
    config = GestureConfig(
        zone_dwell_time=0.01,  # Time to stay in zone before action
        cooldown_time=0.4,    # Minimum time between actions
        confidence_threshold=0.4
    )
    
    controller = GestureController(config)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    paused = False
    frame_count = 0
    fps_start_time = time.time()
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                print(" Failed to read from camera")
                break
            
            # Flip image 
            img = cv2.flip(img, 1)
            
            frame_count += 1
            
            if not paused:
                # Process frame for gesture detection
                img, detected_gesture = controller.process_frame(img)
                
                # Execute gesture action
                if detected_gesture:
                    controller.execute_gesture_action(detected_gesture)
            else:
                # Show paused indicator
                cv2.putText(img, "PAUSED", (img.shape[1]//2 - 60, img.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Draw information overlay
          #  img = controller.draw_info_overlay(img, detected_gesture if not paused else None)
            
            # Calculate and display FPS
    #        if frame_count % 30 == 0:
       #         fps = 30 / (time.time() - fps_start_time)
       #         fps_start_time = time.time()
           #     cv2.putText(img, f"FPS: {fps:.1f}", (img.shape[1] - 100, 30), 
           #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Grid Zone Gesture Control", img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                controller.reset_tracking()
            elif key == ord(' '):  # Space
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f" Unexpected error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        controller.hands.close()
        print(" Cleanup completed")

if __name__ == "__main__":
    main()