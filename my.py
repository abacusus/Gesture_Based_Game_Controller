import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class GestureConfig:
    """Configuration for gesture detection"""
    max_hands: int = 2
    cooldown_time: float = 0.8
    confidence_threshold: float = 0.6

class TekkenGestureController:
    def __init__(self):
        self.config = GestureConfig()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Tracking variables
        self.last_action_time = 0
        
        # Controls mapping
        self.controls = {
            # Left Hand Movement
            "left_open": 'd',      # Move RIGHT (open hand = go right)
            "left_closed": 'a',    # Move LEFT (closed hand = go left)
            "left_index": 'w',     # Move UP (index finger = up)
            "left_two": 's',       # Move DOWN (two fingers = down)
            
            # Right Hand Attacks
            "right_open": 'u',     # Heavy Attack
            "right_closed": 'h',   # Light Attack  
            "right_index": 'j',    # Medium Attack
            "right_two": 'k',      # Special Attack
        }
        
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1

    def count_extended_fingers(self, hand_landmarks) -> int:
        """Count how many fingers are extended"""
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_mcp = [5, 9, 13, 17]    # Corresponding MCP joints
        thumb_tip = 4
        thumb_ip = 2
        
        extended_fingers = 0
        
        # Check fingers (index, middle, ring, pinky)
        for tip, mcp in zip(finger_tips, finger_mcp):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
                extended_fingers += 1
        
        # Check thumb (different logic - x coordinate for horizontal hand)
        if hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_ip].x:
            extended_fingers += 1
            
        return extended_fingers

    def get_hand_gesture(self, hand_landmarks, hand_type: str) -> Optional[str]:
        """Get gesture type based on finger count"""
        finger_count = self.count_extended_fingers(hand_landmarks)
        
        if hand_type == "left":
            if finger_count == 5:    # All fingers open
                return "left_open"
            elif finger_count == 0:  # Fist
                return "left_closed"
            elif finger_count == 1:  # Just index finger
                return "left_index"
            elif finger_count == 2:  # Index + middle
                return "left_two"
                
        elif hand_type == "right":
            if finger_count == 5:    # All fingers open
                return "right_open"
            elif finger_count == 0:  # Fist
                return "right_closed"
            elif finger_count == 1:  # Just index finger
                return "right_index"
            elif finger_count == 2:  # Index + middle
                return "right_two"
                
        return None

    def detect_gestures(self, left_hand, right_hand) -> List[str]:
        """Detect gestures from both hands"""
        current_time = time.time()
        gestures = []
        
        if (current_time - self.last_action_time) < self.config.cooldown_time:
            return gestures

        # Check left hand for movement
        if left_hand:
            left_gesture = self.get_hand_gesture(left_hand, "left")
            if left_gesture:
                gestures.append(left_gesture)

        # Check right hand for attacks
        if right_hand:
            right_gesture = self.get_hand_gesture(right_hand, "right")
            if right_gesture:
                gestures.append(right_gesture)
        
        if gestures:
            self.last_action_time = current_time
            
        return gestures

    def execute_actions(self, gestures: List[str]) -> None:
        """Execute actions for detected gestures"""
        try:
            for gesture in gestures:
                if gesture in self.controls:
                    key = self.controls[gesture]
                    pyautogui.press(key)
                    
                    # Friendly names for display
                    gesture_names = {
                        "left_open": "🖐️ LEFT OPEN → RIGHT",
                        "left_closed": "✊ LEFT FIST → LEFT", 
                        "left_index": "☝️ LEFT 1 FINGER → UP",
                        "left_two": "✌️ LEFT 2 FINGERS → DOWN",
                        "right_open": "🖐️ RIGHT OPEN → HEAVY",
                        "right_closed": "✊ RIGHT FIST → LIGHT",
                        "right_index": "☝️ RIGHT 1 FINGER → MEDIUM",
                        "right_two": "✌️ RIGHT 2 FINGERS → SPECIAL"
                    }
                    print(f"🎮 {gesture_names[gesture]}")
                    
        except Exception as e:
            print(f"✗ Error: {e}")

    def draw_gesture_overlay(self, img: np.ndarray, left_hand, right_hand, gestures: List[str]) -> np.ndarray:
        """Draw gesture information overlay"""
        h, w = img.shape[:2]
        
        # Header
        cv2.putText(img, "TEKKEN - GESTURE CONTROL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 255), 2)
        
        # Left hand status
        left_fingers = self.count_extended_fingers(left_hand) if left_hand else "No hand"
        cv2.putText(img, f"LEFT: {left_fingers} fingers", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 0, 0), 1)
        
        # Right hand status  
        right_fingers = self.count_extended_fingers(right_hand) if right_hand else "No hand"
        cv2.putText(img, f"RIGHT: {right_fingers} fingers", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1)
        
        # Current gestures
        if gestures:
            gesture_text = " + ".join([g.replace('_', ' ').upper() for g in gestures])
            cv2.putText(img, f"ACTION: {gesture_text}", (w//2 - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
        
        # Instructions - Left Hand (Movement)
        left_instructions = [
            "LEFT HAND - MOVEMENT:",
            "🖐️ OPEN HAND = RIGHT (D)",
            "✊ CLOSED FIST = LEFT (A)", 
            "☝️ 1 FINGER = UP (W)",
            "✌️ 2 FINGERS = DOWN (S)"
        ]
        
        for i, instruction in enumerate(left_instructions):
            cv2.putText(img, instruction, (20, 120 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Instructions - Right Hand (Attacks)
        right_instructions = [
            "RIGHT HAND - ATTACKS:",
            "🖐️ OPEN HAND = HEAVY (U)",
            "✊ CLOSED FIST = LIGHT (H)", 
            "☝️ 1 FINGER = MEDIUM (J)",
            "✌️ 2 FINGERS = SPECIAL (K)"
        ]
        
        for i, instruction in enumerate(right_instructions):
            cv2.putText(img, instruction, (w - 200, 120 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw hand indicators
        if left_hand:
            wrist = left_hand.landmark[0]
            x, y = int(wrist.x * w), int(wrist.y * h)
            cv2.circle(img, (x, y), 20, (255, 0, 0), -1)  # Blue for left
            cv2.putText(img, f"L:{left_fingers}", (x-15, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
        if right_hand:
            wrist = right_hand.landmark[0]
            x, y = int(wrist.x * w), int(wrist.y * h)
            cv2.circle(img, (x, y), 20, (0, 255, 0), -1)  # Green for right
            cv2.putText(img, f"R:{right_fingers}", (x-15, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return img

    def process_frame(self, img: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Process frame and detect gestures"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        left_hand = None
        right_hand = None
        gestures = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                hand_type = "left" if handedness == "Left" else "right"
                
                if hand_type == "left":
                    left_hand = hand_landmarks
                else:
                    right_hand = hand_landmarks
                
                # Draw hand landmarks
                hand_color = (255, 0, 0) if hand_type == "left" else (0, 255, 0)
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(color=hand_color, thickness=2, circle_radius=3),
                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=hand_color, thickness=1)
                )
        
        # Detect gestures
        gestures = self.detect_gestures(left_hand, right_hand)
        
        # Draw overlay
        img = self.draw_gesture_overlay(img, left_hand, right_hand, gestures)
        
        return img, gestures

def main():
    """Main execution function"""
    print("🥋 TEKKEN - GESTURE CONTROL SYSTEM")
    print("🎯 LEFT HAND - MOVEMENT:")
    print("   🖐️  OPEN HAND    = Move RIGHT (D)")
    print("   ✊  CLOSED FIST   = Move LEFT (A)")
    print("   ☝️  1 FINGER     = Move UP (W)")
    print("   ✌️  2 FINGERS    = Move DOWN (S)")
    print("🎯 RIGHT HAND - ATTACKS:")
    print("   🖐️  OPEN HAND    = HEAVY Attack (U)")
    print("   ✊  CLOSED FIST   = LIGHT Attack (H)")
    print("   ☝️  1 FINGER     = MEDIUM Attack (J)")
    print("   ✌️  2 FINGERS    = SPECIAL Attack (K)")
    print("-" * 50)
    
    controller = TekkenGestureController()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    paused = False
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                break
            
            img = cv2.flip(img, 1)
            
            if not paused:
                img, gestures = controller.process_frame(img)
                if gestures:
                    controller.execute_actions(gestures)
            else:
                cv2.putText(img, "PAUSED", (img.shape[1]//2 - 50, img.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("TEKKEN Gesture Control", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # Space
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        controller.hands.close()
        print("✅ Cleanup completed")

if __name__ == "__main__":
    main()