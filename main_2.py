import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageTk
from hangeul_image_processor import process_hangeul_image_with_dynamic_interpolation_and_filter
import json
import os
import tkinter as tk
from tkinter import messagebox

class HangeulApp:
    def load_and_filter_chars(self):
        print("글꼴 파일을 확인하고 지원되는 문자 목록을 생성하는 중입니다... (시간이 걸릴 수 있습니다)")
        all_hangul = [chr(0xAC00 + i) for i in range(11172)]
        valid_chars = []
        try:
            font = ImageFont.truetype(self.font_path, self.font_size)
        except IOError:
            print(f"오류: '{self.font_path}' 폰트 파일을 찾을 수 없습니다.")
            return ['가', '나', '다'] # Fallback

        # 깨진 문자를 식별하기 위한 기준 이미지 생성 (보통 .notdef 글꼴 문자가 렌더링됨)
        placeholder_img = Image.new("L", (self.font_size, self.font_size), 0)
        draw = ImageDraw.Draw(placeholder_img)
        draw.text((0, 0), '\uFFFF', font=font, fill=255)
        placeholder_array = np.array(placeholder_img)

        for i, char in enumerate(all_hangul):
            if (i + 1) % 500 == 0:
                print(f"  ... {i+1}/{len(all_hangul)} 문자 확인 중 ...")
            
            char_img = Image.new("L", (self.font_size, self.font_size), 0)
            draw = ImageDraw.Draw(char_img)
            draw.text((0, 0), char, font=font, fill=255)
            char_array = np.array(char_img)

            # 문자 이미지가 기준 '깨진 문자' 이미지와 다를 경우에만 유효한 문자로 추가
            if not np.array_equal(char_array, placeholder_array):
                valid_chars.append(char)

        print(f"총 11,172개의 한글 문자 중 {len(valid_chars)}개를 사용할 수 있습니다.")

        return valid_chars

    def __init__(self, root):
        self.root = root
        self.root.title("한글 획 데이터 수집 도구")
        self.root.geometry("500x550")
        
        # --- 상태 변수 --- #
        self.image_size = (500, 550)
        self.font_path = "mm.ttf"
        self.font_size = 500
        self.target_chars = self.load_and_filter_chars()

        # 이미지 저장을 위한 폴더 생성
        self.img_dir = "img"
        os.makedirs(self.img_dir, exist_ok=True)

        self.completed_chars = set()
        self.current_char_index = 0
        self.strokes = []
        self.current_stroke = []
        self.saved_strokes = {}
        self.progress_file = "progress.json"
        self.base_image = None
        self.is_dragging = False

        # --- UI 구성 --- #
        # 컨트롤 프레임
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # 이미지 표시 라벨
        self.image_label = tk.Label(root)
        self.image_label.pack(side=tk.LEFT, padx=10, pady=10)

        # 버튼
        self.prev_button = tk.Button(control_frame, text="이전 글자", command=self.previous_character)
        self.prev_button.pack(pady=5, fill=tk.X)

        self.next_button = tk.Button(control_frame, text="다음 글자", command=self.next_character)
        self.next_button.pack(pady=5, fill=tk.X)

        self.reset_button = tk.Button(control_frame, text="초기화", command=self.reset_current_character)
        self.reset_button.pack(pady=5, fill=tk.X)

        self.exit_button = tk.Button(control_frame, text="종료", command=self.on_exit)
        self.exit_button.pack(pady=5, fill=tk.X)
        
        # 상태 표시 라벨
        self.status_label = tk.Label(control_frame, text="", width=20)
        self.status_label.pack(pady=10, fill=tk.X)

        # --- 이벤트 바인딩 --- #
        self.image_label.bind("<ButtonPress-1>", self.mouse_down)
        self.image_label.bind("<B1-Motion>", self.mouse_drag)
        self.image_label.bind("<ButtonRelease-1>", self.mouse_up)
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # --- 초기화 --- #
        self.load_progress()
        self.draw_current_character()

    def save_progress(self):
        progress = {
            "image_size": list(self.image_size),
            "font_path": self.font_path,
            "font_size": self.font_size,
            "completed": list(self.completed_chars),
            "current_index": self.current_char_index,
            "characters": {}
        }
        
        for char, char_data in self.saved_strokes.items():
            strokes = char_data.get("strokes", [])
            
            # Prepare the new format
            all_points = []
            sequence = []
            stroke_groups = []
            
            point_index = 0
            for stroke in strokes:
                stroke_indices = []
                for point in stroke:
                    all_points.append(list(point))
                    sequence.append(point_index)
                    stroke_indices.append(point_index)
                    point_index += 1
                stroke_groups.append(stroke_indices)
            
            progress["characters"][char] = {
                "image_path": char_data.get("image_path", ""),
                "points": all_points,
                "sequence": sequence,
                "strokes": stroke_groups
            }
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        print(f"✅ {self.target_chars[self.current_char_index]} 상태 저장 완료")

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    self.completed_chars = set(data.get("completed", []))
                    self.current_char_index = data.get("current_index", 0)
                    
                    # Initialize saved_strokes
                    self.saved_strokes = {}
                    
                    # Check for new format (characters)
                    if "characters" in data:
                        for char, char_data in data["characters"].items():
                            if "points" in char_data:  # New format
                                points = char_data["points"]
                                stroke_groups = char_data.get("strokes", [])
                                strokes = [[tuple(points[i]) for i in group] 
                                         for group in stroke_groups]
                                self.saved_strokes[char] = {
                                    "strokes": strokes,
                                    "image_path": char_data.get("image_path", "")
                                }
                    # Handle old format
                    elif "saved_strokes" in data:
                        for char, char_data in data["saved_strokes"].items():
                            if isinstance(char_data, dict):
                                strokes = char_data.get("strokes", [])
                                image_path = char_data.get("image_path", "")
                                
                                # Handle Korean filename conversion
                                if image_path and not os.path.exists(image_path):
                                    char_code = f"U+{ord(char):04X}"
                                    dir_name = os.path.dirname(image_path)
                                    new_filename = f"{char_code}.png"
                                    image_path = os.path.join(dir_name, new_filename)
                                    
                                    old_path = os.path.join(self.img_dir, f"{char}.png")
                                    if os.path.exists(old_path):
                                        import shutil
                                        shutil.copy2(old_path, image_path)
                            else:  # Oldest format
                                strokes = char_data
                                image_path = ""
                            
                            self.saved_strokes[char] = {
                                "strokes": [[tuple(p) for p in stroke] for stroke in strokes],
                                "image_path": image_path
                            }
                    
                    print("✅ 진행 상황을 불러왔습니다.")
            except Exception as e:
                print(f"❌ 진행 상황 불러오기 중 오류 발생: {e}")
                self.completed_chars = set()
                self.current_char_index = 0
                self.saved_strokes = {}

    def generate_character_image(self, char):
        image = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype(self.font_path, self.font_size)
        except IOError:
            font = ImageFont.load_default()
        
        # 텍스트 크기 계산 및 중앙 정렬
        text_bbox = draw.textbbox((0, 0), char, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (self.image_size[0] - text_width) // 2
        y = (self.image_size[1] - text_height) // 2 - text_bbox[1]
        
        # 검은색으로 텍스트 그리기
        draw.text((x, y), char, font=font, fill="black")
        
        # 이미지 파일로 저장 (유니코드 이스케이프를 사용한 파일명)
        char_code = f"U+{ord(char):04X}"  # 예: '가' -> 'U+AC00'
        img_filename = f"{char_code}.png"
        img_path = os.path.join(self.img_dir, img_filename)
        
        # 폴더가 없으면 생성
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        
        # 이미지 저장
        image.save(img_path)
        
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), img_path

    def update_image_display(self, cv_image):
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.image_label.img_tk = img_tk
        self.image_label.configure(image=img_tk)

    def draw_points_and_lines(self):
        if self.base_image is None: return
        temp_image = self.base_image.copy()

        for pt in self.filtered_points:
            cv2.circle(temp_image, pt, 5, (0, 255, 0), -1)
        for stroke in self.strokes:
            for idx, pt in enumerate(stroke):
                color = (0, 255, 255) if idx == 0 else (255, 0, 0)
                cv2.circle(temp_image, pt, 5, color, -1)
                if idx > 0: cv2.line(temp_image, stroke[idx - 1], pt, (0, 0, 255), 2)
        for idx, pt in enumerate(self.current_stroke):
            color = (255, 0, 255) if idx == 0 else (0, 255, 255)
            cv2.circle(temp_image, pt, 5, color, -1)
            if idx > 0: cv2.line(temp_image, self.current_stroke[idx - 1], pt, (0, 0, 255), 2)
    
    def draw_points_and_lines(self):
        temp_image = self.base_image.copy()
        current_char = self.target_chars[self.current_char_index]
        
        # Draw all saved strokes
        for stroke in self.saved_strokes.get(current_char, {}).get("strokes", []):
            if len(stroke) > 1:
                for i in range(1, len(stroke)):
                    cv2.line(temp_image, stroke[i-1], stroke[i], (0, 0, 255), 2)
        
        # Draw current stroke in progress
        for idx, pt in enumerate(self.current_stroke):
            color = (255, 0, 255) if idx == 0 else (0, 255, 255)
            cv2.circle(temp_image, pt, 5, color, -1)
            if idx > 0: 
                cv2.line(temp_image, self.current_stroke[idx-1], pt, (255, 0, 255), 2)
        
        self.update_image_display(temp_image)

    def draw_current_character(self):
        self.current_char = self.target_chars[self.current_char_index]
        
        # Clear the current stroke
        self.current_stroke = []
        
        # Initialize character data if it doesn't exist
        if self.current_char not in self.saved_strokes:
            self.saved_strokes[self.current_char] = {"strokes": []}
        
        # Update status label
        self.status_label.config(text=f"현재 문자: {self.current_char} ({self.current_char_index + 1}/{len(self.target_chars)})")
        
        # Generate and display character image
        char_img, img_path = self.generate_character_image(self.current_char)
        
        # Save the image path
        self.saved_strokes[self.current_char]["image_path"] = img_path
        
        # Update the base image for drawing
        gray_image = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        self.base_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        # Process the image to get stroke points
        self.filtered_points = process_hangeul_image_with_dynamic_interpolation_and_filter(img_path)
        
        # Update the image label
        self.char_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.base_image, cv2.COLOR_BGR2RGB)))
        self.image_label.config(image=self.char_photo)
        
        # Redraw any existing strokes
        self.draw_points_and_lines()

    def mouse_up(self, event):
        if self.current_stroke:
            self.complete_stroke()
            self.redraw_canvas()

    def complete_stroke(self):
        if len(self.current_stroke) > 1:  # Only save strokes with at least 2 points
            current_char = self.target_chars[self.current_char_index]
            if current_char not in self.saved_strokes:
                self.saved_strokes[current_char] = {"strokes": []}
            self.saved_strokes[current_char]["strokes"].append(self.current_stroke.copy())
        self.current_stroke = []

    def next_character(self, event=None):
        # Complete any active stroke
        if self.current_stroke:
            self.complete_stroke()
        
        # Save progress before moving to next character
        self.save_progress()
        
        if self.current_char_index < len(self.target_chars) - 1:
            self.current_char_index += 1
            self.draw_current_character()
        else:
            messagebox.showinfo("완료", "모든 글자 작업을 완료했습니다!")

    def previous_character(self):
        if self.current_char_index > 0:
            self.save_progress()
            self.current_char_index -= 1
            self.draw_current_character()

    def reset_current_character(self):
        self.strokes = []
        self.current_stroke = []
        self.saved_strokes[self.target_chars[self.current_char_index]] = {"strokes": [], "num_strokes": 0, "image_path": ""}
{{ ... }}

    def find_nearest_point(self, x, y):
        if not self.filtered_points: return None
        distances = [np.sqrt((px - x) ** 2 + (py - y) ** 2) for px, py in self.filtered_points]
        min_dist_val = min(distances)
        if min_dist_val < 20: # 클릭 반경 증가
            return self.filtered_points[distances.index(min_dist_val)]
        return None

    def mouse_down(self, event):
        self.is_dragging = True
        self.current_stroke = []
        self.add_point_to_stroke(event.x, event.y)

    def mouse_drag(self, event):
        if self.is_dragging:
            self.add_point_to_stroke(event.x, event.y)

    def mouse_up(self, event):
        self.is_dragging = False
        if len(self.current_stroke) > 1:
            self.strokes.append(self.current_stroke.copy())
        self.current_stroke = []
        self.save_progress()
        self.draw_points_and_lines() # 최종 상태 업데이트

    def add_point_to_stroke(self, x, y):
        nearest_point = self.find_nearest_point(x, y)
        if nearest_point and nearest_point not in self.current_stroke:
            self.current_stroke.append(nearest_point)
            self.draw_points_and_lines()

    def on_exit(self):
        if messagebox.askokcancel("종료", "작업을 저장하고 종료하시겠습니까?"):
            self.save_progress()
            self.root.quit()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HangeulApp(root)
    root.mainloop()
