import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import json
import os
import tkinter as tk
from tkinter import messagebox
import random
from hangeul_image_processor import process_hangeul_image_with_dynamic_interpolation_and_filter


CHOSEONG_LIST = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ',
                 'ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

def get_choseong(char):
    code = ord(char) - 0xAC00
    if code < 0 or code > 11171:
        return None
    return CHOSEONG_LIST[code // 588]

class HangeulApp:
    def __init__(self, root):
        self.root = root
        self.root.title("한글 획 데이터 수집 도구")
        self.root.minsize(700, 600)

        self.image_size = (500, 550)
        self.font_path = "mm.ttf"
        self.font_size = 500
        self.target_chars = self.load_and_filter_chars()

        self.img_dir = "img"
        os.makedirs(self.img_dir, exist_ok=True)

        self.completed_chars = set()
        self.current_char_index = 0
        self.current_stroke = []
        self.saved_strokes = {}
        self.progress_file = "progress.json"
        self.base_image = None
        self.filtered_points = []
        self.is_dragging = False

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(main_frame)
        self.image_label.pack(side=tk.LEFT, padx=10, pady=10)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        btn_style = {"padx": 5, "pady": 5, "fill": tk.X, "ipady": 6}

        tk.Button(control_frame, text="이전 글자", command=self.previous_character).pack(**btn_style)
        tk.Button(control_frame, text="다음 글자", command=self.next_character).pack(**btn_style)
        tk.Button(control_frame, text="초기화", command=self.reset_current_character).pack(**btn_style)
        tk.Button(control_frame, text="랜덤 글자", command=self.select_random_unprocessed_character).pack(**btn_style)
        tk.Button(control_frame, text="종료", command=self.on_exit).pack(**btn_style)

        self.dropdown_var = tk.StringVar()
        self.dropdown_menu = tk.OptionMenu(control_frame, self.dropdown_var, "로딩 중...")
        self.dropdown_menu.pack(**btn_style)

        self.status_label = tk.Label(control_frame, text="", width=20, font=("Arial", 12, "bold"))
        self.status_label.pack(pady=20)

        self.image_label.bind("<ButtonPress-1>", self.mouse_down)
        self.image_label.bind("<B1-Motion>", self.mouse_drag)
        self.image_label.bind("<ButtonRelease-1>", self.mouse_up)
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.load_progress()
        self.draw_current_character()
        self.update_choseong_dropdown()

    def get_unprocessed_chars(self):
        all_chars = set(self.target_chars)
        processed_chars = set(self.saved_strokes.keys())
        return list(all_chars - processed_chars)

    def select_random_unprocessed_character(self):
        unprocessed = self.get_unprocessed_chars()
        if not unprocessed:
            messagebox.showinfo("알림", "모든 글자가 완료되었습니다.")
            return
        random_char = random.choice(unprocessed)
        self.current_char_index = self.target_chars.index(random_char)
        self.draw_current_character()

    def get_choseong_stats(self):
        stats = {ch: {'done': 0, 'undone': 0} for ch in CHOSEONG_LIST}
        for ch in self.target_chars:
            cho = get_choseong(ch)
            if not cho:
                continue
            if ch in self.saved_strokes:
                stats[cho]['done'] += 1
            else:
                stats[cho]['undone'] += 1
        return stats

    def update_choseong_dropdown(self):
        stats = self.get_choseong_stats()
        items = [f"{ch} ({v['done']} / {v['undone']})" for ch, v in stats.items()]
        self.dropdown_var.set(items[0])
        self.dropdown_menu['menu'].delete(0, 'end')
        for item in items:
            self.dropdown_menu['menu'].add_command(label=item, command=tk._setit(self.dropdown_var, item))

    # 나머지 메서드는 기존 그대로 유지


    def load_and_filter_chars(self):
        all_hangul = [chr(0xAC00 + i) for i in range(11172)]
        valid_chars = []
        try:
            font = ImageFont.truetype(self.font_path, self.font_size)
        except IOError:
            return ['가', '나', '다']

        placeholder = Image.new("L", (self.font_size, self.font_size), 0)
        draw = ImageDraw.Draw(placeholder)
        draw.text((0, 0), '\uFFFF', font=font, fill=255)
        placeholder_array = np.array(placeholder)

        for char in all_hangul:
            img = Image.new("L", (self.font_size, self.font_size), 0)
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), char, font=font, fill=255)
            if not np.array_equal(np.array(img), placeholder_array):
                valid_chars.append(char)
        return valid_chars

    def generate_character_image(self, char):
        image = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(self.font_path, self.font_size)
        except IOError:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), char, font=font)
        x = (self.image_size[0] - (bbox[2] - bbox[0])) // 2
        y = (self.image_size[1] - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), char, font=font, fill="black")

        img_path = os.path.join(self.img_dir, f"U+{ord(char):04X}.png")
        image.save(img_path)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), img_path

    def compute_thicknesses(self, image, points):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        thicknesses = []
        for x, y in points:
            xi, yi = int(x), int(y)
            if 0 <= yi < dist_map.shape[0] and 0 <= xi < dist_map.shape[1]:
                thicknesses.append(round(float(dist_map[yi, xi] * 2), 2))
            else:
                thicknesses.append(0.0)
        return thicknesses

    def draw_points_and_lines(self):
        temp_image = self.base_image.copy()
        current_char = self.target_chars[self.current_char_index]

        for pt in self.filtered_points:
            cv2.circle(temp_image, pt, 4, (0, 255, 0), -1)

        for stroke in self.saved_strokes.get(current_char, {}).get("strokes", []):
            if len(stroke) >= 1:
                start_pt = tuple(self.filtered_points[stroke[0]])
                end_pt = tuple(self.filtered_points[stroke[-1]])
                cv2.circle(temp_image, start_pt, 6, (0, 0, 255), -1)
                cv2.circle(temp_image, end_pt, 6, (255, 0, 0), -1)

            for i in range(1, len(stroke)):
                pt1 = tuple(self.filtered_points[stroke[i - 1]])
                pt2 = tuple(self.filtered_points[stroke[i]])
                cv2.line(temp_image, pt1, pt2, (0, 0, 255), 2)

        for i in range(1, len(self.current_stroke)):
            cv2.line(temp_image, self.current_stroke[i - 1], self.current_stroke[i], (255, 0, 255), 2)
        for pt in self.current_stroke:
            cv2.circle(temp_image, pt, 5, (255, 0, 255), -1)

        self.update_image_display(temp_image)

    def draw_current_character(self):
        self.current_char = self.target_chars[self.current_char_index]
        self.current_stroke = []
        if self.current_char not in self.saved_strokes:
            self.saved_strokes[self.current_char] = {"strokes": [], "image_path": "", "points": [], "thicknesses": [],
                                                     "stroke_labels": []}
        self.status_label.config(text=f"{self.current_char} ({self.current_char_index + 1}/{len(self.target_chars)})")

        char_img, img_path = self.generate_character_image(self.current_char)
        self.saved_strokes[self.current_char]["image_path"] = img_path

        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        self.base_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.filtered_points = process_hangeul_image_with_dynamic_interpolation_and_filter(img_path)
        self.saved_strokes[self.current_char]["points"] = self.filtered_points.copy()
        self.saved_strokes[self.current_char]["thicknesses"] = self.compute_thicknesses(self.base_image,
                                                                                        self.filtered_points)
        self.draw_points_and_lines()

    def complete_stroke(self):
        if len(self.current_stroke) > 1:
            indices = [self.filtered_points.index(pt) for pt in self.current_stroke if pt in self.filtered_points]
            self.saved_strokes[self.current_char]["strokes"].append(indices)
        self.current_stroke = []

    def save_progress(self):
        progress = {
            "image_size": list(self.image_size),
            "font_path": self.font_path,
            "font_size": self.font_size,
            "completed": list(self.completed_chars),
            "current_index": self.current_char_index,
            "characters": {}
        }

        for char, data in self.saved_strokes.items():
            strokes = data.get("strokes", [])
            points = data.get("points", [])
            thicknesses = data.get("thicknesses", [])
            sequence = [i for stroke in strokes for i in stroke]

            # stroke_labels 생성
            stroke_labels = [-1] * len(points)
            for idx, stroke in enumerate(strokes):
                for i in stroke:
                    if 0 <= i < len(points):
                        stroke_labels[i] = idx

            char_record = {
                "image_path": data.get("image_path", ""),
                "points": points,
                "thicknesses": thicknesses,
                "strokes": strokes,
                "sequence": sequence,
                "stroke_labels": stroke_labels,
                "stroke_lengths": [len(s) for s in strokes],
                "num_strokes": len(strokes),
                "num_points": len(points)
            }
            progress["characters"][char] = char_record

        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def load_progress(self):
        if not os.path.exists(self.progress_file): return
        with open(self.progress_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.completed_chars = set(data.get("completed", []))
        self.current_char_index = data.get("current_index", 0)
        for char, info in data.get("characters", {}).items():
            self.saved_strokes[char] = {
                "image_path": info.get("image_path", ""),
                "strokes": info.get("strokes", []),
                "points": info.get("points", []),
                "thicknesses": info.get("thicknesses", []),
                "stroke_labels": info.get("stroke_labels", [])
            }

    def update_image_display(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil)
        self.image_label.img_tk = tk_img
        self.image_label.configure(image=tk_img)

    def find_nearest_point(self, x, y):
        if not self.filtered_points: return None
        distances = [np.hypot(px - x, py - y) for px, py in self.filtered_points]
        min_dist = min(distances)
        return self.filtered_points[distances.index(min_dist)] if min_dist < 20 else None

    def add_point_to_stroke(self, x, y):
        pt = self.find_nearest_point(x, y)
        if pt and pt not in self.current_stroke:
            self.current_stroke.append(pt)
            self.draw_points_and_lines()

    def mouse_down(self, event):
        self.is_dragging = True
        self.current_stroke = []
        self.add_point_to_stroke(event.x, event.y)

    def mouse_drag(self, event):
        if self.is_dragging:
            self.add_point_to_stroke(event.x, event.y)

    def mouse_up(self, event):
        self.is_dragging = False
        if self.current_stroke:
            self.complete_stroke()
        self.draw_points_and_lines()

    def next_character(self):
        if self.current_stroke:
            self.complete_stroke()
        self.save_progress()
        if self.current_char_index < len(self.target_chars) - 1:
            self.current_char_index += 1
            self.draw_current_character()
        else:
            messagebox.showinfo("확인", "모든 글자 완료!")

    def previous_character(self):
        self.save_progress()
        if self.current_char_index > 0:
            self.current_char_index -= 1
            self.draw_current_character()

    def reset_current_character(self):
        self.current_stroke = []
        self.saved_strokes[self.current_char] = {"strokes": [], "image_path": "", "points": [], "thicknesses": [],
                                                 "stroke_labels": []}
        self.draw_points_and_lines()

    def on_exit(self):
        if messagebox.askokcancel("종료", "저장하고 종료합니까?"):
            self.save_progress()
            self.root.quit()
            self.root.destroy()
if __name__ == "__main__":
    root = tk.Tk()
    app = HangeulApp(root)
    root.mainloop()
