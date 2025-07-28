import json
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ✅ 한글 폰트 설정 (Windows)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

class HangeulVisualizer:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.characters = list(self.data['characters'].keys())
        self.index = 0

        self.fig, self.ax = plt.subplots(figsize=(6, 6.5))
        self.fig.subplots_adjust(bottom=0.2)

        axprev = plt.axes([0.2, 0.05, 0.15, 0.075])
        axnext = plt.axes([0.65, 0.05, 0.15, 0.075])
        self.bnext = Button(axnext, '→')
        self.bprev = Button(axprev, '←')
        self.bnext.on_clicked(self.next_char)
        self.bprev.on_clicked(self.prev_char)

        self.fig.canvas.mpl_connect('key_press_event', self.key_event)
        self.draw_character()

        plt.show()

    def draw_character(self):
        self.ax.clear()
        char = self.characters[self.index]
        char_data = self.data['characters'][char]

        img_path = char_data['image_path'].replace("\\", "/")
        points = char_data.get('points', [])
        thicknesses = char_data.get('thicknesses', [])
        strokes = char_data.get('strokes', [])

        if not os.path.exists(img_path):
            self.ax.set_title(f"[{char}] 이미지 없음: {img_path}")
            return

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax.imshow(img)
        self.ax.axis('off')

        # ▶ Draw thickness-scaled points
        for (x, y), t in zip(points, thicknesses):
            self.ax.scatter(x, y, s=t**1.8, c='orange', edgecolors='black', linewidths=0.6, alpha=0.75)

        # ▶ Draw strokes with numbering
        for stroke_idx, stroke in enumerate(strokes):
            stroke_pts = [points[i] for i in stroke if 0 <= i < len(points)]
            if len(stroke_pts) < 2:
                continue  # skip invalid or too short strokes

            xs, ys = zip(*stroke_pts)
            self.ax.plot(xs, ys, color='red', linewidth=2)
            self.ax.scatter(xs[0], ys[0], s=40, c='red')     # 시작점
            self.ax.scatter(xs[-1], ys[-1], s=40, c='blue')  # 끝점

            # 획 번호 표시
            self.ax.text(xs[0] + 5, ys[0] - 5, str(stroke_idx + 1),
                         fontsize=12, color='black', weight='bold',
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        self.ax.set_title(f"{char} ({self.index + 1}/{len(self.characters)})", fontsize=14)
        self.fig.canvas.draw()

    def next_char(self, event=None):
        self.index = (self.index + 1) % len(self.characters)
        self.draw_character()

    def prev_char(self, event=None):
        self.index = (self.index - 1) % len(self.characters)
        self.draw_character()

    def key_event(self, event):
        if event.key == 'left':
            self.prev_char()
        elif event.key == 'right':
            self.next_char()
        elif event.key == 's':
            self.save_current_plot()

    def save_current_plot(self):
        char = self.characters[self.index]
        filename = f"plot_{char}_idx{self.index+1}.png"
        self.fig.savefig(filename, dpi=300)
        print(f"✅ 저장 완료: {filename}")

# 실행
if __name__ == "__main__":
    HangeulVisualizer("progress.json")
