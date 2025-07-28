import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from hangeul_image_processor import process_hangeul_image_with_dynamic_interpolation_and_filter

# 클릭한 점들을 저장할 리스트 (획 구분)
strokes = []  # 여러 획 저장
current_stroke = []  # 현재 획 저장
filtered_points = []
image_display = None
is_dragging = False


def text_to_image(text, font_path, image_path, font_size=100, image_size=(1080, 740), text_color=(0, 0, 0)):
    image = Image.new('RGB', image_size, (255, 255, 255))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
    draw.text(position, text, fill=text_color, font=font)
    image.save(image_path)
    numpy_image = np.array(image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def draw_points_and_lines():
    global image_display, strokes, current_stroke

    temp_image = image_display.copy()

    # 모든 점 표시 (초록색)
    for pt in filtered_points:
        cv2.circle(temp_image, pt, 5, (0, 255, 0), -1)

        # 저장된 획 표시
    for stroke in strokes:
        for idx, pt in enumerate(stroke):
            color = (0, 255, 255) if idx == 0 else (255, 0, 0)  # 첫 번째 점은 노란색, 나머지는 빨간색
            cv2.circle(temp_image, pt, 5, color, -1)
            if idx > 0:
                cv2.line(temp_image, stroke[idx - 1], pt, (0, 0, 255), 2)

    # 현재 진행 중인 획 표시
    for idx, pt in enumerate(current_stroke):
        color = (0, 255, 255) if idx == 0 else (255, 0, 0)  # 첫 번째 점은 노란색, 나머지는 빨간색
        cv2.circle(temp_image, pt, 5, color, -1)
        if idx > 0:
            cv2.line(temp_image, current_stroke[idx - 1], pt, (0, 0, 255), 2)

    cv2.imshow('Connect Points', temp_image)


def find_nearest_point(x, y):
    if len(filtered_points) == 0:
        return None

    distances = [np.sqrt((px - x) ** 2 + (py - y) ** 2) for px, py in filtered_points]
    min_dist = min(distances)

    # 클릭한 좌표에서 일정 범위(10px) 이내에 있는 점만 허용
    if min_dist < 10:
        nearest_index = distances.index(min_dist)
        return filtered_points[nearest_index]
    return None


def mouse_callback(event, x, y, flags, param):
    global current_stroke, is_dragging, strokes

    if event == cv2.EVENT_LBUTTONDOWN:
        # 마우스 왼쪽 버튼을 눌렀을 때 -> 새로운 획 시작 가능
        is_dragging = True
        current_stroke = []  # 새로운 획으로 초기화

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging:
            nearest_point = find_nearest_point(x, y)
            if nearest_point and nearest_point not in current_stroke:
                current_stroke.append(nearest_point)
                print(f"Recorded point: {nearest_point}")
                draw_points_and_lines()

    elif event == cv2.EVENT_LBUTTONUP:
        # 마우스 왼쪽 버튼에서 손을 뗄 때 -> 현재 획 저장
        if len(current_stroke) > 1:  # 획이 최소 두 개의 점을 포함해야 저장
            strokes.append(current_stroke.copy())
            print(f"Saved stroke: {current_stroke}")
        current_stroke = []  # 현재 획 초기화
        is_dragging = False
        draw_points_and_lines()


def project_filtered_points_on_image(image, points):
    global image_display, filtered_points
    image_display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    filtered_points = points

    for pt in filtered_points:
        cv2.circle(image_display, pt, 5, (0, 255, 0), -1)  # 초록색 점 표시

    cv2.imshow('Connect Points', image_display)
    cv2.setMouseCallback('Connect Points', mouse_callback)


def save_connections_to_file(output_file):
    with open(output_file, "w") as file:
        for i, stroke in enumerate(strokes):
            file.write(f"Stroke {i + 1}:\n")
            for j, pt in enumerate(stroke):
                file.write(f"  {j + 1}: {pt}\n")
            file.write("\n")
    print(f"연결된 점 좌표가 '{output_file}'에 저장되었습니다.")


def main():
    font_path = "mm.ttf"
    text_to_image("가", font_path, "output_image.png", font_size=500)

    original_image = cv2.imread('output_image.png', cv2.IMREAD_GRAYSCALE)

    # 필터링된 점 좌표 가져오기
    filtered_points = process_hangeul_image_with_dynamic_interpolation_and_filter('output_image.png')

    # 필터링된 점을 GUI에 표시
    project_filtered_points_on_image(original_image, filtered_points)

    print("마우스를 드래그해서 점을 연결하세요. 'q'를 누르면 저장 후 종료됩니다.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            save_connections_to_file("connections.txt")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
