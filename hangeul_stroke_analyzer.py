import cv2
import numpy as np
from scipy.spatial import KDTree
from multiprocessing import Pool
import math
import random
from hangeul_image_processor import process_hangeul_image_with_dynamic_interpolation_and_filter
from PIL import Image, ImageDraw, ImageFont

def text_to_image(text, font_path, image_path, font_size=100, image_size=(1080, 720), text_color=(0, 0, 0)):
    # 1. 빈 이미지 생성 (배경은 흰색으로 설정)
    image = Image.new('RGB', image_size, (255, 255, 255))

    # 2. 폰트 설정ㄲㄹ
    font = ImageFont.truetype(font_path, font_size)

    # 3. 이미지에 텍스트 그리기
    draw = ImageDraw.Draw(image)

    # 4. 텍스트 크기를 계산하고 이미지 중앙에 배치
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    draw.text(position, text, fill=text_color, font=font)

    # 5. 이미지 저장
    image.save(image_path)
    print(f"이미지가 '{image_path}'에 저장되었습니다.")

    numpy_image= np.array(image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image

def calculate_slope_angle(pt1, pt2):
    delta_x = pt2[0] - pt1[0]
    delta_y = pt2[1] - pt1[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def interpolate_points(pt1, pt2, distance_threshold=1):
    dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
    num_points = max(2, int(dist // distance_threshold))
    x = np.linspace(pt1[0], pt2[0], num_points, dtype=int)
    y = np.linspace(pt1[1], pt2[1], num_points, dtype=int)
    return list(zip(x, y))

def filter_points(points, min_distance=5):
    if not points:
        return []
    tree = KDTree(points)
    unique_indices = tree.query_ball_point(points, min_distance)
    unique_points = [points[i] for i in range(len(points)) if i == unique_indices[i][0]]
    return unique_points

def is_line_valid(image, pt1, pt2):
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.line(mask, pt1, pt2, 255, 1)
    overlap = cv2.bitwise_and(image, image, mask=mask)
    return np.all(overlap[mask == 255] != 255)

def determine_stroke_direction(points):
    """
    점들의 집합에서 획의 주요 방향을 결정하는 함수
    수평(가로), 수직(세로), 대각선 방향을 판단
    """
    if len(points) < 2:
        return "unknown"
    
    # 첫 번째 점과 마지막 점의 방향 계산
    start_pt = points[0]
    end_pt = points[-1]
    
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    
    # 방향 판단 (각도 기준)
    angle = math.degrees(math.atan2(dy, dx))
    
    # 각도 범위에 따라 방향 결정 (더 엄격한 기준 적용)
    if -20 <= angle <= 20:
        return "horizontal"  # 가로 방향
    elif 70 <= angle <= 110 or -110 <= angle <= -70:
        return "vertical"    # 세로 방향
    else:
        return "diagonal"    # 대각선 방향

def find_stroke_start_points(points, image):
    """
    한글 획의 시작점을 찾는 함수
    - 상단 좌측에서 시작하는 점들을 우선적으로 선택
    - 한글의 기본 획 순서 규칙 적용
    """
    if not points:
        return []
    
    # 이미지의 중심점 계산
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # 점들을 상단에서 하단으로, 좌측에서 우측으로 정렬
    sorted_points = sorted(points, key=lambda p: (p[1] // 30) * width + p[0])
    
    # 시작점 후보 선택 (상단 좌측 영역의 점들)
    start_candidates = []
    for pt in sorted_points:
        # 상단 좌측 영역에 있는 점들을 시작점 후보로 선택
        if pt[1] < height * 0.4 and pt[0] < width * 0.6:
            start_candidates.append(pt)
    
    # 시작점 후보가 없으면 상단에서 가장 왼쪽에 있는 점 선택
    if not start_candidates:
        start_candidates = [min(points, key=lambda p: p[0])]
    
    return start_candidates

def merge_strokes(strokes, max_distance=50):
    if not strokes or len(strokes) <= 1:
        return strokes
    
    def calculate_angle(stroke):
        if len(stroke) < 2:
            return 0
        start = np.array(stroke[0])
        end = np.array(stroke[-1])
        return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(strokes):
            j = i + 1
            while j < len(strokes):
                # 두 획의 방향 확인
                angle1 = calculate_angle(strokes[i])
                angle2 = calculate_angle(strokes[j])
                angle_diff = abs(angle1 - angle2)
                
                # 방향이 크게 다른 획은 병합하지 않음 (45도 이상 차이나는 경우)
                if angle_diff > 45 and angle_diff < 315:
                    j += 1
                    continue
                
                # 두 획의 끝점 간 거리 계산
                dist1 = np.linalg.norm(np.array(strokes[i][-1]) - np.array(strokes[j][0]))
                dist2 = np.linalg.norm(np.array(strokes[i][-1]) - np.array(strokes[j][-1]))
                dist3 = np.linalg.norm(np.array(strokes[i][0]) - np.array(strokes[j][0]))
                dist4 = np.linalg.norm(np.array(strokes[i][0]) - np.array(strokes[j][-1]))
                
                min_dist = min(dist1, dist2, dist3, dist4)
                
                if min_dist < max_distance:
                    if min_dist == dist1:
                        strokes[i].extend(strokes[j])
                    elif min_dist == dist2:
                        strokes[i].extend(reversed(strokes[j]))
                    elif min_dist == dist3:
                        strokes[i] = list(reversed(strokes[i]))
                        strokes[i].extend(strokes[j])
                    else:
                        strokes[i] = list(reversed(strokes[i]))
                        strokes[i].extend(reversed(strokes[j]))
                    
                    strokes.pop(j)
                    merged = True
                    break
                j += 1
            if merged:
                break
            i += 1
    
    return strokes

def determine_stroke_order(points, image):
    """
    한글 글자의 획 순서를 결정하는 함수
    - 한글의 기본 획 순서 규칙 적용
    - 획의 방향성을 고려하여 점들을 연결
    """
    if not points:
        return []
    
    # 이미지의 중심점 계산
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # 시작점 찾기 (상단 좌측에서 시작)
    start_points = find_stroke_start_points(points, image)
    
    # 획 그룹화
    strokes = []
    used_points = set()
    
    # 각 시작점에서 획 시작
    for start_point in start_points:
        if start_point in used_points:
            continue
        
        # 새로운 획 시작
        current_stroke = [start_point]
        used_points.add(start_point)
        
        # 현재 점에서 가장 가까운 이웃 점들을 찾아 획 확장
        remaining_points = [p for p in points if p not in used_points]
        
        # 획의 방향 결정 (초기에는 가로 방향으로 가정)
        stroke_direction = "horizontal"
        
        while remaining_points:
            last_point = current_stroke[-1]
            
            # 마지막 점에서 가장 가까운 점 찾기
            distances = [(p, np.linalg.norm(np.array(last_point) - np.array(p))) for p in remaining_points]
            distances.sort(key=lambda x: x[1])
            
            # 가장 가까운 점이 너무 멀면 현재 획 종료
            if distances[0][1] > 80:  # 거리 임계값 증가
                break
            
            next_point = distances[0][0]
            
            # 획의 방향 확인 및 조정
            if len(current_stroke) >= 2:
                # 현재 획의 방향 계산
                current_direction = determine_stroke_direction(current_stroke[-3:])
                
                # 방향이 일관되게 유지되면 계속 진행
                if current_direction == stroke_direction or stroke_direction == "unknown":
                    stroke_direction = current_direction
                else:
                    # 방향이 크게 바뀌면 현재 획 종료
                    break
            
            current_stroke.append(next_point)
            used_points.add(next_point)
            remaining_points.remove(next_point)
        
        if len(current_stroke) > 1:  # 최소 2개 이상의 점이 있는 획만 추가
            strokes.append(current_stroke)
    
    # 남은 점들에 대해 새로운 획 시작
    remaining_points = [p for p in points if p not in used_points]
    while remaining_points:
        # 남은 점들 중에서 상단 좌측에 있는 점을 시작점으로 선택
        start_point = min(remaining_points, key=lambda p: (p[1] // 30) * width + p[0])
        
        # 새로운 획 시작
        current_stroke = [start_point]
        used_points.add(start_point)
        remaining_points.remove(start_point)
        
        # 현재 점에서 가장 가까운 이웃 점들을 찾아 획 확장
        while remaining_points:
            last_point = current_stroke[-1]
            
            # 마지막 점에서 가장 가까운 점 찾기
            distances = [(p, np.linalg.norm(np.array(last_point) - np.array(p))) for p in remaining_points]
            distances.sort(key=lambda x: x[1])
            
            # 가장 가까운 점이 너무 멀면 현재 획 종료
            if distances[0][1] > 80:  # 거리 임계값 증가
                break
            
            next_point = distances[0][0]
            current_stroke.append(next_point)
            used_points.add(next_point)
            remaining_points.remove(next_point)
        
        if len(current_stroke) > 1:  # 최소 2개 이상의 점이 있는 획만 추가
            strokes.append(current_stroke)
    
    # 획 병합 (가까운 획들을 하나로 합침)
    merged_strokes = merge_strokes(strokes)
    
    # 획 순서 최적화 (한글 획 순서 규칙에 따라)
    optimized_strokes = optimize_stroke_order(merged_strokes, image)
    
    return optimized_strokes

def optimize_stroke_order(strokes, image):
    if not strokes:
        return []
    
    height, width = image.shape[:2]
    
    def calculate_angle(stroke):
        if len(stroke) < 2:
            return 0
        start = np.array(stroke[0])
        end = np.array(stroke[-1])
        return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    
    # 각 획의 정보 계산
    stroke_info = []
    for i, stroke in enumerate(strokes):
        # 획의 중심점
        center_x = sum(pt[0] for pt in stroke) / len(stroke)
        center_y = sum(pt[1] for pt in stroke) / len(stroke)
        
        # 획의 방향과 각도
        direction = determine_stroke_direction(stroke)
        angle = calculate_angle(stroke)
        
        # 획의 길이
        length = sum(np.linalg.norm(np.array(stroke[j]) - np.array(stroke[j-1])) 
                    for j in range(1, len(stroke)))
        
        # 획의 시작점과 끝점
        start_pt = stroke[0]
        end_pt = stroke[-1]
        
        # 획의 위치 (상단, 중단, 하단)
        if center_y < height * 0.35:  # 상단 영역 축소
            position = "top"
        elif center_y > height * 0.65:  # 하단 영역 확대
            position = "bottom"
        else:
            position = "middle"
        
        # 획의 기울기 (-45도 ~ 45도는 가로, 45도 ~ 135도는 세로로 간주)
        if -45 <= angle <= 45 or angle <= -135 or angle >= 135:
            slope_type = "horizontal"
        else:
            slope_type = "vertical"
        
        stroke_info.append({
            'index': i,
            'stroke': stroke,
            'direction': direction,
            'center_x': center_x,
            'center_y': center_y,
            'start_pt': start_pt,
            'end_pt': end_pt,
            'length': length,
            'position': position,
            'angle': angle,
            'slope_type': slope_type
        })
    
    # 획 정렬 기준 설정
    def sort_key(s):
        pos_weight = {'top': 0, 'middle': 1, 'bottom': 2}
        slope_weight = {'vertical': 0, 'horizontal': 1}
        
        # 위치에 따른 가중치 (가장 큰 가중치)
        y_section = pos_weight[s['position']] * 10000
        
        # 같은 위치 내에서의 기울기 가중치
        slope_order = slope_weight[s['slope_type']] * 1000
        
        # 같은 위치, 같은 기울기 내에서 좌우 순서
        x_order = s['center_x']
        
        return y_section + slope_order + x_order
    
    # 획 정렬
    stroke_info.sort(key=sort_key)
    
    # 정렬된 획 반환
    return [s['stroke'] for s in stroke_info]

def connect_points_with_stroke_order(image, points):
    """
    획 순서를 반영하여 점들을 연결하는 함수
    """
    # 획 순서 결정
    strokes = determine_stroke_order(points, image)
    
    # 결과 이미지 생성
    connected_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 각 획을 다른 색상으로 표시
    colors = [
        (255, 0, 0),    # 빨간색
        (0, 255, 0),    # 초록색
        (0, 0, 255),    # 파란색
        (255, 255, 0),  # 청록색
        (255, 0, 255),  # 마젠타
        (0, 255, 255),  # 노란색
    ]
    
    # 각 획 그리기
    for i, stroke in enumerate(strokes):
        color = colors[i % len(colors)]
        
        # 획의 점들을 연결
        for j in range(len(stroke) - 1):
            cv2.line(connected_image, stroke[j], stroke[j+1], color, 2)
        
        # 각 점에 번호 표시 (첫 번째와 마지막 점만)
        if len(stroke) > 0:
            # 첫 번째 점에 획 번호 표시
            cv2.putText(connected_image, f"{i+1}", 
                        (stroke[0][0] - 10, stroke[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 마지막 점에 화살표 표시 (방향 표시)
            if len(stroke) > 1:
                last_pt = stroke[-1]
                prev_pt = stroke[-2]
                angle = math.degrees(math.atan2(last_pt[1] - prev_pt[1], last_pt[0] - prev_pt[0]))
                
                # 화살표 그리기
                arrow_length = 20
                arrow_angle = math.radians(30)
                
                # 화살표 끝점 계산
                end_x = last_pt[0] + arrow_length * math.cos(math.radians(angle))
                end_y = last_pt[1] + arrow_length * math.sin(math.radians(angle))
                
                # 화살표 날개 계산
                wing1_x = end_x - arrow_length * 0.5 * math.cos(math.radians(angle) + arrow_angle)
                wing1_y = end_y - arrow_length * 0.5 * math.sin(math.radians(angle) + arrow_angle)
                
                wing2_x = end_x - arrow_length * 0.5 * math.cos(math.radians(angle) - arrow_angle)
                wing2_y = end_y - arrow_length * 0.5 * math.sin(math.radians(angle) - arrow_angle)
                
                # 화살표 그리기
                cv2.line(connected_image, (int(end_x), int(end_y)), (int(wing1_x), int(wing1_y)), color, 2)
                cv2.line(connected_image, (int(end_x), int(end_y)), (int(wing2_x), int(wing2_y)), color, 2)
    
    return connected_image, strokes

def save_connections_to_file(connections, output_file):
    with open(output_file, "w") as file:
        for group_idx, group in enumerate(connections):
            file.write(f"Stroke {group_idx + 1}:\n")
            for point in group:
                file.write(f"  {point}\n")
            file.write("\n")

def process_point_group(args):
    image, points, max_angle_change = args
    return connect_points_with_stroke_order(image, points)

def project_filtered_points_on_image(image, points):
    projected_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for pt in points:
        cv2.circle(projected_image, pt, 3, (0, 255, 0), -1)  # 초록색 점
    cv2.imshow('Filtered Points Projection', projected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    font_path = "mm.ttf"  # 원하는 글씨체 파일 경로
    test_characters = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하"]
    
    for idx, char in enumerate(test_characters):
        try:
            print(f"\n테스트 글자: {char}")
            output_image = f"output_{idx+1}.png"  # 한글 대신 숫자로 파일명 생성
            text_to_image(char, font_path, output_image, font_size=500)

            original_image = cv2.imread(output_image, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                print(f"Error: Could not read image for character {char}")
                continue

            filtered_points = process_hangeul_image_with_dynamic_interpolation_and_filter(output_image)
            if not filtered_points:
                print(f"Error: No points found for character {char}")
                continue

            # Filtered points를 이미지에 투영
            project_filtered_points_on_image(original_image, filtered_points)

            # 획 순서를 반영하여 점 연결
            connected_image, strokes = connect_points_with_stroke_order(original_image, filtered_points)
            
            # 결과 저장
            output_txt_path = f"stroke_order_{idx+1}.txt"
            save_connections_to_file(strokes, output_txt_path)
            
            output_path_final = f"stroke_order_result_{idx+1}.png"
            cv2.imwrite(output_path_final, connected_image)
            
            # 결과 표시
            cv2.imshow(f'Stroke Order - {char}', connected_image)
            cv2.waitKey(1000)  # 1초 동안 표시
            
            print(f"획 순서가 '{output_txt_path}'에 저장되었습니다.")
            print(f"획 순서 이미지가 '{output_path_final}'에 저장되었습니다.")
            
        except Exception as e:
            print(f"Error processing character {char}: {str(e)}")
            continue
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()