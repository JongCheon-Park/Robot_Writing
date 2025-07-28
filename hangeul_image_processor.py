import cv2
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def process_hangeul_image_with_dynamic_interpolation_and_filter(image_path):
    """
    한글 이미지를 처리하여 추출한 중심선 궤적을 원본 이미지에 점으로 투영하며,
    두 점 사이의 거리에 따라 보간 개수를 동적으로 조정하고,
    일정 범위 내에 있는 중복된 점들을 제거하는 함수.
    """
    # 1. 이미지 읽기 (원본 이미지 저장)
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 2. 이진화 (배경과 글씨 분리)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    # 3. 중심선 추출 (OpenCV thinning 대체 방식 사용)
    skeleton_image = cv2.ximgproc.thinning(binary_image)  # 중심선 추출

    # 4. 중심선을 따라 점을 찍을 준비
    contours, _ = cv2.findContours(skeleton_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 보간을 통한 점 추가 (동적 보간)
    def interpolate_points(pt1, pt2, distance_threshold=1):
        """
        pt1과 pt2 사이의 거리에 따라 점을 보간하여 더 많은 점을 추가하는 함수.
        거리가 멀수록 더 많은 점을 추가.
        """
        dist = np.linalg.norm(np.array(pt2) - np.array(pt1))  # 두 점 사이의 거리 계산
        num_points = max(2, int(dist // distance_threshold))  # 거리에 따라 보간할 점 개수 결정
        points = []
        for i in range(num_points):
            x = int(pt1[0] + (pt2[0] - pt1[0]) * i / (num_points - 1))
            y = int(pt1[1] + (pt2[1] - pt1[1]) * i / (num_points - 1))
            points.append((x, y))
        return points

    # 6. 일정 범위 내의 점을 제거하기 위한 필터 함수
    def filter_points(points, min_distance=5):
        """
        주어진 점 목록에서 일정 거리 내에 있는 점들을 필터링하여 제거.
        """
        filtered_points = []
        for pt in points:
            if all(np.linalg.norm(np.array(pt) - np.array(fp)) > min_distance for fp in filtered_points):
                filtered_points.append(pt)
        return filtered_points

    all_points = []

    # 7. 원본 이미지에 중심선 궤적을 빨간색 점으로 표시
    for contour in contours:
        for i in range(1, len(contour)):
            pt1 = tuple(contour[i - 1][0])
            pt2 = tuple(contour[i][0])
            # 두 점 사이에 동적 보간하여 점 추가
            interpolated_points = interpolate_points(pt1, pt2)
            all_points.extend(interpolated_points)

    # 8. 점 필터링
    filtered_points = filter_points(all_points, min_distance=10)

    pt_before = 0
    pt_after = 0
    # 9. 필터링된 점들만 그리기
    for pt in filtered_points:
        pt_after = pt
        if pt_before != 0 :
            distance = np.linalg.norm(np.asarray(pt_after) - np.asarray(pt_before))
            #if distance <= 50 :
            #    cv2.line(original_image, pt_before, pt_after, (0, 255, 0), 1)
        cv2.circle(original_image, pt, 2, (0, 0, 255), -1)  # 작은 점으로 표시
        pt_before = pt


    return filtered_points
