import cv2
import numpy as np
import time

# 全局变量存储点击的点
selected_points = []
frame_display = None  # 初始化用于显示的帧
PIXELS_PER_MM = None  # 像素与毫米的比例

def select_point(event, x, y, flags, param):
    """
    鼠标回调函数，用于选择特征点。
    允许选择两个点：原点和比例参考点。
    """
    global selected_points, frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 2:  # 限制最多选择2个点
            selected_points.append([x, y])
            color = (0, 255, 0) if len(selected_points) == 1 else (255, 0, 0)
            label = 'Origin' if len(selected_points) == 1 else 'Scale'
            cv2.circle(frame_display, (x, y), 5, color, -1)
            cv2.putText(frame_display, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
            cv2.imshow('Select Points', frame_display)

def main():
    global frame_display, PIXELS_PER_MM

    cam_index = 0  # 固定使用摄像头索引0，选择0或1
    print(f"尝试使用摄像头 (索引: {cam_index})")

    # =========================
    # 1. 初始化视频捕捉
    # =========================
    cap = cv2.VideoCapture(cam_index)  # 使用固定的摄像头索引0

    if not cap.isOpened():
        print(f"无法打开摄像头 (索引: {cam_index})")
        return

    # 设置分辨率（可根据需求调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 获取摄像头分辨率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"使用的摄像头分辨率: {frame_width}x{frame_height}")

    # =========================
    # 2. 读取并初始化摄像头（丢弃初始帧）
    # =========================
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧，可能摄像头出现问题。")
            cap.release()
            return
    print("已丢弃前10帧，摄像头初始化完成。")

    # 读取第一帧用于选择特征点
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取第一帧")
        cap.release()
        return

    # 转换为灰度图像
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    frame_display = first_frame.copy()

    # 设置窗口并绑定鼠标回调函数
    cv2.namedWindow('Select Points')
    cv2.setMouseCallback('Select Points', select_point)

    print("请在视频窗口中点击选择两个特征点：")
    print("1. 第一个点作为原点")
    print("2. 第二个点作为已知距离的参考点")

    # 显示初始帧并等待用户选择两个点
    while True:
        cv2.imshow('Select Points', frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if len(selected_points) == 2:
            break

    cv2.destroyWindow('Select Points')

    if len(selected_points) < 2:
        print("未选择足够的特征点，退出程序。")
        cap.release()
        return

    # 询问用户已知的实际距离（毫米）
    try:
        actual_distance_mm = float(input("请输入两个选定点之间的实际距离（毫米）："))
        if actual_distance_mm <= 0:
            print("实际距离必须为正数。")
            cap.release()
            return
    except ValueError:
        print("请输入有效的数字。")
        cap.release()
        return

    # 计算像素与毫米的比例
    p1, p2 = selected_points
    pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    PIXELS_PER_MM = pixel_distance / actual_distance_mm
    print(f"像素与毫米的比例: {PIXELS_PER_MM:.2f} 像素/mm")

    # 设置原点特征点（固定）
    origin_initial = np.array([p1], dtype=np.float32).reshape(-1, 1, 2)
    # 设置第二个特征点用于跟踪
    scale_initial = np.array([p2], dtype=np.float32).reshape(-1, 1, 2)

    # =========================
    # 3. 设置光流参数
    # =========================
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # =========================
    # 4. 初始化统计变量
    # =========================
    frame_count = 0
    start_time_total = time.perf_counter()

    # =========================
    # 5. 创建显示窗口
    # =========================
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

    # =========================
    # 6. 主循环
    # =========================
    p0 = scale_initial.copy()  # 初始点为第二个特征点

    while True:
        # 捕捉帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧，退出循环")
            break

        # 转换为灰度图
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流（从上一帧到当前帧）
        p1_optical, st, err = cv2.calcOpticalFlowPyrLK(first_gray, frame_gray, p0, None, **lk_params)

        if p1_optical is not None and st[0][0] == 1:
            new_position = p1_optical.reshape(-1, 1, 2)

            # 计算横向移动距离（毫米）基于初始原点
            displacement_x_pixels = new_position[0][0][0] - origin_initial[0][0][0]
            displacement_x_mm = displacement_x_pixels / PIXELS_PER_MM

            # 打印横向移动距离
            print(f"Frame {frame_count + 1}: 横向移动距离: {displacement_x_mm:.2f} mm")

            # 在图像上绘制原点和当前点
            origin_coords = tuple(origin_initial[0][0].astype(int))
            current_coords = tuple(new_position[0][0].astype(int))
            frame = cv2.circle(frame, origin_coords, 5, (0, 0, 255), -1)  # 红色圆圈表示原点
            frame = cv2.circle(frame, current_coords, 5, (0, 255, 0), -1)  # 绿色圆圈表示当前点
            frame = cv2.line(frame, origin_coords, current_coords, (255, 0, 0), 2)  # 蓝色线条表示位移

            # 更新光流跟踪点为当前点
            p0 = p1_optical.reshape(-1, 1, 2)

            # 更新第一帧为当前帧，以便下一次光流计算
            first_gray = frame_gray.copy()
        else:
            print(f"Frame {frame_count + 1}: 无法跟踪特征点。")
            break

        frame_count += 1

        # 显示图像
        cv2.imshow('Frame', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # =========================
    # 7. 打印统计信息
    # =========================
    end_time_total = time.perf_counter()
    total_duration = end_time_total - start_time_total
    print(f"\n=== 统计信息 ===")
    print(f"总帧数: {frame_count}")
    print(f"总处理时间: {total_duration:.2f} 秒")

    # =========================
    # 8. 清理资源
    # =========================
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
