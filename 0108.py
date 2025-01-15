import cv2
import numpy as np
import time

# 全局变量存储点击的点
selected_points = []
frame_display = None  # 初始化用于显示的帧

def select_point(event, x, y, flags, param):
    """
    鼠标回调函数，用于选择特征点。
    只允许选择一个点，并在选中点的位置绘制一个绿色圆圈。
    """
    global selected_points, frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 1:  # 限制最多选择1个点
            selected_points.append([x, y])
            cv2.circle(frame_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select Points', frame_display)

def main():
    global frame_display

    cam_index = 0  # 固定使用摄像头索引0
    print(f"尝试使用摄像头 (索引: {cam_index})")

    # =========================
    # 1. 初始化视频捕捉
    # =========================
    cap = cv2.VideoCapture(cam_index)  # 使用固定的摄像头索引0

    if not cap.isOpened():
        print(f"无法打开摄像头 (索引: {cam_index})")
        return

    # 设置较低的分辨率（可选，视需求而定）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 获取摄像头分辨率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"使用的摄像头分辨率: {frame_width}x{frame_height}")

    # =========================
    # 2. 读取并初始化摄像头（丢弃初始帧）
    # =========================
    # 读取并丢弃前10帧，以让摄像头完成初始化和曝光调整
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧，可能摄像头出现问题。")
            cap.release()
            return
    print("已丢弃前10帧，摄像头初始化完成。")

    # 读取第一帧用于选择特征点
    ret, old_frame = cap.read()
    if not ret:
        print("无法读取第一帧")
        cap.release()
        return

    # 转换为灰度图像
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_display = old_frame.copy()

    # 设置窗口并绑定鼠标回调函数
    cv2.namedWindow('Select Points')
    cv2.setMouseCallback('Select Points', select_point)

    print("请在目标物体上点击选择一个特征点，按 'q' 完成选择。")

    # 显示初始帧并等待用户选择点
    while True:
        cv2.imshow('Select Points', frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow('Select Points')

    if len(selected_points) == 0:
        print("未选择任何特征点，退出程序。")
        cap.release()
        return

    # 转换为适合光流跟踪的格式
    p0 = np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)

    # =========================
    # 3. 设置光流参数
    # =========================
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # =========================
    # 4. 初始化统计变量
    # =========================
    frame_count = 0
    total_times = {
        'capture_time': 0.0,       # 捕捉时间
        'convert_time': 0.0,       # 转换时间
        'optical_flow_time': 0.0,  # 光流计算时间
        'drawing_time': 0.0,       # 绘制时间
        'update_time': 0.0         # 更新时间
    }

    start_time_total = time.perf_counter()

    # 用于帧率计算
    fps = 0.0
    prev_time = time.time()

    # =========================
    # 5. 创建显示窗口
    # =========================
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

    # =========================
    # 6. 主循环
    # =========================
    while True:
        # Reception: 捕捉帧
        start_capture = time.perf_counter()  # 捕捉开始时间
        ret, frame = cap.read()               # 捕捉帧
        end_capture = time.perf_counter()
        capture_time = (end_capture - start_capture) * 1000  # 转换为毫秒
        total_times['capture_time'] += capture_time

        if not ret:
            print("无法读取帧，退出循环")
            break

        # Processing: 转换为灰度图
        start_convert = time.perf_counter()  # 转换开始时间
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        end_convert = time.perf_counter()
        convert_time = (end_convert - start_convert) * 1000  # 转换时间，毫秒
        total_times['convert_time'] += convert_time

        # Processing: 计算光流
        start_optical = time.perf_counter()  # 光流计算开始时间
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)  # 计算光流
        end_optical = time.perf_counter()
        optical_flow_time = (end_optical - start_optical) * 1000  # 光流计算时间，毫秒
        total_times['optical_flow_time'] += optical_flow_time

        # Feedback: 绘制跟踪点
        start_drawing = time.perf_counter()  # 绘制开始时间
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                displacement = np.sqrt((a - c)**2 + (b - d)**2)
                # 根据位移进行控制逻辑
                # 例如：print(f"Point {i} displacement: {displacement}")

                # 在图像上绘制跟踪点
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
        end_drawing = time.perf_counter()
        drawing_time = (end_drawing - start_drawing) * 1000  # 绘制时间，毫秒
        total_times['drawing_time'] += drawing_time

        # Feedback: 更新跟踪点
        start_update = time.perf_counter()  # 更新开始时间
        # 更新上一帧和上一点
        old_gray = frame_gray.copy()
        if p1 is not None:
            p0 = good_new.reshape(-1, 1, 2)
        else:
            # 如果没有点被跟踪，重新检测特征点
            p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)
        end_update = time.perf_counter()
        update_time = (end_update - start_update) * 1000  # 更新时间，毫秒
        total_times['update_time'] += update_time

        # Calculate frame rate: 计算帧率
        current_time = time.time()
        time_diff = current_time - prev_time
        if time_diff > 0:
            fps = 1.0 / time_diff
        prev_time = current_time

        # Annotate frame with resolution and FPS: 在帧上标注分辨率和帧率
        cv2.putText(frame, f"Resolution: {frame_width}x{frame_height}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Annotate frame with processing times: 在帧上标注处理时间
        cv2.putText(frame, f"Capture: {capture_time:.2f} ms", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Convert: {convert_time:.2f} ms", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"OpticalFlow: {optical_flow_time:.2f} ms", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Drawing: {drawing_time:.2f} ms", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Update: {update_time:.2f} ms", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display the frame: 显示图像
        cv2.imshow('Frame', frame)

        frame_count += 1
        # 打印每帧的详细时间
        print(f"Frame {frame_count}: Capture={capture_time:.2f} ms, Convert={convert_time:.2f} ms, "
              f"OpticalFlow={optical_flow_time:.2f} ms, Drawing={drawing_time:.2f} ms, Update={update_time:.2f} ms")

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # =========================
    # 7. 打印统计信息
    # =========================
    end_time_total = time.perf_counter()
    total_duration = end_time_total - start_time_total
    average_times = {k: v / frame_count for k, v in total_times.items()} if frame_count > 0 else {k: 0 for k in total_times}

    print(f"\n=== 统计信息 ===")
    print(f"总帧数: {frame_count}")
    print(f"总处理时间: {total_duration:.2f} 秒")
    print(f"平均每帧处理时间:")
    for step, avg_time in average_times.items():
        print(f"  {step}: {avg_time:.2f} ms")

    # =========================
    # 8. 清理资源
    # =========================
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
