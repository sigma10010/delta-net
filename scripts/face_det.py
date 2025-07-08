import cv2
import mediapipe as mp
import json
import os
import re

os.environ['GLOG_minloglevel'] = '2'  # 只显示 error
mp_face_mesh = mp.solutions.face_mesh

def extract_and_save_face_data(image_path, output_dir="/home/sigma/gaze/datasets/gc_mp/", save_json=True, return_rects=True):
    """
    从图像中提取人脸、眼睛裁剪图像，并将关键点和裁剪框保存为 JSON
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"图像无法读取：{image_path}")
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    match = re.search(r'/gc/(\d{5})/', image_path)
    if match:
        rec_id = match.group(1)
    filename = os.path.basename(image_path)
    file_id = os.path.splitext(filename)[0]

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, rec_id)
    os.makedirs(output_dir, exist_ok=True)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            print("❌ 未检测到人脸")
            return None

        landmarks = results.multi_face_landmarks[0] # normalized coords
        keypoints = [] # pixel coords
        for i, lm in enumerate(landmarks.landmark):
            keypoints.append({
                "id": i,
                "x": int(lm.x * w),
                "y": int(lm.y * h),
                "z": lm.z
            })

        # 计算人脸区域的边界
        xs = [pt["x"] for pt in keypoints]
        ys = [pt["y"] for pt in keypoints]
        x1, y1 = max(min(xs) - 20, 0), max(min(ys) - 20, 0)
        x2, y2 = min(max(xs) + 20, w), min(max(ys) + 20, h)
        face_rect = [int(x1), int(y1), int(x2), int(y2)] # for face grid

        # 计算中心与正方形边长
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        half_size = max(x2 - x1, y2 - y1) // 2

        # 扩展为正方形并裁剪
        fx1 = max(center_x - half_size, 0)
        fx2 = min(center_x + half_size, w)
        fy1 = max(center_y - half_size, 0)
        fy2 = min(center_y + half_size, h)

        if fx2 <= fx1 or fy2 <= fy1:
            print(f"❌ face: 裁剪尺寸非法 [{fx1}, {fy1}, {fx2}, {fy2}]")
            return None
        face_crop = image[fy1:fy2, fx1:fx2]
        if face_crop.size == 0:
            print(f"❌ face: 裁剪结果为空")
            return None
        
        # 眼睛索引
        left_eye_ids = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                        246, 161, 160, 159, 158, 157, 173, 468, 469, 470, 471, 472]
        right_eye_ids = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                         398, 384, 385, 386, 387, 388, 466, 473, 474, 475, 476, 477]

        # 裁剪眼睛
        def crop_eye(eye_ids, name, margin_ratio=0.25):
            # 筛选有效关键点
            pts = [keypoints[i] for i in eye_ids if 0 <= i < len(keypoints)]
            if not pts:
                print(f"❌ {name}: 无有效关键点")
                return None

            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            width = x_max - x_min
            height = y_max - y_min
            margin_x = int(width * margin_ratio)
            margin_y = int(height * margin_ratio)

            # 中心 + 正方形裁剪区域（更统一眼部输入尺寸）
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            half_size = max(width, height) // 2
            half_size = int(half_size * (1 + margin_ratio))

            # 保证不越界
            ex1 = max(center_x - half_size, 0)
            ex2 = min(center_x + half_size, w)
            ey1 = max(center_y - half_size, 0)
            ey2 = min(center_y + half_size, h)

            if ex2 <= ex1 or ey2 <= ey1:
                print(f"❌ {name}: 裁剪尺寸非法 [{ex1}, {ey1}, {ex2}, {ey2}]")
                return None

            eye_crop = image[ey1:ey2, ex1:ex2]
            if eye_crop.size == 0:
                print(f"❌ {name}: 裁剪结果为空")
                return None

            save_path = os.path.join(output_dir, name)
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, f"{file_id}.jpg"), eye_crop)

            return [int(ex1), int(ey1), int(ex2), int(ey2)]

        left_eye_rect = crop_eye(left_eye_ids, "appleLeftEye")
        right_eye_rect = crop_eye(right_eye_ids, "appleRightEye")

        if any(x is None for x in [keypoints, face_rect, left_eye_rect, right_eye_rect]):
            print("❌ 检测数据缺失，跳过该图像")
            return None

        os.makedirs(os.path.join(output_dir, "appleFace"), exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "appleFace", f"{file_id}.jpg"), face_crop)

        # 保存关键点和框信息
        result = {
            "face_keypoints": keypoints,
            "face_rect": face_rect,
            "left_eye_rect": left_eye_rect,
            "right_eye_rect": right_eye_rect
        }

        if save_json:
            os.makedirs(os.path.join(output_dir, "face_data"), exist_ok=True)
            json_path = os.path.join(output_dir, "face_data", "%s.json"%file_id)
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            # print(f"✅ 已保存关键点和框信息: {json_path}")

        # print(f"✅ 图像裁剪保存至: {output_dir}")

        return result if return_rects else None

if __name__ == "__main__":        
    image_path = '/home/sigma/gaze/datasets/gc/00002/frames/00043.jpg'
    face_data = extract_and_save_face_data(image_path)