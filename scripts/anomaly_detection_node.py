#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detect anomalies with PatchCore and map confirmed detections to the latest SLAM pose."""

import os
import time
import threading

import cv2
import numpy as np
import torch
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from anomalib.models import Patchcore

# -------------------- AYARLAR --------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.environ.get(
    "PATCHCORE_CHECKPOINT",
    os.path.join(PROJECT_ROOT, "models", "model.ckpt")
)

IMAGE_TOPIC = "/webcam/image_raw"
POSE_TOPIC = "/slam_out_pose"
HEIGHT_TOPIC = "/mavros/local_position/pose"
MARKER_TOPIC = "/anomaly_markers"
MAP_FRAME = "map"

IMAGE_SIZE = 256
SCORE_THRESHOLD = 0.98  # Kendi modelinin normal/anomali skorlarına göre ayarla.
PIXEL_PERCENTILE = 99.0  # Anomali haritasında box oluşturma eşiği.
MIN_BOX_AREA = 80        # Küçük gürültü bölgelerini elemek için minimum alan (piksel).
BOX_PADDING = 4          # Box çevresine eklenecek piksel payı.
MORPH_KERNEL_SIZE = 5    # Maske gürültüsünü azaltmak için tek sayı.
REQUIRED_CONSECUTIVE_FRAMES = 5  # Kaç ardışık işlenen karede anomali görülmeli?
POINT_REPLACE_DISTANCE = 2  # metre: bu mesafedeki en yakın eski nokta silinip yenisiyle değiştirilir.
POSE_TIMEOUT = 1.0       # Saniye: bu süreden eski SLAM pozu kullanılmaz.
HEIGHT_TRIGGER = 0.5     # Metre: bunun üzerine çıkıp tekrar altına inince tur tamamlanır.
MARKER_Z = 0.12
MARKER_DIAMETER = 0.30
DISPLAY_SCALE = 2.0      # Kamera penceresini kaç kat büyük göster (1.0 = orijinal boyut).
WINDOW_NAME = "PatchCore - SLAM Anomaly Map"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bridge = CvBridge()
model = None
marker_pub = None

latest_frame = None
frame_lock = threading.Lock()
pose_lock = threading.Lock()
vehicle_pose = None           # (x, y, z)
pose_received_wall_time = None
vehicle_height = None
height_received_wall_time = None

consecutive_anomaly_frames = 0
next_marker_id = 0
# Sadece haritada tutulacak aktif noktalar: her kayıt x/y/z/score/id içerir.
anomaly_points = []
last_height_above_trigger = False
height_cycle_started = False
green_marker_id = 100000
inference_fps = 0.0
last_inference_time = time.time()


def load_model():
    global model
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError("PatchCore checkpoint bulunamadı: " + CHECKPOINT)

    rospy.loginfo("PatchCore yükleniyor; device=%s", DEVICE)
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        rospy.logwarn("Checkpoint'te eksik anahtar sayısı: %d", len(missing))
    if unexpected:
        rospy.logwarn("Checkpoint'te beklenmeyen anahtar sayısı: %d", len(unexpected))
    model.to(DEVICE)
    model.eval()
    rospy.loginfo("PatchCore hazır.")


def image_callback(msg):
    global latest_frame
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except CvBridgeError as exc:
        rospy.logerr_throttle(2.0, "CvBridge görüntü hatası: %s", exc)
        return
    with frame_lock:
        latest_frame = frame


def pose_callback(msg):
    """Store the current Hector SLAM pose in the map frame."""
    global vehicle_pose, pose_received_wall_time
    # hector_mapping publishes geometry_msgs/PoseStamped on /slam_out_pose.
    pose = msg.pose
    p = pose.position
    with pose_lock:
        vehicle_pose = (float(p.x), float(p.y), float(p.z))
        pose_received_wall_time = time.time()


def height_callback(msg):
    """MAVROS local pose z değerini izler; 0.5 m üstüne çıkıp altına dönünce turu bitirir."""
    global vehicle_height, height_received_wall_time
    z = float(msg.pose.position.z)
    with pose_lock:
        vehicle_height = z
        height_received_wall_time = time.time()
    update_height_cycle(z)


def get_current_vehicle_pose():
    with pose_lock:
        p = vehicle_pose
        received = pose_received_wall_time
    if p is None or received is None:
        rospy.logwarn_throttle(3.0, "%s konumu bekleniyor.", POSE_TOPIC)
        return None
    age = time.time() - received
    if age > POSE_TIMEOUT:
        rospy.logwarn_throttle(3.0, "SLAM pozu eski (%.2f s); nokta kaydedilmiyor.", age)
        return None
    return p


def preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    return torch.from_numpy(tensor).unsqueeze(0).to(DEVICE)


def get_output_field(output, name):
    if hasattr(output, name):
        return getattr(output, name)
    if isinstance(output, dict):
        return output.get(name)
    return None


def get_score(output):
    score = get_output_field(output, "pred_score")
    if score is None:
        return 0.0
    if torch.is_tensor(score):
        score = score.detach().float().cpu().numpy()
    score = np.asarray(score).reshape(-1)
    return float(score[0]) if score.size else 0.0


def publish_anomaly_point(x, y, z, score):
    """Yeni noktayı ekler; yakınında eski nokta varsa eskisini silip yenisiyle değiştirir."""
    global next_marker_id, anomaly_points

    # Mesafe tabanlı birleştirme: POINT_REPLACE_DISTANCE içindeki en yakın eski noktayı sil.
    now = rospy.Time.now()
    nearby_index = None
    nearby_distance = float("inf")
    for i, old in enumerate(anomaly_points):
        distance = float(np.hypot(x - old["x"], y - old["y"]))
        if distance <= POINT_REPLACE_DISTANCE and distance < nearby_distance:
            nearby_index = i
            nearby_distance = distance

    if nearby_index is not None:
        old = anomaly_points.pop(nearby_index)
        deletes = MarkerArray()
        for namespace in ("confirmed_anomaly_locations",):
            marker = Marker()
            marker.header.frame_id = MAP_FRAME
            marker.header.stamp = now
            marker.ns = namespace
            marker.id = old["id"]
            marker.action = Marker.DELETE
            deletes.markers.append(marker)
        marker_pub.publish(deletes)
        rospy.loginfo("Yakın eski nokta silindi: id=%d, uzaklık=%.2f m", old["id"], nearby_distance)

    marker_id = next_marker_id
    next_marker_id += 1
    anomaly_points.append({"id": marker_id, "x": x, "y": y, "z": z, "score": score})

    markers = MarkerArray()
    point = Marker()
    point.header.frame_id = MAP_FRAME
    point.header.stamp = now
    point.ns = "confirmed_anomaly_locations"
    point.id = marker_id
    point.type = Marker.SPHERE
    point.action = Marker.ADD
    point.pose.position.x = x
    point.pose.position.y = y
    point.pose.position.z = max(0.0, z) + MARKER_Z
    point.pose.orientation.w = 1.0
    point.scale.x = point.scale.y = point.scale.z = MARKER_DIAMETER
    point.color.r, point.color.g, point.color.b, point.color.a = 1.0, 0.0, 0.0, 1.0
    point.lifetime = rospy.Duration(0)
    markers.markers.append(point)

    # Mevcut noktaları da tekrar yayınla; latch sayesinde RViz sonradan açılsa da görür.
    for old in anomaly_points:
        if old["id"] == marker_id:
            continue
        p = Marker()
        p.header.frame_id = MAP_FRAME
        p.header.stamp = now
        p.ns, p.id = "confirmed_anomaly_locations", old["id"]
        p.type, p.action = Marker.SPHERE, Marker.ADD
        p.pose.position.x, p.pose.position.y = old["x"], old["y"]
        p.pose.position.z = max(0.0, old["z"]) + MARKER_Z
        p.pose.orientation.w = 1.0
        p.scale.x = p.scale.y = p.scale.z = MARKER_DIAMETER
        p.color.r, p.color.g, p.color.b, p.color.a = 1.0, 0.0, 0.0, 1.0
        p.lifetime = rospy.Duration(0)
        markers.markers.append(p)

    marker_pub.publish(markers)
    rospy.logwarn("ANOMALİ NOKTASI: id=%d, map x=%.3f y=%.3f z=%.3f score=%.4f | toplam=%d",
                  marker_id, x, y, z, score, len(anomaly_points))


def publish_tunnel_average_point():
    """Tur bitince noktaları numara sırasına göre baştan ve sondan eşleştirir:
    1. ile son, 2. ile sondan bir önceki, 3. ile sondan iki önceki ... Her çiftin
    ortasına bir yeşil nokta koyar. Nokta sayısı tekse ortadaki nokta tek başına
    kalır ve kendi konumuna yeşil nokta konur."""
    global green_marker_id
    n = len(anomaly_points)
    if n == 0:
        rospy.logwarn("Tur tamamlandı ancak kayıtlı anomali noktası yok.")
        return

    now = rospy.Time.now()
    arr = MarkerArray()
    green_count = 0

    for i in range((n + 1) // 2):
        a = anomaly_points[i]
        b = anomaly_points[n - 1 - i]   # i == n-1-i ise (ortadaki nokta) a ve b aynı nokta olur
        avg_x = (a["x"] + b["x"]) / 2.0
        avg_y = (a["y"] + b["y"]) / 2.0
        avg_z = (a["z"] + b["z"]) / 2.0

        point = Marker()
        point.header.frame_id = MAP_FRAME
        point.header.stamp = now
        point.ns = "tunnel_average_point"
        point.id = green_marker_id
        point.type = Marker.SPHERE
        point.action = Marker.ADD
        point.pose.position.x = avg_x
        point.pose.position.y = avg_y
        point.pose.position.z = max(0.0, avg_z) + MARKER_Z
        point.pose.orientation.w = 1.0
        point.scale.x = point.scale.y = point.scale.z = MARKER_DIAMETER * 1.5
        point.color.r, point.color.g, point.color.b, point.color.a = 0.0, 1.0, 0.0, 1.0
        point.lifetime = rospy.Duration(0)
        arr.markers.append(point)

        rospy.logwarn("YEŞİL NOKTA id=%d: x=%.3f y=%.3f | kırmızı #%d (id=%d) ile #%d (id=%d) ortası",
                      green_marker_id, avg_x, avg_y, i + 1, a["id"], n - i, b["id"])
        green_marker_id += 1
        green_count += 1

    marker_pub.publish(arr)
    rospy.logwarn("Tur bitti: %d kırmızı nokta -> %d yeşil nokta.", n, green_count)


def update_height_cycle(z):
    """Araç 0.5 m üzerine çıkıp tekrar altına indiğinde yeşil noktaları üretir."""
    global last_height_above_trigger, height_cycle_started
    above = z > HEIGHT_TRIGGER
    if above:
        height_cycle_started = True
    elif height_cycle_started and last_height_above_trigger and z <= HEIGHT_TRIGGER:
        publish_tunnel_average_point()
        height_cycle_started = False
    last_height_above_trigger = above


def get_anomaly_map(output):
    """PatchCore çıktısından piksel bazlı anomali haritasını alır."""
    amap = get_output_field(output, "anomaly_map")
    if amap is None:
        return None
    if torch.is_tensor(amap):
        amap = amap.detach().float().cpu().numpy()
    else:
        amap = np.asarray(amap)

    if amap.ndim == 4:
        amap = amap[0, 0]
    elif amap.ndim == 3:
        amap = amap[0]
    elif amap.ndim != 2:
        return None

    return np.nan_to_num(amap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_anomaly_boxes(anomaly_map, frame_width, frame_height):
    """Anomali haritasından görüntü koordinatlarında bounding box'lar çıkarır."""
    if anomaly_map is None or anomaly_map.size == 0:
        return []

    amap = cv2.resize(anomaly_map, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
    threshold = float(np.percentile(amap, PIXEL_PERCENTILE))
    mask = (amap >= threshold).astype(np.uint8) * 255

    kernel_size = max(1, int(MORPH_KERNEL_SIZE))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < MIN_BOX_AREA or w < 3 or h < 3:
            continue
        pad = BOX_PADDING
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame_width - 1, x + w + pad), min(frame_height - 1, y + h + pad)
        region = amap[y:y + h, x:x + w]
        region_score = float(np.max(region)) if region.size else threshold
        boxes.append((x1, y1, x2, y2, region_score))

    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes[:20]


def process_frame(frame):
    global consecutive_anomaly_frames
    global inference_fps, last_inference_time

    with torch.inference_mode():
        output = model(preprocess(frame))
    score = get_score(output)
    anomaly_map = get_anomaly_map(output)
    anomaly_now = score >= SCORE_THRESHOLD
    h, w = frame.shape[:2]
    boxes = extract_anomaly_boxes(anomaly_map, w, h) if anomaly_now else []

    if anomaly_now:
        consecutive_anomaly_frames += 1
    else:
        consecutive_anomaly_frames = 0

    if consecutive_anomaly_frames >= REQUIRED_CONSECUTIVE_FRAMES:
        # Her N ardışık karede bir konum kaydet ve sayacı sıfırla.
        p = get_current_vehicle_pose()
        if p is not None:
            publish_anomaly_point(p[0], p[1], p[2], score)
        else:
            rospy.logwarn_throttle(2.0, "Anomali doğrulandı fakat güncel SLAM pozu yok; konum kaydedilmedi.")
        consecutive_anomaly_frames = 0

    now = time.time()
    dt = now - last_inference_time
    if dt > 0:
        current_fps = 1.0 / dt
        inference_fps = current_fps if inference_fps == 0 else 0.9 * inference_fps + 0.1 * current_fps
    last_inference_time = now

    # Sade ekran: sadece anomali boxları ve FPS değeri. Görüntü DISPLAY_SCALE kadar büyütülür.
    scale = DISPLAY_SCALE
    if scale != 1.0:
        display = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        display = frame.copy()
    label = "Anomali"
    for x1, y1, x2, y2, _ in boxes:
        bx1, by1 = int(x1 * scale), int(y1 * scale)
        bx2, by2 = int(x2 * scale), int(y2 * scale)
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 0, 255), 3)

        # Box üstüne kırmızı zemin üzerinde beyaz yazı; üstte yer yoksa box'ın içine yaz.
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ty1 = by1 - th - baseline - 6
        if ty1 < 0:
            ty1 = by1
        ty2 = ty1 + th + baseline + 6
        cv2.rectangle(display, (bx1, ty1), (bx1 + tw + 10, ty2), (0, 0, 255), -1)
        cv2.putText(display, label, (bx1 + 5, ty2 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(display, "FPS: %.1f" % inference_fps,
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return display


def main():
    global marker_pub
    rospy.init_node("patchcore_slam_anomaly_map", anonymous=False)
    global IMAGE_TOPIC, POSE_TOPIC, HEIGHT_TOPIC, MARKER_TOPIC, MAP_FRAME
    IMAGE_TOPIC = rospy.get_param("~image_topic", IMAGE_TOPIC)
    POSE_TOPIC = rospy.get_param("~slam_pose_topic", POSE_TOPIC)
    HEIGHT_TOPIC = rospy.get_param("~height_topic", HEIGHT_TOPIC)
    MARKER_TOPIC = rospy.get_param("~marker_topic", MARKER_TOPIC)
    MAP_FRAME = rospy.get_param("~map_frame", MAP_FRAME)
    if DEVICE.type == "cpu":
        rospy.logwarn("CUDA kullanılamıyor; PatchCore CPU üzerinde çalışacak.")

    marker_pub = rospy.Publisher(MARKER_TOPIC, MarkerArray, queue_size=10, latch=True)
    # Önceki sürümden RViz'de kalmış yazı ve işaretleri temizle.
    clear_markers = MarkerArray()
    clear_marker = Marker()
    clear_marker.action = Marker.DELETEALL
    clear_markers.markers.append(clear_marker)
    marker_pub.publish(clear_markers)
    rospy.sleep(0.2)
    load_model()
    rospy.Subscriber(IMAGE_TOPIC, Image, image_callback, queue_size=1, buff_size=2**24)
    rospy.Subscriber(POSE_TOPIC, PoseStamped, pose_callback, queue_size=5)
    rospy.Subscriber(HEIGHT_TOPIC, PoseStamped, height_callback, queue_size=5)

    # Yeniden boyutlandırılabilir pencere: istersen fareyle daha da büyütebilirsin.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    rate = rospy.Rate(30)
    rospy.loginfo("Görüntü: %s | SLAM x/y: %s | yükseklik: %s | Marker: %s",
                  IMAGE_TOPIC, POSE_TOPIC, HEIGHT_TOPIC, MARKER_TOPIC)
    rospy.loginfo("Yeşil noktalar için yükseklik %.2f m üzerine çıkıp tekrar altına inmeli.", HEIGHT_TRIGGER)
    rospy.loginfo("Anomali, skor %.4f veya üzerindeyken %d ardışık karede aranacak; "
                  "nokta değiştirme mesafesi=%.1f m; yükseklik eşiği=%.2f m.",
                  SCORE_THRESHOLD, REQUIRED_CONSECUTIVE_FRAMES, POINT_REPLACE_DISTANCE, HEIGHT_TRIGGER)
    try:
        while not rospy.is_shutdown():
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                rospy.loginfo_throttle(3.0, "Kamera görüntüsü bekleniyor: %s", IMAGE_TOPIC)
                rate.sleep()
                continue
            try:
                display = process_frame(frame)
                cv2.imshow(WINDOW_NAME, display)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    rospy.signal_shutdown("q tuşuna basıldı")
                    break
            except Exception as exc:
                rospy.logerr_throttle(2.0, "İşleme hatası: %s", exc)
            rate.sleep()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()