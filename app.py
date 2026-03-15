from io import BytesIO
import base64
import json
from collections import deque
from threading import Lock
import time
import os
from datetime import datetime
from dotenv import load_dotenv
import threading

import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request
from twilio.rest import Client

# Monkeypatch torch.load before importing YOLO dependencies for PyTorch 2.6 compatibility
import torch
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

from emotion_engine import EmotionEngine
from danger_detector import DangerDetector

app = Flask(__name__)

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '')
RECIPIENT_PHONE_NUMBER = os.getenv('RECIPIENT_PHONE_NUMBER', '')
ENABLE_SMS_ALERTS = os.getenv('ENABLE_SMS_ALERTS', 'false').lower() == 'true'
TWILIO_STATUS_POLL_SECONDS = float(os.getenv('TWILIO_STATUS_POLL_SECONDS', '5'))

twilio_client = None
if ENABLE_SMS_ALERTS and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print("✓ Twilio SMS service initialized successfully")
    except Exception as e:
        print(f"⚠ Twilio initialization failed: {e}")
        twilio_client = None
else:
    if ENABLE_SMS_ALERTS:
        print("⚠ SMS alerts enabled but Twilio credentials not found in .env")

try:
    engine = EmotionEngine()
except Exception as error:
    print(f"Failed to initialize emotion engine: {error}")
    engine = None

try:
    danger_detector = DangerDetector('yolov8n.pt')
except Exception as error:
    print(f"Failed to initialize danger detector: {error}")
    danger_detector = None


TRACKED_EMOTIONS = ("Angry", "Fear", "Sad")
DEFAULT_LAT = 17.405677517091238
DEFAULT_LNG = 78.6207778
emotion_timestamps = {emotion: deque() for emotion in TRACKED_EMOTIONS}
last_terminal_print = {emotion: 0.0 for emotion in TRACKED_EMOTIONS}
last_sms_alert_sent = {emotion: 0.0 for emotion in TRACKED_EMOTIONS}
emotion_tracker_lock = Lock()

esp_status_lock = Lock()
ESP_STATUS_FILE = os.path.join(os.path.dirname(__file__), 'esp_status.json')
esp_status = {
    'lat': DEFAULT_LAT,
    'lng': DEFAULT_LNG,
    'attached': None,
    'updated_at': None
}


def attached_label(value):
    if value is True:
        return 'ATTACHED'
    if value is False:
        return 'REMOVED'
    return 'NO DATA'


def load_esp_status_from_disk():
    if not os.path.exists(ESP_STATUS_FILE):
        return

    try:
        with open(ESP_STATUS_FILE, 'r', encoding='utf-8') as file:
            saved = json.load(file)

        with esp_status_lock:
            esp_status['lat'] = float(saved.get('lat', DEFAULT_LAT))
            esp_status['lng'] = float(saved.get('lng', DEFAULT_LNG))
            attached_value = saved.get('attached')
            esp_status['attached'] = attached_value if isinstance(attached_value, bool) else None
            esp_status['updated_at'] = saved.get('updated_at')
    except Exception as error:
        print(f"⚠ Failed to load ESP status from disk: {error}", flush=True)


def save_esp_status_to_disk():
    try:
        with esp_status_lock:
            snapshot = dict(esp_status)
        with open(ESP_STATUS_FILE, 'w', encoding='utf-8') as file:
            json.dump(snapshot, file)
    except Exception as error:
        print(f"⚠ Failed to save ESP status to disk: {error}", flush=True)


def log_current_esp_status(prefix='ESP STATUS'):
    with esp_status_lock:
        current = dict(esp_status)

    lat, lng = normalize_esp_location(current.get('lat'), current.get('lng'))
    print(
        f"[{prefix}] attached={attached_label(current.get('attached'))} | "
        f"lat={lat:.6f}, lng={lng:.6f} | updated_at={current.get('updated_at')}",
        flush=True
    )


def parse_attached_value(payload):
    """Parse ESP attached status from common key/value formats."""
    attached_value = payload.get('attached', payload.get('isAttached', payload.get('status')))

    if isinstance(attached_value, bool):
        return attached_value

    if isinstance(attached_value, (int, float)):
        return bool(attached_value)

    if isinstance(attached_value, str):
        normalized = attached_value.strip().lower()
        truthy = {'true', '1', 'yes', 'high', 'attached', 'on'}
        falsy = {'false', '0', 'no', 'low', 'removed', 'detached', 'off'}
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False

    raise ValueError('Invalid or missing attached status')


def parse_esp_payload(req):
    """Parse ESP payload from JSON, raw body JSON, or form data."""
    payload = req.get_json(silent=True)
    if isinstance(payload, dict) and payload:
        return payload

    raw_body = req.get_data(cache=False, as_text=True)
    if raw_body:
        try:
            parsed_raw = json.loads(raw_body)
            if isinstance(parsed_raw, dict):
                return parsed_raw
        except Exception:
            pass

    if req.form:
        return req.form.to_dict(flat=True)

    return {}


def normalize_esp_location(lat, lng):
    """Return default coordinates when location is missing or zero."""
    try:
        lat_value = float(lat)
        lng_value = float(lng)
    except (TypeError, ValueError):
        return DEFAULT_LAT, DEFAULT_LNG

    if lat_value == 0.0 or lng_value == 0.0:
        return DEFAULT_LAT, DEFAULT_LNG

    return lat_value, lng_value


def send_sms_alert(alert_type, alert_message):
    """Send SMS alert via Twilio for detected emotion or dangerous object."""
    if not twilio_client or not RECIPIENT_PHONE_NUMBER:
        return False

    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message_body = f"🚨 {alert_type}: {alert_message} at {timestamp}"
        
        message = twilio_client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )

        print(
            f"✓ SMS request accepted by Twilio: {alert_message} "
            f"(SID: {message.sid}, initial_status: {message.status})",
            flush=True
        )

        def _log_delivery_status(message_sid):
            try:
                if TWILIO_STATUS_POLL_SECONDS > 0:
                    time.sleep(TWILIO_STATUS_POLL_SECONDS)

                latest = twilio_client.messages(message_sid).fetch()
                final_status = latest.status

                if final_status in {'delivered'}:
                    print(f"✓ SMS delivered (SID: {message_sid})", flush=True)
                elif final_status in {'failed', 'undelivered', 'canceled'}:
                    print(
                        f"✗ SMS not delivered (SID: {message_sid}, status: {final_status}, "
                        f"error_code: {latest.error_code}, error_message: {latest.error_message})",
                        flush=True
                    )
                else:
                    print(
                        f"ℹ SMS delivery pending (SID: {message_sid}, status: {final_status})",
                        flush=True
                    )
            except Exception as status_error:
                print(f"⚠ Unable to fetch SMS delivery status for {message_sid}: {status_error}", flush=True)

        threading.Thread(target=_log_delivery_status, args=(message.sid,), daemon=True).start()
        return True
    except Exception as e:
        print(f"✗ Failed to send SMS alert: {e}", flush=True)
        return False


def update_terminal_emotion_alerts(detections):
    now = time.time()
    current_second = int(now)
    min_second = current_second - 59

    detected_emotions = {
        detection.get("label")
        for detection in detections
        if detection.get("label") in TRACKED_EMOTIONS
    }

    with emotion_tracker_lock:
        for emotion in TRACKED_EMOTIONS:
            while emotion_timestamps[emotion] and emotion_timestamps[emotion][0] < min_second:
                emotion_timestamps[emotion].popleft()

        for emotion in detected_emotions:
            if not emotion_timestamps[emotion] or emotion_timestamps[emotion][-1] != current_second:
                emotion_timestamps[emotion].append(current_second)

        for emotion in TRACKED_EMOTIONS:
            detected_seconds_in_minute = len(emotion_timestamps[emotion])
            reached_threshold = detected_seconds_in_minute >= 10
            cooldown_passed = (now - last_terminal_print[emotion]) >= 1.0
            is_detected_now = emotion in detected_emotions

            if reached_threshold and cooldown_passed and is_detected_now:
                print(f"emotion detected:{emotion.lower()}", flush=True)
                last_terminal_print[emotion] = now
                
                sms_cooldown_passed = (now - last_sms_alert_sent[emotion]) >= 60.0
                if sms_cooldown_passed and ENABLE_SMS_ALERTS:
                    send_sms_alert("Emotion Alert", f"Person detected with {emotion} emotion")
                    last_sms_alert_sent[emotion] = now


load_esp_status_from_disk()
log_current_esp_status('ESP STARTUP STATUS')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/esp-status')
def esp_status_page():
    with esp_status_lock:
        current_status = dict(esp_status)
    current_status['lat'], current_status['lng'] = normalize_esp_location(
        current_status.get('lat'),
        current_status.get('lng')
    )
    return render_template('esp_status.html', status=current_status)


@app.route('/api/esp-status', methods=['GET'])
def get_esp_status():
    with esp_status_lock:
        current_status = dict(esp_status)
    current_status['lat'], current_status['lng'] = normalize_esp_location(
        current_status.get('lat'),
        current_status.get('lng')
    )
    return jsonify({'success': True, 'data': current_status})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200


@app.route('/esp-update', methods=['POST'])
def esp_update():
    payload = parse_esp_payload(request)

    print(f"[ESP RAW] payload={payload}", flush=True)

    try:
        lat = float(payload.get('lat'))
        lng = float(payload.get('lng'))
    except (TypeError, ValueError):
        print(f"[ESP ERROR] Invalid lat/lng in payload: {payload}", flush=True)
        return jsonify({'success': False, 'error': 'Invalid or missing lat/lng'}), 400

    lat, lng = normalize_esp_location(lat, lng)

    try:
        attached = parse_attached_value(payload)
    except ValueError:
        print(f"[ESP ERROR] Invalid attached status in payload: {payload}", flush=True)
        return jsonify({'success': False, 'error': 'Invalid or missing attached status'}), 400

    updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    should_send_device_removed_sms = False

    with esp_status_lock:
        previous_attached = esp_status.get('attached')
        esp_status['lat'] = lat
        esp_status['lng'] = lng
        esp_status['attached'] = attached
        esp_status['updated_at'] = updated_at

        # Trigger only on state transition to REMOVED to avoid alert spam.
        if attached is False and previous_attached is not False:
            should_send_device_removed_sms = True

    save_esp_status_to_disk()

    print(
        f"[ESP UPDATE] attached={attached_label(attached)} | "
        f"lat={lat:.6f}, lng={lng:.6f} | at {updated_at}",
        flush=True
    )

    if should_send_device_removed_sms and ENABLE_SMS_ALERTS:
        send_sms_alert(
            "DEVICE ALERT",
            f"Device removed (detached). Last location: {lat:.6f}, {lng:.6f}"
        )

    return jsonify({'success': True, 'message': 'ESP status updated', 'attached': attached})


@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        if engine is None:
            return jsonify({'error': 'Emotion engine is not loaded'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        img = Image.open(file.stream).convert('RGB')
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        frame_with_detections, results = engine.detect(frame)

        _, buffer = cv2.imencode('.jpg', frame_with_detections)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'detections': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-all', methods=['POST'])
def detect_all():
    """Combined endpoint for emotion + dangerous object detection."""
    try:
        data = request.get_json(silent=True) or {}
        include_image = bool(data.get('include_image', True))

        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Run emotion detection
        emotion_frame, emotion_results = (engine.detect(frame) if engine else (frame, []))
        update_terminal_emotion_alerts(emotion_results)

        # Run danger detection
        danger_frame, danger_results = (danger_detector.detect(emotion_frame) if danger_detector else (emotion_frame, []))

        # Check for dangerous objects and send alerts
        now = time.time()
        for danger in danger_results:
            obj_name = danger.get('object', 'Unknown')
            conf = danger.get('confidence', 0)
            print(f"🚨 DANGEROUS OBJECT DETECTED: {obj_name} (confidence: {conf})", flush=True)
            
            # Send SMS alert if enabled
            if ENABLE_SMS_ALERTS:
                send_sms_alert("CHILD SAFETY ALERT", f"Dangerous object detected: {obj_name}")

        response = {
            'success': True,
            'emotions': emotion_results,
            'emotion_count': len(emotion_results),
            'dangerous_objects': danger_results,
            'danger_count': len(danger_results)
        }

        if include_image:
            _, buffer = cv2.imencode('.jpg', danger_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response['image'] = f'data:image/jpeg;base64,{img_base64}'

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-base64', methods=['POST'])
def detect_base64():
    try:
        if engine is None:
            return jsonify({'error': 'Emotion engine is not loaded'}), 500

        data = request.get_json(silent=True) or {}
        include_image = bool(data.get('include_image', True))

        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        frame_with_detections, results = engine.detect(frame)
        update_terminal_emotion_alerts(results)

        response = {
            'success': True,
            'detections': results,
            'count': len(results)
        }

        if include_image:
            _, buffer = cv2.imencode('.jpg', frame_with_detections)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response['image'] = f'data:image/jpeg;base64,{img_base64}'

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-video', methods=['POST'])
def detect_video():
    try:
        if engine is None:
            return jsonify({'error': 'Emotion engine is not loaded'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        import tempfile
        import os
        
        video_input_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(video_input_path)
        
        cap = cv2.VideoCapture(video_input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = os.path.join(tempfile.gettempdir(), 'processed_' + file.filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        emotion_summary = {emotion: 0 for emotion in ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']}
        total_detections = 0
        frame_idx = 0
        
        print(f"Processing video: {file.filename} ({frame_count} frames)", flush=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            frame_with_detections, results = engine.detect(frame)
            
            if results:
                total_detections += len(results)
                for detection in results:
                    emotion = detection.get('label', '')
                    if emotion in emotion_summary:
                        emotion_summary[emotion] += 1
            
            out.write(frame_with_detections)
            
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames...", flush=True)
        
        cap.release()
        out.release()
        
        print(f"✓ Video processing complete: {file.filename}", flush=True)
        
        return jsonify({
            'success': True,
            'video_url': f'/static/processed_{file.filename}',
            'total_detections': total_detections,
            'summary': emotion_summary,
            'frames_processed': frame_idx
        })
    
    except Exception as e:
        print(f"Error processing video: {e}", flush=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
