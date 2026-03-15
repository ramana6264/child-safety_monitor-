# Facial Expression Recognition

Tutorial Link: https://www.youtube.com/watch?v=a573pDNNFEY

Dataset Link: https://www.kaggle.com/datasets/msambare/fer2013

## Quick Deploy (Docker)

This project includes TensorFlow + OpenCV, so Docker is the most reliable deployment path.

### 1) Build image

```bash
docker build -t fer-app .
```

### 2) Run container

```bash
docker run --rm -p 5000:5000 --env PORT=5000 fer-app
```

### 3) Open app

```text
http://127.0.0.1:5000
```

Health check endpoint:

```text
http://127.0.0.1:5000/health
```

## Cloud Deploy (Render/Railway/Fly.io)

1. Push this repo to GitHub.
2. Create a new Web Service from the repo.
3. Choose Docker deployment (auto-detect using `Dockerfile`).
4. Set environment variables in your platform dashboard:

```text
PORT=5000
FLASK_DEBUG=false
ENABLE_SMS_ALERTS=false
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
RECIPIENT_PHONE_NUMBER=
TWILIO_STATUS_POLL_SECONDS=5
```

5. Deploy and verify `/health` returns `{"status":"ok"}`.

## Run Without Docker (Local)

1. Create and activate your virtual environment.
2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

3. Start Flask backend:

	```bash
	python app.py
	```

4. Open in browser:

	```text
	http://127.0.0.1:5000
	```

The frontend is in `templates/index.html` and the backend API is in `app.py`.

## Run Desktop Webcam Script

```bash
python Test.py
```

Press `q` to quit webcam mode.
