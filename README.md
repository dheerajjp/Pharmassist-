gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:8000 app:app
find . -name "*.pyc" -delete