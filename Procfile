web: PRELOAD_MODELS=false python -m flask bootstrap && gunicorn wsgi:app --workers 1 --threads 4 --timeout 180 --graceful-timeout 30 --access-logfile - --error-logfile -
