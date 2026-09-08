# Recipe for the container Cloud Run will run.
# Each line is a step. Google follows them to build the image.

FROM python:3.12-slim

WORKDIR /app

# Requirements are copied and installed on their own, before the source
# code. Docker caches each step, so as long as requirements.txt has not
# changed, later deploys skip the install and finish much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run tells the container which port to listen on through the PORT
# variable. This default is only used when running the image locally.
ENV PORT=8080

# One worker, because the search is plain Python and extra workers would
# only fight over the single CPU. Parallel builds come from Cloud Run
# starting more containers, which is now safe because the app keeps no
# state. The timeout is generous so a long search is never cut off.
CMD exec gunicorn --workers 1 --threads 4 --timeout 300 --bind 0.0.0.0:$PORT app:app
