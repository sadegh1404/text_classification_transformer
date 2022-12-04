FROM python:3.8

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "5000"]