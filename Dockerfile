FROM python:3.8

COPY . /app
WORKDIR /app

RUN pip install wheels/torch-1.10.0+cpu-cp38-cp38-linux_x86_64.whl
RUN pip install -r requirements.txt

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "5000"]