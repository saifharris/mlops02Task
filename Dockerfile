FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

COPY model/model.pkl /app/model/model.pkl

ENV NAME MLApp

CMD ["python", "app.py"]
