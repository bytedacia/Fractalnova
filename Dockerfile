FROM python:3.9-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD python app.py 