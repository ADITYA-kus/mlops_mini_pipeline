FROM python:3.13.3

WORKDIR /app 

COPY fastapi_app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY fastapi_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

COPY models/bow_model.pkl /app/models/bow_model.pkl

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



