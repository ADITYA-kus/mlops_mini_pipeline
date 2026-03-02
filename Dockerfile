FROM python 3.13.3


WORKDIR /app 

COPY fastapi_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

RUN pip install --no-cache-dir -r requirements.txt

