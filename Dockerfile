FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PORT=8501
EXPOSE 8501

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
