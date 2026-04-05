FROM python:3.12-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# install pytorch cpu version explicitly
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir matplotlib

COPY . .

RUN mkdir -p data models figures

CMD ["sh", "-c", "python spark_etl.py && python bayesian_forecasting.py && python visualize_results.py"]