FROM python:3.6.10

WORKDIR /usr/src/work

COPY requirements_docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./work_sample.py" ]
