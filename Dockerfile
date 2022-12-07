FROM python:3.8
# you can use alpine image for light weight.
# FROM python:3.9-alpine
# workdir is used to set the pwd inside docker container
WORKDIR /code
COPY requirements.txt /requirements.txt
# Install pip dependancy.
RUN pip install --no-cache-dir -r /requirements.txt
# copy whole directory inside /code working directory.
COPY . /code
# This command execute at the time when conatiner start.
CMD ["python3", "finetuning.py"]
