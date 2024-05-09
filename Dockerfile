FROM python:3.11.9
WORKDIR /root/GIES/
COPY debian/debian.sources /etc/apt/sources.list.d/debian.sources
COPY requirements.txt /root/GIES/requirements.txt
COPY apps /root/GIES/apps
COPY static /root/GIES/static
COPY app.py /root/GIES/app.py
COPY executor.py /root/GIES/executor.py
RUN apt update && \
    apt upgrade -y && \
    apt install -y libgl1-mesa-glx
RUN pip install -r /root/GIES/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
CMD ["python","/root/GIES/app.py"]