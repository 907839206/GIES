FROM python:3.11.9
WORKDIR /root/GIES/
COPY bootstrap.sh /root/GIES/
COPY requirements.txt /root/GIES/requirements.txt
COPY debian/debian.sources /etc/apt/sources.list.d/debian.sources
RUN apt update && \
    apt upgrade -y && \
    apt install -y libgl1-mesa-glx
RUN pip install -r /root/GIES/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN chmod +x /root/GIES/bootstrap.sh
CMD ["/root/GIES/bootstrap.sh"]
