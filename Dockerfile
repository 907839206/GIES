FROM python:3.11.9
WORKDIR /root/GIES/
COPY bootstrap.sh /root/GIES/
COPY requirements.txt /root/GIES/requirements.txt
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian/|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|http://deb.debian.org/debian-security|https://mirrors.aliyun.com/debian-security/|g' /etc/apt/sources.list.d/debian.sources && \
    apt update && \
    apt upgrade -y && \
    apt install -y libgl1-mesa-glx
RUN pip install -r /root/GIES/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN chmod +x /root/GIES/bootstrap.sh
CMD ["/root/GIES/bootstrap.sh"]
