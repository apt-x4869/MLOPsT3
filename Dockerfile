FROM centos:latest
RUN yum update -y
RUN yum install python3 python3-pip -y
RUN pip3 install keras
RUN pip3 install numpy
RUN pip3 install opencv-python
RUN pip3 install matplotlib
RUN pip3 install pillow
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install scipy
RUN yum install gcc gcc-c++ python36-devel -y
RUN pip3 install pystan
RUN pip3 install tensorflow
WORKDIR /project
CMD [ "/bin/bash" ]