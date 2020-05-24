FROM python:3

WORKDIR /App
RUN pip install torch torchvision scikit-image scipy numpy kornia exifread pillow