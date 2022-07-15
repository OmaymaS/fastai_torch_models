
FROM fastdotai/fastai@sha256:c36b43104474006d8f8cd2a65f740bfd505693c670644c1d2dbedb5a6fb2de8a
RUN pip install -U fire==0.4.0 pandas==1.3.5 google-cloud-pubsub==2.13.0 google-cloud-storage==1.35.0 gcsfs==2022.5.0  
WORKDIR /app
COPY train.py .
COPY test_model.py .
COPY gcp_utils.py .

ENTRYPOINT ["python", "train.py"]
