
FROM python:3.9-slim
RUN pip3 install kfp==2.4.0
RUN pip3 install google-cloud-aiplatform==1.34.0
RUN pip3 install google-cloud-aiplatform[autologging]
RUN pip3 install scikit-learn==1.0.2
RUN pip3 install google-cloud-pipeline-components==2.6.0
