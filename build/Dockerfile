FROM gcr.io/tfx-oss-public/tfx:1.14.0

ENV RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT=1

COPY requirements.txt requirements.txt


RUN sed -i 's/python3/python/g' /usr/bin/pip
RUN pip install -r requirements.txt
RUN pip install typing_extensions==4.7.1