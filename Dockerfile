FROM gcr.io/tfx-oss-public/tfx:1.14.0

COPY requirementsfinal.txt requirements.txt


# Download and install Python 3.10.2
#RUN wget https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz && \
#    tar -xzf Python-3.10.2.tgz && \
#    cd Python-3.10.2 && \
#    ./configure --enable-optimizations && \
#    make altinstall
RUN sed -i 's/python3/python/g' /usr/bin/pip
RUN pip install -r requirements.txt

COPY src/ src/

ENV PYTHONPATH="/pipeline:${PYTHONPATH}"