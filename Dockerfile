FROM python:3.8
ENV MY_DIR=/bo_pr
WORKDIR ${MY_DIR}
COPY . .
RUN pip install -e .

CMD bash