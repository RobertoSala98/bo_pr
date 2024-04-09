FROM python:3.9.6
ENV MY_DIR=/bo_pr
WORKDIR ${MY_DIR}
COPY . .
RUN pip install -e .

CMD bash
