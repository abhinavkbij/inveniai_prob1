FROM python:3.11.5-bookworm

WORKDIR /app/sklearn

# COPY files
COPY ./nbPipelineCV.pkl /app/sklearn
COPY ./gbPipelineTfidf.pkl /app/sklearn/
COPY ./sklearn_model.py /app/sklearn/
COPY ./requirements.txt /app/sklearn/
COPY ./sklearn.db /app/sklearn/

# install dependencies
RUN pip install -r requirements.txt

CMD ["python", "sklearn_model.py"]