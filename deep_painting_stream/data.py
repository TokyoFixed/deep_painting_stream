# import pandas as pd
# from google.cloud import Storage

# def get_data_from_gcs(bucket_name, file_name):
#     """
#     Get data from GCS
#     """
#     storage_client = Storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(file_name)
#     data = blob.v
#     return data
