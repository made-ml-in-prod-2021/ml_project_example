from boto3 import client

if __name__ == "__main__":
    s3 = client("s3")
    s3.upload_file("../models/model.pkl", "for-dvc", "model.pkl")
