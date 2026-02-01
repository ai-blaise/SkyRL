# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import os
from urllib.parse import urlparse

from enum import Enum
import torch
import torch.distributed

import io


class Cloud(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


def uploadDirectoryToS3(path, bucketname, prefix):
    import boto3

    s3 = boto3.client("s3")
    for root, dirs, files in os.walk(path):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), path)
            s3_key = os.path.join(prefix, relative_path)
            s3.upload_file(os.path.join(root, file), bucketname, s3_key)


def upload_file_to_s3(path, bucketname, prefix):
    import boto3

    s3 = boto3.client("s3")
    s3.upload_file(path, bucketname, prefix)


def upload_dir_to_anyscale(local_path, remote_key):
    save_bucket, remote_prefix, cloud = _get_anyscale_bucket_and_file_key(remote_key)
    if cloud == Cloud.AWS:
        uploadDirectoryToS3(local_path, save_bucket, remote_prefix)
    elif cloud == Cloud.GCP:
        upload_directory_to_gcs(local_path, save_bucket, remote_prefix)
    elif cloud == Cloud.AZURE:
        upload_directory_to_azure(local_path, save_bucket, remote_prefix)
    else:
        raise NotImplementedError(f"Unsupported cloud provider: {cloud}. Supported: AWS, GCP, Azure")


def upload_file_to_anyscale(local_path, remote_key):
    save_bucket, remote_prefix, cloud = _get_anyscale_bucket_and_file_key(remote_key)
    if cloud == Cloud.AWS:
        upload_file_to_s3(local_path, save_bucket, remote_prefix)
    elif cloud == Cloud.GCP:
        upload_file_to_gcs(local_path, save_bucket, remote_prefix)
    elif cloud == Cloud.AZURE:
        upload_file_to_azure(local_path, save_bucket, remote_prefix)
    else:
        raise NotImplementedError(f"Unsupported cloud provider: {cloud}. Supported: AWS, GCP, Azure")


def _get_anyscale_bucket_and_file_key(path):
    parsed_url = urlparse(os.environ["ANYSCALE_ARTIFACT_STORAGE"])
    if parsed_url.scheme == "s3":
        cloud = Cloud.AWS
    elif parsed_url.scheme in ("gs", "gcs"):
        cloud = Cloud.GCP
    elif parsed_url.scheme in ("az", "azure", "wasb", "wasbs", "abfs", "abfss"):
        cloud = Cloud.AZURE
    elif parsed_url.netloc.endswith(".blob.core.windows.net"):
        # Handle https://account.blob.core.windows.net/container format
        cloud = Cloud.AZURE
    else:
        cloud = Cloud.GCP  # Default to GCP for backwards compatibility
    save_bucket, prefix = parsed_url.netloc, parsed_url.path
    prefix = prefix.lstrip("/")
    save_bucket = save_bucket.rstrip("/")
    path = os.path.join(prefix, path)
    return save_bucket, path, cloud


def write_to_s3(obj, path: str):
    import boto3

    save_bucket, path, _ = _get_anyscale_bucket_and_file_key(path)
    s3 = boto3.client("s3")
    cpu_buffer = io.BytesIO()
    torch.save(obj, cpu_buffer)  # save to cpu
    cpu_buffer.seek(0)
    s3.upload_fileobj(cpu_buffer, save_bucket, path)
    cpu_buffer.close()


ONEGB = 1024 * 1024 * 1024


# Upload a single file to Google Cloud Storage
def upload_file_to_gcs(local_file_path, bucket_name, destination_blob_path):
    import os

    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_UPLOAD_THRESHOLD"] = str(
        100 * 1024 * 1024
    )  # 100 MiB threshold
    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_PARTS"] = "10"  # 10 parts in parallel
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_path, chunk_size=ONEGB)

    blob.upload_from_filename(local_file_path)

    print(f"File {local_file_path} uploaded to gs://{bucket_name}/{destination_blob_path}")


# Upload an entire directory to Google Cloud Storage
def upload_directory_to_gcs(local_directory, bucket_name, destination_prefix=""):
    import os

    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_UPLOAD_THRESHOLD"] = str(
        100 * 1024 * 1024
    )  # 100 MiB threshold
    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_PARTS"] = "10"  # 10 parts in parallel
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)

            # Determine the blob path in GCS
            relative_path = os.path.relpath(local_path, local_directory)
            blob_path = os.path.join(destination_prefix, relative_path).replace(
                "\\", "/"
            )  # Ensure proper path separators

            # Upload the file
            blob = bucket.blob(blob_path, chunk_size=ONEGB)
            blob.upload_from_filename(local_path)

            print(f"File {local_path} uploaded to gs://{bucket_name}/{blob_path}")

    print("Directory upload complete")


# Upload a single file to Azure Blob Storage
def upload_file_to_azure(local_file_path, container_or_account, destination_blob_path):
    """Upload a single file to Azure Blob Storage.

    Args:
        local_file_path: Local path to the file to upload
        container_or_account: Either container name or account.blob.core.windows.net format
        destination_blob_path: Destination path in the container
    """
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential

    # Parse container_or_account - could be "account.blob.core.windows.net" or just container name
    if ".blob.core.windows.net" in container_or_account:
        # Format: account.blob.core.windows.net
        account_url = f"https://{container_or_account}"
        # Container is the first path component of destination_blob_path
        parts = destination_blob_path.split("/", 1)
        container_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
    else:
        # container_or_account is the container name, use env var for account
        account_name = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
        if not account_name:
            raise ValueError("AZURE_STORAGE_ACCOUNT environment variable required when using container name directly")
        account_url = f"https://{account_name}.blob.core.windows.net"
        container_name = container_or_account
        blob_name = destination_blob_path

    # Try connection string first, then DefaultAzureCredential
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    else:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url, credential=credential)

    container_client = blob_service_client.get_container_client(container_name)

    # Create container if it doesn't exist
    try:
        container_client.create_container()
    except Exception:
        pass  # Container already exists

    blob_client = container_client.get_blob_client(blob_name)

    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True, max_concurrency=4)

    print(f"File {local_file_path} uploaded to az://{container_name}/{blob_name}")


# Upload an entire directory to Azure Blob Storage
def upload_directory_to_azure(local_directory, container_or_account, destination_prefix=""):
    """Upload an entire directory to Azure Blob Storage.

    Args:
        local_directory: Local directory to upload
        container_or_account: Either container name or account.blob.core.windows.net format
        destination_prefix: Prefix path in the container
    """
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential

    # Parse container_or_account
    if ".blob.core.windows.net" in container_or_account:
        account_url = f"https://{container_or_account}"
        parts = destination_prefix.split("/", 1) if destination_prefix else ["default", ""]
        container_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
    else:
        account_name = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
        if not account_name:
            raise ValueError("AZURE_STORAGE_ACCOUNT environment variable required when using container name directly")
        account_url = f"https://{account_name}.blob.core.windows.net"
        container_name = container_or_account
        prefix = destination_prefix

    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    else:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url, credential=credential)

    container_client = blob_service_client.get_container_client(container_name)

    try:
        container_client.create_container()
    except Exception:
        pass

    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_directory)
            blob_path = os.path.join(prefix, relative_path).replace("\\", "/")

            blob_client = container_client.get_blob_client(blob_path)

            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True, max_concurrency=4)

            print(f"File {local_path} uploaded to az://{container_name}/{blob_path}")

    print("Azure directory upload complete")


def upload_to_remote_background(config, global_step, local_global_step_folder, main_rank_latest_checkpointed_iteration):
    import time

    def _upload_to_remote_background(config, global_step, local_global_step_folder):
        dir_path = os.path.join(config.trainer.remote_upload_dir, f"global_step_{global_step}")
        print(f"Uploading checkpoint to path: {dir_path}")
        s = time.time()
        upload_dir_to_anyscale(local_global_step_folder, dir_path)
        e = time.time()
        print(f"took {e - s} to upload")

    # only upload on main rank/ caller
    file_path = os.path.join(config.trainer.remote_upload_dir, "latest_checkpointed_iteration.txt")
    upload_file_to_anyscale(main_rank_latest_checkpointed_iteration, file_path)

    # use num_cpus > 0 to schedule only on worker nodes
    remote_func = ray.remote(num_cpus=0.01, scheduling_strategy="SPREAD")(_upload_to_remote_background)

    return [remote_func.remote(config, global_step, local_global_step_folder) for _ in range(config.trainer.nnodes)]
