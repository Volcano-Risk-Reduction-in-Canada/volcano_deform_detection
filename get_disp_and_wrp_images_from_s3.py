import boto3
import os

def download_images_from_s3(bucket_name, site, beam, disp_local_dir, wrp_local_dir):
    """
    Download paired .disp.geo.tif and .adf.wrp.geo.tif images from an S3 bucket if both files exist.

    Parameters:
    - bucket_name: str - Name of the S3 bucket
    - site: str - Site folder in S3 bucket
    - beam: str - Beam folder in S3 bucket
    - disp_local_dir: str - Local directory to save disp images
    - wrp_local_dir: str - Local directory to save wrp images
    """
    s3 = boto3.client("s3", verify=False)
    disp_suffix, wrp_suffix = ".disp.geo.tif", ".adf.wrp.geo.tif"

    # Ensure local directories exist
    os.makedirs(f"{disp_local_dir}/{site}/{beam}", exist_ok=True)
    os.makedirs(f"{wrp_local_dir}/{site}/{beam}", exist_ok=True)

    # List all objects in the bucket folder path
    prefix = f"{site}/{beam}/"
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # Check if the bucket is empty
    if "Contents" not in response:
        print(f"No files found in bucket: {bucket_name}")
        return

    # Collect matching disp and wrp files by prefix
    disp_files = {obj["Key"][:-len(disp_suffix)] for obj in response["Contents"] if obj["Key"].endswith(disp_suffix)}
    wrp_files = {obj["Key"][:-len(wrp_suffix)] for obj in response["Contents"] if obj["Key"].endswith(wrp_suffix)}
    paired_prefixes = disp_files.intersection(wrp_files)

    # Download pairs if they don’t already exist locally
    for p in paired_prefixes:
        disp_s3_key = f"{p}{disp_suffix}"
        wrp_s3_key = f"{p}{wrp_suffix}"
        disp_local_path = os.path.join(disp_local_dir, site, beam, os.path.basename(disp_s3_key))
        wrp_local_path = os.path.join(wrp_local_dir, site, beam, os.path.basename(wrp_s3_key))

        if not os.path.exists(disp_local_path):
            s3.download_file(bucket_name, disp_s3_key, disp_local_path)
            print(f"Downloaded: {disp_s3_key} to {disp_local_path}")
        else:
            print(f"File already exists: {disp_local_path} - Skipping download")

        if not os.path.exists(wrp_local_path):
            s3.download_file(bucket_name, wrp_s3_key, wrp_local_path)
            print(f"Downloaded: {wrp_s3_key} to {wrp_local_path}")
        else:
            print(f"File already exists: {wrp_local_path} - Skipping download")

    print("Download of paired files complete.")
    return {p.split('/')[-1] for p in paired_prefixes}