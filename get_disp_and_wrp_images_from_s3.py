import argparse
import boto3
import os


# ####################### HELPER FUNCTION #######################################
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


def main():
    args = parse_args()
    download_images_from_s3(
        args.bucket_name,
        args.site,
        args.beam,
        args.disp_dir,
        args.wrp_dir
    )

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Download paired .disp.geo.tif and .adf.wrp.geo.tif images from S3")
    parser.add_argument("--bucket_name",
                        type=str,
                        help="Name of S3 Bucket",
                        required=True),
    parser.add_argument("--site",
                        type=str,
                        help="Site folder in S3 bucket",
                        required=True),
    parser.add_argument("--beam",
                        type=str,
                        help="Beam folder in S3 bucket",
                        required=True),
    parser.add_argument("--disp_dir",
                        type=str,
                        help="Local directory to save disp images",
                        required=True),
    parser.add_argument("--wrp_dir",
                        type=str,
                        help="Local directory to save wrp images",
                        required=True),
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()