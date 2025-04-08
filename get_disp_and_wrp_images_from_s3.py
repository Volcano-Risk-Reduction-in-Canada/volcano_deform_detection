import os
import json
from datetime import datetime

from data_utils import s3

def serialize(obj):
    """Convert datetime objects to ISO 8601 strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def save_paginated_response(pages, output_file):
    """
    Save paginated S3 responses to a JSON file for debugging.

    Parameters:
    - pages: iterable - Paginated response from S3
    - output_file: str - File path to save the JSON
    """
    all_contents = []

    for page in pages:
        if "Contents" in page:
            all_contents.extend(page["Contents"])

    # Write all collected contents to the JSON file
    with open(output_file, 'w') as file:
        json.dump({"Contents": all_contents}, file, indent=4, default=serialize)
    
    # Count total items
    total_items = len(all_contents)

    print(f"Full paginated response saved to {output_file}")
    print(f"Total number of items in 'Contents': {total_items}")

    return total_items


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
    disp_suffix, wrp_suffix = ".disp.geo.tif", ".adf.wrp.geo.tif"

    # Ensure local directories exist
    os.makedirs(f"{disp_local_dir}/{site}/{beam}", exist_ok=True)
    os.makedirs(f"{wrp_local_dir}/{site}/{beam}", exist_ok=True)

    # List all objects in the bucket folder path
    prefix = f"{site}/{beam}/"
    # response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, PaginationConfig={"PageSize": 1000})

    # Convert pages to a list so it can be reused
    pages_list = list(pages)

    # Save the paginated response to a JSON file
    total_items = save_paginated_response(pages_list, 's3_list_objects_response.json')
    print(f"Total items saved: {total_items}")
   

    # # Collect matching disp and wrp files by prefix
    # disp_files = {obj["Key"][:-len(disp_suffix)] for obj in response["Contents"] if obj["Key"].endswith(disp_suffix)}
    # wrp_files = {obj["Key"][:-len(wrp_suffix)] for obj in response["Contents"] if obj["Key"].endswith(wrp_suffix)}
    disp_files = []
    wrp_files = []

    page_count = 0  # Counter for pages
    object_count = 0  # Counter for objects

    for page in pages_list:
        page_count += 1  # Increment page counter
        print(f"Processing Page {page_count}: Contents Found - {'Contents' in page}")
        print(f"IsTruncated: {page.get('IsTruncated', False)}")
        if "Contents" not in page:
            # Check if the bucket is empty
            print(f"No files found in bucket: {bucket_name}")
            return

        for obj in page["Contents"]:
            object_count += 1  # Increment object counter
            key = obj["Key"]
            print(key)
            if key.endswith(disp_suffix):
                disp_files.append(key[:-len(disp_suffix)])
                print('ADDING DISP', key)
            if key.endswith(wrp_suffix):
                wrp_files.append(key[:-len(wrp_suffix)])
                print('ADDING WRP', key)
    # Final counts
    print(f"Total Pages Processed: {page_count}")
    print(f"Total Objects Processed: {object_count}")
    print(f"Total DISP Files Found: {len(disp_files)}")
    print(f"Total WRP Files Found: {len(wrp_files)}")

    paired_prefixes = set(disp_files).intersection(set(wrp_files))
    print('DISP FILES', disp_files)
    print('WRP FILES', wrp_files)
    print('PAIRED PREFIXES', paired_prefixes)
    if not paired_prefixes:
        print(f"No paired files found in bucket: {bucket_name}, site: {site}, beam: {beam}")
        return

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
