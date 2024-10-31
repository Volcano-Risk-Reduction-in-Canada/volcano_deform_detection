import argparse
import time
import logging
import gc

logging.basicConfig(level=logging.DEBUG, filename='debug.log')

from get_probability_map_func import run_volcano_deformation_detection


def main():
    mstart = time.time()
    args = parse_args()

    # MODEL 1 - Latitude/Longitude coordinate system
    start = time.time()
    print(f"Model 1 Latitude/Longitude")
    run_volcano_deformation_detection(
        args.image_name,
        args.site,
        args.beam,
        "models/model1.pd",
        True,
        True
    )
    run_volcano_deformation_detection(
        args.image_name,
        args.site,
        args.beam,
        "models/model1.pd",
        True,
        False
    )
    endt = time.time()
    logging.debug(f"Time elapsed for MODEL 1 - Latitude/Longitude: {endt - start} seconds")

    # MODEL 1 - UTM coordinate system
    start = time.time()
    print(f"Model 1 UTM")
    run_volcano_deformation_detection(
        args.image_name,
        args.site,
        args.beam,
        "models/model1.pd",
        False,
        True
    )
    run_volcano_deformation_detection(
        args.image_name,
        args.site,
        args.beam,
        "models/model1.pd",
        False,
        False
    )
    endt = time.time()
    logging.debug(f"Time elapsed for MODEL 1 - UTM: {endt - start} seconds")

    # MODEL 2 - Latitude/Longitude coordinate system
    start = time.time()
    print(f"Model 2 Latitude/Longitude")
    run_volcano_deformation_detection(
        args.image_name,
        args.site,
        args.beam,
        "models/model2.pd",
        True,
        True
    )
    run_volcano_deformation_detection(
        args.image_name,
        args.site,
        args.beam,
        "models/model2.pd",
        True,
        False
    )
    endt = time.time()
    logging.debug(f"Time elapsed for MODEL 2 - Latitude/Longitude: {endt - start} seconds")

    # MODEL 2 - UTM coordinate system
    start = time.time()
    print(f"Model 2 UTM")
    run_volcano_deformation_detection(
        args.image_name,
        args.site,
        args.beam,
        "models/model2.pd",
        False,
        True
    )
    run_volcano_deformation_detection(
        args.image_name,
        args.site,
        args.beam,
        "models/model2.pd",
        False,
        False
    )
    endt = time.time()
    logging.debug(f"Time elapsed for MODEL 2 - UTM: {endt - start} seconds")

    mendt = time.time()
    print(f"Time elapsed: {mendt - mstart} seconds")



def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Volcano Deformation Detection at various resolutions")
    parser.add_argument("--image_name",
                        type=str,
                        help="Name of InSAR image",
                        required=True),
    parser.add_argument("--site",
                        type=str,
                        help="Site of InSAR image",
                        required=True),
    parser.add_argument("--beam",
                        type=str,
                        help="Beam of InSAR image",
                        required=True),
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
