from pixsfm.refine_hloc import PixSfM

if __name__ == "__main__":
    # refiner = PixSfM()
    # model, debug_outputs = refiner.reconstruction(
    #     path_to_working_directory,
    #     path_to_image_dir,
    #     path_to_list_of_image_pairs,
    #     path_to_keypoints.h5,
    #     path_to_matches.h5,
    # )

    path_to_reference_model = '/mnt/data4/yswangdata4/dataset/tnt_train/Barn'
    # model is a pycolmap.Reconstruction 3D model
    conf = {"BA": {"optimizer": {
        "refine_focal_length": False,
        "refine_extra_params": False,  # distortion parameters
        "refine_extrinsics": False,  # camera poses
    }}}
    refiner = PixSfM(conf=conf)
    refiner.triangulation(...)
    model, _ = refiner.triangulation(..., path_to_reference_model, ...)