import numpy as np
import sys
import os
from pathlib import Path

from src.utils.calibration import calibrate_camera_for_intrinsic_parameters, check_calibration, stereo_calibrate
from src.utils.config import load_cam_instric_data, load_cam_rot_trans_data, parse_calibration_settings_file, save_extrinsic_calibration_parameters
from src.utils.save import save_camera_intrinsics, save_frames_single_camera, save_frames_two_cams
from argparse import ArgumentParser

# save camera intrinsic parameters to file
if __name__ == '__main__':
    parser = ArgumentParser(
        description='Stereo camera calibration')
    parser.add_argument('settings',
                        type=lambda x: Path(x).exists(),
                        help='Settings file',
                        default='calibration_settings.yaml')
    parser.add_argument('--camera0',
                        type=str,
                        help='Id of camera0',
                        required=True)
    parser.add_argument('--camera1',
                        type=str,
                        help='Id of camera1 (could be rtsp url, camera id, etc)',
                        required=True)

    parser.add_argument('--output_frames',
                        type=str,
                        help='Output folder',
                        default='output_frames')
    parser.add_argument('--output_frames_pair',
                        type=str,
                        default='output_frames',
                        help='Output folder for paired frames')
    parser.add_argument('--output_intrinsics',
                        type=str,
                        default='camera_parameters',
                        help='Output folder for camera intrinsics')

    parser.add_argument('--load_cam_param_folder',
                        type=str,
                        help='Load camera rotation/translation/intrinsics param from folder, (will skip calibration)',
                        default=None)

    parser.parse_args()

    calibration_settings = parse_calibration_settings_file(parser.settings)

    match parser.load_cam_param_folder is not None:
        case False:
            print("Calibrating cameras")
            """Step1. Save calibration frames for single cameras"""
            save_frames_single_camera('camera0', parser.output_frames)
            save_frames_single_camera('camera1', parser.output_frames)

            """Step2. Obtain camera intrinsic matrices and save them"""
            def camera_instrinsics(camera_name):
                images_prefix = os.path.join('frames', f'{camera_name}*')
                cmtx, dist = calibrate_camera_for_intrinsic_parameters(
                    images_prefix, calibration_settings)
                save_camera_intrinsics(
                    cmtx, dist, camera_name, parser.output_intrinsics)
                return cmtx, dist

            cmtx0, dist0 = camera_instrinsics('camera0')
            cmtx1, dist1 = camera_instrinsics('camera1')

            """Step3. Save calibration frames for both cameras simultaneously"""
            save_frames_two_cams('camera0', 'camera1',
                                 calibration_settings, parser.output_frames_pair)
            """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
            frames_prefix_c0 = os.path.join(
                parser.output_frames_pair, 'camera0*')
            frames_prefix_c1 = os.path.join(
                parser.output_frames_pair, 'camera1*')
            r, t = stereo_calibrate(cmtx0, dist0, cmtx1, dist1,
                                    frames_prefix_c0, frames_prefix_c1, calibration_settings)

            """Step5. Save calibration data where camera0 defines the world space origin."""
            # camera0 rotation and translation is identity matrix and zeros vector
            r0 = np.eye(3, dtype=np.float32)
            t0 = np.array([0., 0., 0.]).reshape((3, 1))

            save_extrinsic_calibration_parameters(
                r0, t0, r, t, output_folder=parser.output_intrinsics)
            r1 = r
            t1 = t  # to avoid confusion, camera1 R and T are labeled R1 and T1

        case _:
            cmtx0, dist0 = load_cam_instric_data(
                f"{parser.load_intrinsics_folder}/camera0_intrinsics.dat")
            cmtx1, dist1 = load_cam_instric_data(
                f"{parser.load_intrinsics_folder}/camera1_intrinsics.dat")

            r0, t0 = load_cam_rot_trans_data(
                f"{parser.load_intrinsics_folder}/camera0_rot_trans.dat"
            )
            r1, t1 = load_cam_rot_trans_data(
                f"{parser.load_intrinsics_folder}/camera1_rot_trans.dat"
            )

    camera0_data = [cmtx0, dist0, r0, t0]
    camera1_data = [cmtx1, dist1, r1, t1]
    check_calibration('camera0', camera0_data, 'camera1',
                      camera1_data, _zshift=60., calibration_settings=calibration_settings)

    """Optional. Define a different origin point and save the calibration data"""
    # #get the world to camera0 rotation and translation
    # R_W0, T_W0 = get_world_space_origin(cmtx0, dist0, os.path.join('frames_pair', 'camera0_4.png'))
    # #get rotation and translation from world directly to camera1
    # R_W1, T_W1 = get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0,
    #                                           cmtx1, dist1, R1, T1,
    #                                           os.path.join('frames_pair', 'camera0_4.png'),
    #                                           os.path.join('frames_pair', 'camera1_4.png'),)

    # #save rotation and translation parameters to disk
    # save_extrinsic_calibration_parameters(R_W0, T_W0, R_W1, T_W1, prefix = 'world_to_') #this will write R and T to disk
