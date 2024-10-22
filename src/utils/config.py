from pathlib import Path
import numpy as np
import os
import yaml


# Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()

    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    # rudimentray check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()
    return calibration_settings


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix='', output_folder='camera_parameters'):

    # create folder if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    camera0_rot_trans_filename = os.path.join(
        output_folder, prefix + 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    # R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join(
        'camera_parameters', prefix + 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1


def load_cam_instric_data(path: Path):
    """
    File is structured like:
    intrinsic:
    3827.525751047498 0.0 988.4865219418614
    0.0 3863.6973721859827 539.4811218948392
    0.0 0.0 1.0
    distortion:
    -0.8937816248419109 8.488864130033766 0.03514306607745752 -0.05924502203830264 -130.39750176892838
    """
    cmtx = np.zeros((3, 3), dtype=np.float32)
    dist = np.zeros((1, 5), dtype=np.float32)
    with path.open('r') as f:
        lines = f.readlines()
        # split the lines into intrinsic and distortion parts
        intrinsic = lines[1:4]
        cmtx = np.array([list(map(float, line.split()))
                         for line in intrinsic], dtype=np.float32)
        distortion = lines[5].split()
        dist = np.array([float(num) for num in distortion]).reshape((1, 5))
    return cmtx, dist


def load_cam_rot_trans_data(path: Path):
    """
    R:
    1.0 0.0 0.0 
    0.0 1.0 0.0 
    0.0 0.0 1.0 
    T:
    0.0 
    0.0 
    0.0 
    """
    R = np.zeros((3, 3), dtype=np.float32)
    T = np.zeros((3, 1), dtype=np.float32)

    with path.open('r') as f:
        lines = f.readlines()
        # split the lines into R and T parts
        R = np.array([list(map(float, line.split()))
                      for line in lines[1:4]], dtype=np.float32)
        T = np.array([list(map(float, line.split()))
                      for line in lines[5:8]], dtype=np.float32)
    return R, T
