import cv2 as cv
import os
from .video_capture import VideoCapture


def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name, output_folder):

    # create folder if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    out_filename = os.path.join(
        output_folder, camera_name + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')


def save_frames_single_camera(camera_name, calibration_settings, output_folder):
    # create frames directory
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # get settings
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    # view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    # open video stream and change resolution.
    # Note: if unsupported resolution is used, this does NOT raise an error.
    cap = cv.VideoCapture(camera_device_id)
    cap.set(3, width)
    cap.set(4, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if ret == False:
            # if no video data is received, can't calibrate the camera, so exit.
            print("No video data received from camera. Exiting...")
            quit()

        current_width = frame.shape[1]
        current_height = frame.shape[0]
        frame_small = cv.resize(
            frame, None, fx=width/current_width, fy=height/current_height)

        if not start:
            cv.putText(frame_small, "Press SPACEBAR to start collection frames",
                       (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown),
                       (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count),
                       (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            # save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join(
                    output_folder, camera_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)

        if k == 27:
            # if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            # Press spacebar to start data collection
            start = True

        # break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()
    cap.release()

# TODO: to refactor because main thread is provoking latency


def save_frames_two_cams(camera0_name, camera1_name, calibration_settings, output_folder):

    # create frames directory
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # settings for taking data
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']
    number_to_save = calibration_settings['stereo_calibration_frames']

    # open the video streams
    vs0 = VideoCapture(calibration_settings[camera0_name])
    vs1 = VideoCapture(calibration_settings[camera1_name])
    # cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    # cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    # set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    vs0.set_cap_flag(3, width)
    vs0.set_cap_flag(4, height)
    vs1.set_cap_flag(3, width)
    vs1.set_cap_flag(4, height)

    start = False
    saved_count = 0
    while True:

        ret0, frame0 = vs0.read()
        ret1, frame1 = vs1.read()

        if not ret0 or not ret1:
            print('Cameras not returning video data. Exiting...')
            quit()

        frame0_small = cv.resize(
            frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv.resize(
            frame1, None, fx=1./view_resize, fy=1./view_resize)

        if not start:
            cv.putText(frame0_small, "Make sure both cameras can see the calibration pattern well",
                       (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to start collection frames",
                       (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        if start:
            cv.putText(frame0_small, "Num frames: " + str(saved_count),
                       (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            cv.putText(frame1_small, "Num frames: " + str(saved_count),
                       (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        cv.imshow('frame0_small', frame0_small)
        cv.imshow('frame1_small', frame1_small)
        k = cv.waitKey(1)

        match k:
            case 27:
                # if ESC is pressed at any time, the program will exit.
                quit()
            case 32:
                # Press spacebar to start data collection
                start = True
            # when k is pressed
            case 107:
                savename = os.path.join(
                    output_folder, camera0_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame0)

                savename = os.path.join(
                    output_folder, camera1_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame1)

        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()
    vs0.stop()
    vs1.stop()
