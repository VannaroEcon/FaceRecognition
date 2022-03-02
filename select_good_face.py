import dlib
import cv2
import os
import imutils

from scipy.spatial import distance as dist
from imutils import face_utils


def eye_aspect_ratio(eye):
    """
    Compute the eye aspect ratio.
    """
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y) coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y) coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear


def variance_of_laplacian(image):
    """
    Compute the Laplacian of the image and then return the focus measure, which is simply the variance of the Laplacian.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def select_good_faces(input_directory, output_directory, eye_threshold, blur_threshold):
    """
    Automate face selection which chooses the face images that are not closed eyes and too blur.
    """
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    except OSError:
        print('Error creating directory...')

    counter = 0

    # Initialize dlib's face detector and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("trainset/shape_predictor_68_face_landmarks.dat")

    # Grab the indexes of the facial landmarks for the left and right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    for d in os.listdir(input_directory):
        if not d.startswith('.'):
            filename = os.path.join(input_directory, d)
            img = cv2.imread(filename)
            basename = os.path.splitext(os.path.basename(filename))[0]
            img_resized = imutils.resize(img, width=150)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale fram
            rects = detector(gray, 0)

            # Detect the focus measure
            fm = variance_of_laplacian(gray)

            # loop over the face detection
            for rect in rects:
                # Determine facial landmarks for the face region, then convert the facial landmark (x, y) coordinates to a NumPy array.
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes.
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # Average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # Check to see if the eye aspect ratio is below the threshold, and if so, increment the frame counter.
                if ear > eye_threshold:
                    if fm > blur_threshold:
                        counter += 1
                        cv2.imwrite(f'{output_directory}/{basename}.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    input_dir = "faces_detected"
    output_dir = "faces_selected"
    eye_thres = 0.28
    blur_thres = 120

    select_good_faces(input_dir, output_dir, eye_thres, blur_thres)


if __name__ == "__main__":
    main()
