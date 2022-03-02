import os
import cv2
import face_recognition


def detect_faces(input_directory, output_directory, desired_size, scale_factor):
    """
    Detect if the given image contains a face in it.
    """
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    except OSError:
        print('Error creating directory...')


    for file in os.listdir(input_directory):
        if not file.startswith('.'):
            filename = os.path.join(input_directory, file)

            img = face_recognition.load_image_file(filename)
            basename = os.path.basename(file)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(img_rgb)

            if len(face_locations) == 1:
                print(f"Face detected on image: {basename}")
                for top, right, bottom, left in face_locations:
                    ## Calculate center points and rectangle side length
                    width = right - left
                    height = bottom - top
                    cX = left + width // 2
                    cY = top + height // 2
                    M = (abs(width) + abs(height)) / 2

                    ## Get the resized rectangle points
                    newLeft = max(0, int(cX - scale_factor * M))
                    newTop = max(0, int(cY - scale_factor * M))
                    newRight = min(img_rgb.shape[1], int(cX + scale_factor * M))
                    newBottom = min(img_rgb.shape[0], int(cY + scale_factor * M))

                    # try to crop square face
                    face_crop_img = img_rgb[newTop:newBottom, newLeft:newRight]

                    # get face image size in (height, width) format
                    img_size = face_crop_img.shape[:2]

                    ratio = float(desired_size) / max(img_size)
                    new_size = tuple([int(x * ratio) for x in img_size])  # new_size should be in (width, height) format

                    # resize image
                    image = cv2.resize(face_crop_img, (new_size[1], new_size[0]))

                    # add black padding to make square image
                    delta_w = desired_size - new_size[1]
                    delta_h = desired_size - new_size[0]
                    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                    left, right = delta_w // 2, delta_w - (delta_w // 2)

                    color = [0, 0, 0]
                    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                 value=color)

                    # save image to an output directory
                    cv2.imwrite(os.path.join(output_directory, f"{basename}"), new_img)
            else:
                print(f"No face detected on image: {basename}")


def main():
    input_dir = "images"
    output_dir = "faces_detected"
    desired_size = 350
    scale_factor = 0.8
    detect_faces(input_dir, output_dir, desired_size, scale_factor)


if __name__ == '__main__':
    main()
