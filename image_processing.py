"""
Image Processing Module
AI Precision Agriculture System

Handles:
1. Image loading
2. Image resizing
3. Image normalization
4. Image preprocessing
5. Image enhancement
"""

import cv2
import numpy as np
import os


class ImageProcessor:

    """
    Image processing class used for
    plant disease detection
    """

    def __init__(self):

        self.image_size = 224

    # -------------------------------------------------
    # LOAD IMAGE
    # -------------------------------------------------

    def load_image(self, image_path):

        """
        Load image from disk
        """

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                "Image file does not exist"
            )

        image = cv2.imread(image_path)

        if image is None:

            raise Exception(
                "Unable to read image"
            )

        return image

    # -------------------------------------------------
    # RESIZE IMAGE
    # -------------------------------------------------

    def resize_image(self, image):

        """
        Resize image to CNN input size
        """

        resized = cv2.resize(

            image,

            (self.image_size, self.image_size)

        )

        return resized

    # -------------------------------------------------
    # NORMALIZE IMAGE
    # -------------------------------------------------

    def normalize_image(self, image):

        """
        Normalize image pixels
        """

        image = image.astype("float32")

        image = image / 255.0

        return image

    # -------------------------------------------------
    # CONVERT TO ARRAY
    # -------------------------------------------------

    def convert_to_array(self, image):

        """
        Convert image to numpy array
        """

        array = np.array(image)

        return array

    # -------------------------------------------------
    # ADD BATCH DIMENSION
    # -------------------------------------------------

    def add_batch_dimension(self, image):

        """
        CNN models require batch dimension
        """

        image = np.expand_dims(image, axis=0)

        return image

    # -------------------------------------------------
    # PREPROCESS IMAGE
    # -------------------------------------------------

    def preprocess_image(self, image_path):

        """
        Complete preprocessing pipeline
        """

        image = self.load_image(image_path)

        image = self.resize_image(image)

        image = self.normalize_image(image)

        image = self.convert_to_array(image)

        image = self.add_batch_dimension(image)

        return image

    # -------------------------------------------------
    # SHOW IMAGE
    # -------------------------------------------------

    def show_image(self, image):

        """
        Display image for debugging
        """

        cv2.imshow("Leaf Image", image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    # -------------------------------------------------
    # CONVERT TO GRAYSCALE
    # -------------------------------------------------

    def convert_to_gray(self, image):

        """
        Convert image to grayscale
        """

        gray = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY

        )

        return gray

    # -------------------------------------------------
    # EDGE DETECTION
    # -------------------------------------------------

    def detect_edges(self, image):

        """
        Detect leaf edges
        """

        gray = self.convert_to_gray(image)

        edges = cv2.Canny(

            gray,

            100,

            200

        )

        return edges

    # -------------------------------------------------
    # IMAGE BLUR
    # -------------------------------------------------

    def blur_image(self, image):

        """
        Apply Gaussian blur
        """

        blurred = cv2.GaussianBlur(

            image,

            (5,5),

            0

        )

        return blurred

    # -------------------------------------------------
    # IMAGE SHARPENING
    # -------------------------------------------------

    def sharpen_image(self, image):

        """
        Improve image clarity
        """

        kernel = np.array([

            [0,-1,0],

            [-1,5,-1],

            [0,-1,0]

        ])

        sharpened = cv2.filter2D(

            image,

            -1,

            kernel

        )

        return sharpened

    # -------------------------------------------------
    # IMAGE ROTATION
    # -------------------------------------------------

    def rotate_image(self, image, angle):

        """
        Rotate image
        """

        height, width = image.shape[:2]

        center = (width//2, height//2)

        matrix = cv2.getRotationMatrix2D(

            center,

            angle,

            1.0

        )

        rotated = cv2.warpAffine(

            image,

            matrix,

            (width,height)

        )

        return rotated

    # -------------------------------------------------
    # IMAGE FLIP
    # -------------------------------------------------

    def flip_image(self, image):

        """
        Flip image horizontally
        """

        flipped = cv2.flip(

            image,

            1

        )

        return flipped


# -------------------------------------------------
# TEST IMAGE PROCESSOR
# -------------------------------------------------

def test_processor():

    """
    Test image preprocessing
    """

    processor = ImageProcessor()

    image_path = input(
        "Enter image path: "
    )

    image = processor.load_image(image_path)

    image = processor.resize_image(image)

    image = processor.normalize_image(image)

    print("Image processed successfully")


if __name__ == "__main__":

    test_processor()