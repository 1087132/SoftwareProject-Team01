import unittest
import os
import cv2

class TestClassroomVision(unittest.TestCase):
    def test_cascade_file_exists(self):
        # Verify the OpenCV Haar cascade model file is available on the system
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.assertTrue(os.path.exists(cascade_path), "Haar cascade file is missing. OpenCV may not be installed correctly.")

if __name__ == '__main__':
    unittest.main()
