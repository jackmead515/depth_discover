import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import cv2

def cv_to_piltk(img):
    return ImageTk.PhotoImage(Image.fromarray(img))

class Frame():

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.root = None
        self.scrollbar = None
        self.canvas = None
        self.scales = None
        self.button = None
        self.image = {
            'full': {
                'ocv': None,
                'cv': None,
                'tk': None,
                'ca': None
            },
            'depth': {
                'cv': None,
                'tk': None,
                'ca': None
            }
        }
        self.depth_coefs = None
        self.depth_coefs_limits = None
        self.depth_computer = None

    def compute_depth(self):
        image = self.image['full']['ocv']
        height, width = image.shape[:2]
        left_image = image[0:height, 0 : width // 2]
        right_image = image[0:height, width // 2 : width]

        # new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        #     self.depth_coefs.get('matrix'),
        #     self.depth_coefs.get('distortion'),
        #     (width, height),
        #     1,
        #     (width, height)
        # )
        # left_undistorted = cv2.undistort(
        #     left_image,
        #     self.depth_coefs.get('matrix'),
        #     self.depth_coefs.get('distortion'),
        #     None,
        #     new_camera_matrix
        # )
        # right_undistorted = cv2.undistort(
        #     right_image,
        #     self.depth_coefs.get('matrix'),
        #     self.depth_coefs.get('distortion'),
        #     None, 
        #     new_camera_matrix
        # )

        disparity = self.depth_computer.compute(left_image, right_image)
        #disparity = disparity.astype(np.float32) / 16.0

        #focal_length = (self.depth_coefs.get('matrix')[0][0] + self.depth_coefs.get('matrix')[1][1]) / 2
        #depth = np.zeros(disparity.shape)
        #depth[:] = np.NaN
        #depth[disparity > 0] = self.depth_coefs['distance'] * focal_length / disparity[disparity > 0]

        disparity = cv2.resize(disparity, (int(self.width/2), self.height), interpolation=cv2.INTER_AREA)

        self.image['depth']['cv'] = disparity

    def recompute_depth(self):
        self.depth_computer = cv2.StereoSGBM_create(**self.depth_coefs)
        self.compute_depth()
        self.image['depth']['tk'] = cv_to_piltk(self.image['depth']['cv'])
        self.canvas.itemconfigure(self.image['depth']['ca'], image=self.image['depth']['tk'])

    def on_click_reset(self):
        self.image['full']['cv'] = np.copy(self.image['full']['ocv'])
        self.image['full']['tk'] = cv_to_piltk(self.image['full']['cv'])
        self.canvas.itemconfigure(self.image['full']['ca'], image=self.image['full']['tk'])

    def on_scale_depth_coef(self, param, convert):
        def on_scale(value):
            self.depth_coefs[param] = convert(value)
            self.recompute_depth()
        return on_scale

    def create(self):
        self.create_depth()
        self.compute_depth()
        self.create_ui()

    def create_depth(self):
        self.image['full']['ocv'] = cv2.imread('./stereo.png')

        resized = cv2.resize(self.image['full']['ocv'], (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.image['full']['cv'] = np.copy(resized)
        self.depth_coefs = {
            'minDisparity': 16,
            #'maxDisparity': 1,
            'numDisparities': 96,
            #'SADWindowSize': 3,
            'uniquenessRatio': 10,
            'speckleWindowSize': 100,
            'speckleRange': 32,
            'P1': 216,
            'P2': 864,
            #'fullDP': False
        }

        self.depth_coefs_limits = {
            'minDisparity': [0, 30],
            #'maxDisparity': 1,
            'numDisparities': [0, 200],
            #'SADWindowSize': 3,
            'uniquenessRatio': [0, 20],
            'speckleWindowSize': [0, 500],
            'speckleRange': [0, 100],
            'P1': [0, 1000],
            'P2': [0, 1000],
            #'fullDP': False
        }

        self.depth_computer = cv2.StereoSGBM_create(**self.depth_coefs)

    def create_ui(self):
        self.root = tk.Tk()

        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=400,
            bg='black'
        )
        self.canvas.pack()

        self.image['full']['tk'] = cv_to_piltk(self.image['full']['ocv'])
        self.image['full']['ca'] = self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.image['full']['tk']
        )

        self.image['depth']['tk'] = cv_to_piltk(self.image['depth']['cv'])
        self.image['depth']['ca'] = self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.image['depth']['tk']
        )

        self.button = tk.Button(
            self.root,
            text="Reset",
            command=self.on_click_reset,
        )
        self.button.pack()

        self.create_scales()

    def create_scales(self):
        self.scales = {}
        for depth_key in self.depth_coefs:
            self.scales[depth_key] = tk.Scale(
                self.root,
                label=depth_key,
                from_=self.depth_coefs_limits[depth_key][0],
                to=self.depth_coefs_limits[depth_key][1],
                variable=tk.IntVar(value=self.depth_coefs[depth_key]),
                command=self.on_scale_depth_coef(depth_key, int),
                length=self.width,
                orient=tk.HORIZONTAL,
                resolution=1,
            )
            self.scales[depth_key].pack()

    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    frame = Frame(width=800, height=300)
    frame.create()
    frame.start()