import cv2
import numpy as np
from scipy.sparse import csr_matrix

def get_frame_pairs(frames: np.ndarray):
    return [(frames[i], frames[i + 1]) for i in range(0, len(frames), 2)]

def process_window_lk(window_size_x, window_size_y, start_x, start_y, frame_tplus1: np.ndarray, frame_t: np.ndarray, fps, epsilon=1e-4):
    # Calculate the temporal derivative
    frame_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
    frame_tplus1 = cv2.cvtColor(frame_tplus1, cv2.COLOR_BGR2RGB)
    H, W, _ = frame_t.shape

    delta_t = 1 / fps
    I_t: np.ndarray = np.subtract(frame_tplus1[start_x: start_x + window_size_x, start_y: start_y + window_size_y], frame_t[start_x: start_x + window_size_x, start_y: start_y + window_size_y]) / delta_t
    
    # Placeholder for the spatial derivatives
    I_x = np.zeros((window_size_x, window_size_y), dtype=np.float32)
    I_y = np.zeros((window_size_x, window_size_y), dtype=np.float32)

    if not (start_x + window_size_x < W and start_y + window_size_y < H and len(frame_t[start_x: start_x + window_size_x, start_y: start_y + window_size_y]) > 0):
        return np.zeros((2,1), dtype=np.float32)
    
    smoothed = cv2.GaussianBlur(frame_t[start_x: start_x + window_size_x, start_y: start_y + window_size_y], (5,5), sigmaX=1.0, sigmaY=1.0)

    I_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3) * 0.5  # ∂/∂x
    I_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3) * 0.5  # ∂/∂y

    Ix = I_x.flatten()
    Iy = I_y.flatten()
    It = I_t.flatten()

    # The system of equations would look like this
    # [ 
    #   [∑ I_x^2,   ∑ I_x I_y],
    #   [∑ I_x I_y, ∑ I_y^2]
    # ]

    A00 = np.sum(Ix * Ix)   # ∑ I_x^2
    A01 = np.sum(Ix * Iy)   # ∑ I_x I_y
    A11 = np.sum(Iy * Iy)   # ∑ I_y^2

    B0 = -np.sum(Ix * It)   # -∑ I_x I_t
    B1 = -np.sum(Iy * It)   # -∑ I_y I_t

    A = np.array([[A00, A01],
                  [A01, A11]], dtype=np.float32)
    B = np.array([B0, B1],      dtype=np.float32).reshape(2,1)

    # This is striaght from the video where we check how invertible the matrix is
    # In other words whether the system of equations are well conditioned or not
    # We can think of a system where there is hardly any change, like a patch of texture with no change
    det = A00*A11 - A01*A01
    if det > epsilon:
        uv = np.linalg.solve(A, B)   # 2×1
    else:
        uv = np.zeros((2,1), dtype=np.float32)
    return uv

capture = cv2.VideoCapture("subset_of_frames/frame_%05d.png")
frames = []
fps = 24
captured_all = False
while not captured_all:
    ret, frame = capture.read()
    if not ret:
        captured_all = True
        break
    frames.append(frame.copy())

window_w, window_h = 100, 100

frames = np.array(frames)
T = len(frames)
H, W = frames[0].shape[:2]
nY  = H // window_h
nX  = W // window_w

flows = np.zeros((T-1, nY, nX, 2), dtype=np.float32)

for t in range(T-1):
    ft, ftplus1 = frames[t], frames[t+1]
    for y in range(nY):
        y0 = y * window_h
        for x in range(nX):
            x0 = x * window_w
            uv = process_window_lk(window_w, window_h, start_x=x0, start_y=y0, frame_tplus1=ftplus1, frame_t=ft, fps=fps)
            # store [u,v]
            flows[t, y, x, 0] = uv[0,0]  # u
            flows[t, y, x, 1] = uv[1,0]  # v

T, patch_y, patch_x, uv_shape = flows.shape

print (flows.shape)        