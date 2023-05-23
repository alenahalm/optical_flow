import cv2 as cv
import torch
import time
import numpy as np

Q = np.array((
    [1.0, 0.0, 0.0, -160.0],
    [0.0, 1.0, 0.0, -120.0],
    [0.0, 0.0, 0.0, 350.0],
    [0.0, 0.0, 1.0/90.0, 0.0]), dtype=np.float32)

model_type = "MiDaS_small"

midas = torch.hub.load('intel-isl/MiDaS', model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

cap = cv.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    start = time.time()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    input_batch = transform(frame).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv.normalize(depth_map, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    pointCloud = cv.reprojectImageTo3D(depth_map, Q, handleMissingValues=True)

    mask_map = depth_map > 0.4

    output_points = pointCloud[mask_map]
    output_colors = frame[mask_map]

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv.applyColorMap(depth_map, cv.COLORMAP_MAGMA)

    cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv.imshow('Image', frame)
    cv.imshow('DepthMap', depth_map)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
    '''

    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, ' %f %f %f %d %d %d')

output_file = 'pointCloud.ply'
create_output(output_points, output_colors, output_file)

cap.release()
cv.destroyAllWindows()