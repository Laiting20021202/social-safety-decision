"""Fixed chest D415 optical pose in OpenArm base coordinates (NUC chest hand-eye calibration)."""
import numpy as np
from scipy.spatial.transform import Rotation

WIDTH, HEIGHT = 640, 480
FOCAL = 606.
EYE = np.array([.04646809135, .01551892702, .56912151185])
ROT = Rotation.from_quat([.67195626035,-.61557948041,.33814646293,-.23493330347]).as_matrix()
BASE_FROM_CAMERA = np.eye(4)
BASE_FROM_CAMERA[:3,:3] = ROT
BASE_FROM_CAMERA[:3,3] = EYE
RIGHT_START = [.55,0.,0.,.03,0.,0.,0.]
CURL_DEGREES = 36.
HAND_SCALE = .052
# A visitor reaches in from the far side. Fingers extend toward the moving gripper.
normal = -ROT[:,2]
wrist = np.array([.50,.84,.14])
wrist -= normal*(normal@wrist)
wrist /= np.linalg.norm(wrist)
HAND_ROT = np.column_stack((wrist,np.cross(normal,wrist),normal))
HAND_PARK = np.array([.65,.58,.19])
HAND_REACH = np.array([.43,-.042,.197])
