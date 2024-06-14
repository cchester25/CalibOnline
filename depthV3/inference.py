import numpy as np
from tqdm import tqdm
import torch
from imageio import imread, imwrite
from path import Path
import os
import time
from utils.config import get_opts, get_training_size


from utils.SC_DepthV3 import SC_DepthV3

import utils.custom_transforms as custom_transforms
import PIL.Image as pil
from utils.visualization import *
import rospy #自添加 
import ros_numpy #自添加
from std_msgs.msg import Header #自添加 
from sensor_msgs.msg import Image #自添加 
import cv2
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()
save_path = "/media/ubun/DATA/Projects/calibration/test_pictures/demos/imagedepth/"


@torch.no_grad()
def load_model(args):
    """
    use sc-depthv3
    """
    system = SC_DepthV3(args)
    system = system.load_from_checkpoint(args.ckpt_path, strict=False)
    global model
    model = system.depth_net
    model.cuda()
    model.eval()
    training_size = get_training_size(args.dataset_name)
    # normalization
    global inference_transform
    inference_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )

def image_callback(imgmsg):
    """
    get image
    """
    raw_pub.publish(imgmsg)

    ori_width = imgmsg.width
    ori_height = imgmsg.height
    input_image = ros_numpy.numpify(imgmsg)

    tensor_img = inference_transform([input_image])[0][0].unsqueeze(0).cuda()
    pred_depth = model(tensor_img)
    disp_resized = torch.nn.functional.interpolate(
            pred_depth, (ori_height, ori_width), mode="bilinear", align_corners=False)

    vis = visualize_depth(disp_resized[0, 0]).permute(
        1, 2, 0).numpy() * 255

    blurred = cv2.GaussianBlur(vis.astype(np.uint8), (5, 5), 0)
    edge = cv2.Canny(blurred, 20, 60)

    ros_depth_msg = bridge.cv2_to_imgmsg(edge,encoding="mono8")
    ros_depth_msg.header.stamp = imgmsg.header.stamp
    ros_depth_msg.header.frame_id = "camera_link"
    depth_pub.publish(ros_depth_msg)

if __name__ == '__main__':
    args = get_opts()
    load_model(args)
    rospy.init_node('listener', anonymous=True)
    image_size = get_training_size(args.dataset_name)
    feed_width = image_size[1]
    feed_height = image_size[0]
    # fake image
    img = np.zeros((feed_height, feed_width, 3), dtype=np.float32)
    img = inference_transform([img])[0][0].unsqueeze(0).cuda()
    # warm up
    for i in range(10):
        outputs = model(img)
    # global f
    # f=open("/media/ubun/DATA/Projects/calibration/Code/CalibOnlineV3/src/depthV3/time.txt","w")
    global depth_pub
    global raw_pub
    depth_pub = rospy.Publisher("/img_depth", Image, queue_size=1)
    raw_pub = rospy.Publisher("/raw_img", Image, queue_size=1)
    print("init successfully")
    rospy.Subscriber("/kitti/camera_color_left/image_raw", Image, image_callback,queue_size=10)
    # rospy.Subscriber("/stereo/frame_left/image_raw", Image, image_callback,queue_size=1)
    # rospy.Subscriber("/cam2/cam2_raw", Image, image_callback,queue_size=10)
    # rospy.Subscriber("/pylon_camera_node/cam1/image_raw", Image, image_callback,queue_size=10)
    rospy.spin()
