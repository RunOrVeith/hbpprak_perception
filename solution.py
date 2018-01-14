import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
import sensor_msgs
import std_msgs
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from thread import start_new_thread
import math
import thimblerigger_config as tc
from std_srvs.srv import Trigger, TriggerResponse


class RobotMover(object):

    def __init__(self, model_name="robot"):
        self.model_name = model_name

    def go_to_pose(self, x, y, orientation):
        pass


class TeleportRobotMover(RobotMover):

    def __init__(self, model_name="robot"):
        self.get_position = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.set_position = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        super(TeleportRobotMover, self).__init__(model_name=model_name)

    def go_to_pose(self, x, y, orientation):
        current_robot_pose = self.get_position(self.model_name, "")
        new_state = ModelState()
        new_state.model_name = self.model_name
        new_state.pose = current_robot_pose.pose
        new_state.scale = current_robot_pose.scale
        new_state.twist = current_robot_pose.twist
        new_state.pose.position.x = x
        new_state.pose.position.y = y
        new_state.pose.orientation.x = orientation[1]
        new_state.pose.orientation.y = orientation[2]
        new_state.pose.orientation.z = orientation[3]
        new_state.pose.orientation.w = orientation[0]
        new_state.reference_frame = "world"
        self.set_position(new_state)


class ChallengeInteractor(object):

    def __init__(self):
        self.show_correct_mug = rospy.ServiceProxy(tc.thimblerigger_show_correct_service, Trigger)
        self.hide_correct_mug = rospy.ServiceProxy(tc.thimblerigger_hide_correct_service, Trigger)
        self.shuffle = rospy.ServiceProxy(tc.thimblerigger_shuffle_service, Trigger)

class Solver(object):

    def __init__(self):

        self.estimate = [False] * 3
        self.center_points = []
        self.robot_mover = TeleportRobotMover()
        self.set_joints_start()
        self.current_view = "front"
        self.challenge = ChallengeInteractor()
        self.view = rospy.Subscriber("/icub_model/left_eye_camera/image_raw", sensor_msgs.msg.Image, self.extract_mugs)


    def set_joints_start(self):
        neck = rospy.Publisher("/robot/neck_pitch/pos", std_msgs.msg.Float64, queue_size=1)
        l_elbow = rospy.Publisher("/robot/l_elbow/pos", std_msgs.msg.Float64, queue_size=1)
        r_elbow = rospy.Publisher("/robot/r_elbow/pos", std_msgs.msg.Float64, queue_size=1)
        def lower_gaze_and_arms():
            while not rospy.is_shutdown():
                neck.publish(std_msgs.msg.Float64(-0.8))
                l_elbow.publish(std_msgs.msg.Float64(-25.0))
                r_elbow.publish(std_msgs.msg.Float64(-25.0))
        start_new_thread(lower_gaze_and_arms, ())

    def front_look(self):
        self.current_view = "front"
        self.robot_mover.go_to_pose(x=-0.75, y=0., orientation=(0, 0, 1, 0))

    def side_look(self):
        self.current_view = "side"
        orientation = (math.sqrt(0.5), 0, 0, -math.sqrt(0.5))
        self.robot_mover.go_to_pose(x=0.4, y=-0.7, orientation=orientation)

    def extract_mugs(self, img_msg, contour_thresh=50):
        img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")
        most_vibrant_channel = np.argmax(img, axis=2)
        img[most_vibrant_channel != 2] = 0
        img[img[:,:,2] < 150] = 0
        red = img[:, :, 2]
        _, thresh = cv2.threshold(red, 150, 255, 0)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.erode(thresh, kernel,iterations = 1)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > contour_thresh]

        moments = [cv2.moments(c) for c in contours]
        centers = sorted([(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for M in moments])
        self.center_points = np.array(centers)
        for center in centers:
            cv2.circle(img, center, radius=1, color=(255,0,0), thickness=3 )
        cv2.drawContours(img, contours, -1, (0,255,0), 3)
        cv2.imshow("Red", img)
        cv2.waitKey(1)

    def find_correct_mug_beginning(self):
        self.side_look()
        self.challenge.show_correct_mug()
        correct_mug_id = np.argmin(np.transpose(self.center_points), axis=1)[1]
        self.estimate[correct_mug_id] = True
        self.challenge.hide_correct_mug()

    def verify_guess(self):
        self.side_look()
        self.challenge.show_correct_mug()
        correct_mug_id = np.argmin(np.transpose(self.center_points), axis=1)[1]
        self.challenge.hide_correct_mug()
        return self.estimate.index(True) == correct_mug_id



if __name__ == "__main__":
    rospy.init_node('thimblerigger_solution')
    solver = Solver()
    solver.find_correct_mug_beginning()
    rospy.spin()
