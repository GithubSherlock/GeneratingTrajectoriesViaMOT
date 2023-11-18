from Detector import *

detector = Detector(model_type="OD")

# detector.onImage("Detector_test/images/2.jpg") # /home/deep/PythonCode/Detector_test/image ./image

'''
    Video path form: "Detector_test/videos/sportscheck_cafe_1.mp4", "/home/deep/Videos/sportscheck_cafe_11.mp4"
    Video list of student thesis:
    hauptmensa_1.mp4
    sportscheck_cafe_1.mp4
    sportscheck_street_1.mp4
    sportscheck_street_23min.mp4
'''
detector.onVideo("Detector_test/videos/sportscheck_cafe_1.mp4", "/home/deep/Videos/sportscheck_cafe_1.mp4") #

# Test image not found or existS
# image = cv2.imread('Detector_test/image/1.jpg')
# assert not isinstance(image,type(None)), 'image not found'
