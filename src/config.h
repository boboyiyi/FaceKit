#include "lite/lite.h"

const int num_threads = 30;
const std::string model_folder = "/home/ubuntu/work/git/avatar/tool/share/cv";

/* modify below Macros for your task */
/* --------------------------detector-------------------------- */
// #define USING_UltraFace
// #define USING_RetinaFace
// #define USING_FaceBoxes
#define USING_RetinaFace
/* --------------------------detector_with_landmark-------------------------- */
#define USING_SCRFD
// #define USING_YOLO5Face
// #define USING_YOLOv5BlazeFace
/* --------------------------aligner19-------------------------- */
#define USING_PIPNet19
/* --------------------------aligner29-------------------------- */
#define USING_PIPNet29
/* --------------------------aligner68-------------------------- */
// #define USING_MobileNetV268
// #define USING_MobileNetV2SE68
// #define USING_PFLD68
#define USING_PIPNet68
/* --------------------------aligner98-------------------------- */
// #define USING_PFLD98
#define USING_PIPNet98
/* --------------------------aligner106-------------------------- */
#define USING_PFLD
/* --------------------------aligner1000-------------------------- */
#define USING_FaceLandmark1000

/* --------------------------detector-------------------------- */
#ifdef USING_UltraFace
using Detector = lite::cv::face::detect::UltraFace;
const std::vector<std::string> v_detector_model_name = {
    "ultraface-rfb-640", // 0
    "ultraface-rfb-320"  // 1
};
const int detector_model_name_index = 1;
#endif

#ifdef USING_RetinaFace
using Detector = lite::cv::face::detect::RetinaFace;
const std::vector<std::string> v_detector_model_name = {
    "Pytorch_RetinaFace_resnet50",             // 0
    "Pytorch_RetinaFace_resnet50-640-640",     // 1
    "Pytorch_RetinaFace_resnet50-320-320",     // 2
    "Pytorch_RetinaFace_resnet50-720-1080",    // 3
    "Pytorch_RetinaFace_mobile0.25",           // 4
    "Pytorch_RetinaFace_mobile0.25-640-640",   // 5
    "Pytorch_RetinaFace_mobile0.25-320-320",   // 6
    "Pytorch_RetinaFace_mobile0.25-720-1080"   // 7
};
const int detector_model_name_index = 0;
#endif

#ifdef USING_FaceBoxes
using Detector = lite::cv::face::detect::FaceBoxes;
const std::vector<std::string> v_detector_model_name = {
    "FaceBoxes",             // 0
    "FaceBoxes-640-640",     // 1
    "FaceBoxes-320-320",     // 2
    "FaceBoxes-720-1080"     // 3
};
const int detector_model_name_index = 0;
#endif

#ifdef USING_FaceBoxesV2
using Detector = lite::cv::face::detect::FaceBoxesV2;
const std::vector<std::string> v_detector_model_name = {
    "faceboxesv2-640x640"     // 0
};
const int detector_model_name_index = 0;
#endif

/* --------------------------detector_with_landmark-------------------------- */
#ifdef USING_SCRFD
using DetectorWL = lite::cv::face::detect::SCRFD;
const std::vector<std::string> v_detectorWL_model_name = {
    "scrfd_500m_shape160x160",             // 0
    "scrfd_500m_shape320x320",             // 1
    "scrfd_500m_shape640x640",             // 2
    "scrfd_500m_bnkps_shape160x160",       // 3
    "scrfd_500m_bnkps_shape320x320",       // 4
    "scrfd_500m_bnkps_shape640x640",       // 5
    "scrfd_1g_shape160x160",               // 6
    "scrfd_1g_shape320x320",               // 7
    "scrfd_1g_shape640x640",               // 8
    "scrfd_2.5g_shape160x160",             // 9
    "scrfd_2.5g_shape320x320",             // 10
    "scrfd_2.5g_shape640x640",             // 11
    "scrfd_2.5g_bnkps_shape160x160",       // 12
    "scrfd_2.5g_bnkps_shape320x320",       // 13
    "scrfd_2.5g_bnkps_shape640x640",       // 14
    "scrfd_10g_shape640x640",              // 15
    "scrfd_10g_shape1280x1280",            // 16
    "scrfd_10g_bnkps_shape640x640",        // 17
    "scrfd_10g_bnkps_shape1280x1280"       // 18
};
const int detectorWL_model_name_index = 11;
#endif

#ifdef USING_YOLO5Face
using DetectorWL = lite::cv::face::detect::YOLO5Face;
const std::vector<std::string> v_detectorWL_model_name = {
    "yolov5face-l-640x640",         // 0
    "yolov5face-m-640x640",         // 1
    "yolov5face-n-0.5-320x320",     // 2
    "yolov5face-n-0.5-640x640",     // 3
    "yolov5face-n-640x640",         // 4
    "yolov5face-s-640x640"          // 5

};
const int detectorWL_model_name_index = 3;
#endif

#ifdef USING_YOLOv5BlazeFace
using DetectorWL = lite::cv::face::detect::YOLOv5BlazeFace;
const std::vector<std::string> v_detectorWL_model_name = {
    "yolov5face-blazeface-640x640"     // 0
};
const int detectorWL_model_name_index = 0;
#endif

/* --------------------------aligner106-------------------------- */
#ifdef USING_PFLD
using Aligner106 = lite::cv::face::align::PFLD;
const std::vector<std::string> v_aligner106_model_name = {
    "pfld-106-lite", // 0
    "pfld-106-v3",   // 1
    "pfld-106-v2"    // 2
};
const int aligner106_model_name_index = 1;
#endif

/* --------------------------aligner98-------------------------- */
#ifdef USING_PFLD98
using Aligner98 = lite::cv::face::align::PFLD98;
const std::vector<std::string> v_aligner98_model_name = {
    "PFLD-pytorch-pfld" // 0
};
const int aligner98_model_name_index = 0;
#endif

#ifdef USING_PIPNet98
using Aligner98 = lite::cv::face::align::PIPNet98;
const std::vector<std::string> v_aligner98_model_name = {
    "pipnet_resnet18_10x98x32x256_wflw", // 0
    "pipnet_resnet101_10x98x32x256_wflw" // 1
};
const int aligner98_model_name_index = 0;
#endif

/* --------------------------aligner68-------------------------- */
#ifdef USING_MobileNetV268
using Aligner68 = lite::cv::face::align::MobileNetV268;
const std::vector<std::string> v_aligner68_model_name = {
    "pytorch_face_landmarks_landmark_detection_56" // 0
};
const int aligner68_model_name_index = 0;
#endif

#ifdef USING_MobileNetV2SE68
using Aligner68 = lite::cv::face::align::MobileNetV2SE68;
const std::vector<std::string> v_aligner68_model_name = {
    "pytorch_face_landmarks_landmark_detection_56_se_external" // 0
};
const int aligner68_model_name_index = 0;
#endif

#ifdef USING_PFLD68
using Aligner68 = lite::cv::face::align::PFLD68;
const std::vector<std::string> v_aligner68_model_name = {
    "pytorch_face_landmarks_pfld" // 0
};
const int aligner68_model_name_index = 0;
#endif

#ifdef USING_PIPNet68
using Aligner68 = lite::cv::face::align::PIPNet68;
const std::vector<std::string> v_aligner68_model_name = {
    "pipnet_resnet18_10x68x32x256_300w", // 0
    "pipnet_resnet101_10x68x32x256_300w"
};
const int aligner68_model_name_index = 0;
#endif

/* --------------------------aligner1000-------------------------- */
#ifdef USING_FaceLandmark1000
using Aligner1000 = lite::cv::face::align::FaceLandmark1000;
const std::vector<std::string> v_aligner1000_model_name = {
    "FaceLandmark1000"   // 0
};
const int aligner1000_model_name_index = 0;
#endif

/* --------------------------aligner29-------------------------- */
#ifdef USING_PIPNet29
using Aligner29 = lite::cv::face::align::PIPNet29;
const std::vector<std::string> v_aligner29_model_name = {
    "pipnet_resnet18_10x29x32x256_cofw",   // 0
    "pipnet_resnet101_10x29x32x256_cofw"   // 1
};
const int aligner29_model_name_index = 0;
#endif

/* --------------------------aligner19-------------------------- */
#ifdef USING_PIPNet19
using Aligner19 = lite::cv::face::align::PIPNet29;
const std::vector<std::string> v_aligner19_model_name = {
    "pipnet_resnet18_10x19x32x256_aflw",   // 0
    "pipnet_resnet18_10x19x32x256_aflw"   // 1
};
const int aligner19_model_name_index = 0;
#endif