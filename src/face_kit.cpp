#include "face_kit.h"

FaceKit::FaceKit() {
    detector = nullptr;
    detectorWL = nullptr;
    aligner19 = nullptr;
    aligner29 = nullptr;
    aligner68 = nullptr;
    aligner98 = nullptr;
    aligner106 = nullptr;
    aligner1000 = nullptr;
}

FaceKit::~FaceKit() {
    CloseAllSessions();
}

void FaceKit::CloseAllSessions() {
    if (detector) { delete detector; detector = nullptr; }
    if (detectorWL) { delete detectorWL; detectorWL = nullptr; }
    if (aligner19) { delete aligner19; aligner19 = nullptr; }
    if (aligner29) { delete aligner29; aligner29 = nullptr; }
    if (aligner68) { delete aligner68; aligner68 = nullptr; }
    if (aligner98) { delete aligner98; aligner98 = nullptr; }
    if (aligner106) { delete aligner106; aligner106 = nullptr; }
    if (aligner1000) { delete aligner1000; aligner1000 = nullptr; }
}

void FaceKit::Detection(const cv::Mat& img_bgr, std::vector<lite::types::Boxf>& face_boxes) {
    std::vector<lite::types::Boxf> detected_boxes;
    detector->detect(img_bgr, detected_boxes);
}

std::vector<lite::types::Boxf> FaceKit::DetectionWithLandmark(const cv::Mat& img_bgr) {
    if (!detectorWL) {
        std::string detectorWL_model_path = model_folder + "/" + v_detectorWL_model_name[detectorWL_model_name_index] + ".onnx";
        detectorWL = new DetectorWL(detectorWL_model_path);
    }
    std::vector<lite::types::Boxf> face_boxes;
    std::vector<lite::types::BoxfWithLandmarks> detected_boxes_with_lanamarks;
    detectorWL->detect(img_bgr, detected_boxes_with_lanamarks);
    face_boxes.resize(detected_boxes_with_lanamarks.size());
    for (int i = 0; i < face_boxes.size(); i++) face_boxes[i] = detected_boxes_with_lanamarks[i].box;
    return face_boxes;
}

lite::types::Landmarks FaceKit::Alignment(const cv::Mat& img_bgr, const LandmarkType& landmark_type) {
    lite::types::Landmarks landmarks;
    if (landmark_type == LMK19) {

    } else if (landmark_type == LMK29) {

    } else if (landmark_type == LMK68) {
        if (!aligner68) {
            std::string aligner68_model_path = model_folder + "/" + v_aligner68_model_name[aligner68_model_name_index] + ".onnx";
            aligner68 = new Aligner68(aligner68_model_path);
        }
        aligner68->detect(img_bgr, landmarks);
    } else if (landmark_type == LMK98) {
        if (!aligner98) {
            std::string aligner98_model_path = model_folder + "/" + v_aligner98_model_name[aligner98_model_name_index] + ".onnx";
            aligner98 = new Aligner98(aligner98_model_path);
        }
        aligner98->detect(img_bgr, landmarks);
    } else if (landmark_type == LMK106) {

    } else if (landmark_type == LMK1000) {

    }
    return landmarks;
}