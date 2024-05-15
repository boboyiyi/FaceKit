#pragma once
#include "config.h"

enum LandmarkType { LMK19, LMK29, LMK68, LMK98, LMK106, LMK1000 };

class FaceKit {
public:
    FaceKit();
    ~FaceKit();
    void CloseAllSessions();
    void Detection(const cv::Mat& img_bgr, std::vector<lite::types::Boxf>& face_boxes);
    std::vector<lite::types::Boxf> DetectionWithLandmark(const cv::Mat& img_bgr);
    lite::types::Landmarks Alignment(const cv::Mat& img_bgr, const LandmarkType& landmark_type);
private:
    Detector* detector;
    DetectorWL* detectorWL;
    Aligner19* aligner19;
    Aligner29* aligner29;
    Aligner68* aligner68;
    Aligner98* aligner98;
    Aligner106* aligner106;
    Aligner1000* aligner1000;
};