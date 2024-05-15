#include <iomanip>
#include <sys/stat.h>
#include <filesystem>
#include <sstream>
#include "face_kit.h"
#include "progressbar.hpp"
#include "npy.hpp"
#include "argparse.h"

/*
1. face detection bbox filter with max(0, x) ...
*/

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))
static const char* const usages[] = {
    "face_pipeline [options] [cmd] [args]\n\n"
    "cmds:\n"
    "  'detect_and_align' is used for face detection & alignment",
    nullptr,
};

struct cmd_struct {
    const char* cmd;
    int (*fn)(int, const char**);
};

int DetectAndAlign(int argc, const char** argv) {
    // create debug folder
    const char* debug_folder = ".debug";
    if (!std::filesystem::exists(debug_folder)) {
        if(!std::filesystem::create_directory(".debug")) {
            std::cerr << "Failed to create debug folder." << std::endl;
            return -1;
        }
    }
    const char* vid_path = nullptr;
    const char* npy_path = nullptr;
    const char* lmk_type = nullptr;
    const char* vis_path = nullptr;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_STRING('v', "vid_path", &vid_path, "video path to process", nullptr, 0, 0),
        OPT_STRING('l', "lmk_type", &lmk_type, "facial landmark type (lmk19, lmk29, lmk68, lmk98)", nullptr, 0, 0),
        OPT_STRING('d', "vis_path", &vis_path, "visualization file path", nullptr, 0, 0),
        OPT_STRING('n', "npy_path", &npy_path, "output npy file", nullptr, 0, 0),
        OPT_END(),
    };
    struct argparse argparse;
    argparse_init(&argparse, options, usages, 0);
    argc = argparse_parse(&argparse, argc, argv);
    if (nullptr == vid_path) {
        std::cerr << "Please ensure that you have specified video path." << std::endl;
        return -1;
    }
    std::vector<std::string> v_supported_lmk_type = { "lmk19", "lmk29", "lmk68", "lmk98" };
    if (nullptr == lmk_type || std::find(v_supported_lmk_type.begin(), v_supported_lmk_type.end(), lmk_type) == v_supported_lmk_type.end()) {
        std::cerr << "Please specify correct lmk_type." << std::endl;
        return -1;
    }

    struct stat flag;
    if (stat(vid_path, &flag) != 0 && !(flag.st_mode & S_IFDIR)) {
        std::cerr << "video doesn't exist." << std::endl;
        return -1;
    }

    bool store_visualization = false;
    if (nullptr != vis_path) {
        store_visualization = true;
    }

    FaceKit face_kit;
    cv::VideoCapture video_reader(vid_path);
    if (!video_reader.isOpened()) {
        std::cerr << "Failed to open " << vid_path << "." << std::endl;
        return -1;
    }
    int total_frames = video_reader.get(cv::CAP_PROP_FRAME_COUNT);
    int frame_width = video_reader.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = video_reader.get(cv::CAP_PROP_FRAME_HEIGHT);
    float fps = video_reader.get(cv::CAP_PROP_FPS);
    std::cerr << "Video name: " << vid_path << std::endl;
    std::cerr << "Video info: " 
              << total_frames << "(frames) | "
              << frame_width << "(width) | "
              << frame_height << "(height) | "
              << fps << "(fps)" << std::endl;

    cv::Mat frame;
    float box_scale = 1.2;
    /*------------------ face detection ------------------*/
    std::cerr << "Face detection: ";
    progressbar bar(total_frames);
    face_kit.CloseAllSessions();
    std::vector<std::vector<lite::types::Boxf> > vv_face_box(total_frames);
    for (int i = 0; i < total_frames; i++) {
        bar.update();
        video_reader >> frame;
        vv_face_box[i] = face_kit.DetectionWithLandmark(frame);
        if (vv_face_box[i].size() == 0) {
            // save frame which has no face
            std::ostringstream ss;
            ss << std::setw(6) << std::setfill('0') << i;
            std::string error_frame_name(ss.str());
            std::string error_frame_path = std::string(debug_folder) + "/" + error_frame_name + ".jpg";
            cv::imwrite(error_frame_path, frame);
            std::cerr << "Current frame: " << error_frame_name << " has no detected faces. Check " << error_frame_path << " for detail." << std::endl;
            return -1;
        }
    }
    std::cerr << std::endl;
    // preprocess face boxes
    std::vector<lite::types::Boxf> v_face_box(total_frames);
    for (int i = 0; i < total_frames; i++) {
        video_reader >> frame;
        float score = 0;
        int face_idx = 0;
        for (int j = 0; j < vv_face_box[i].size(); j++) {
            if (vv_face_box[i][j].score > score) {
                face_idx = j;
                score = vv_face_box[i][j].score;
            }
        }
        lite::types::Boxf face_box = vv_face_box[i][face_idx];
        face_box.x1 = int(face_box.x1);
        face_box.x2 = int(face_box.x2);
        face_box.y1 = int(face_box.y1);
        face_box.y2 = int(face_box.y2);
        float face_w = face_box.x2 - face_box.x1 + 1;
        float face_h = face_box.y2 - face_box.y1 + 1;
        face_box.x1 -= int(face_w * (box_scale - 1) / 2);
        face_box.x2 += int(face_w * (box_scale - 1) / 2);
        face_box.y1 += int(face_h * (box_scale - 1) / 2);
        face_box.y2 += int(face_h * (box_scale - 1) / 2);
        face_box.x1 = std::max(int(face_box.x1), 0);
        face_box.y1 = std::max(int(face_box.y1), 0);
        face_box.x2 = std::min(int(face_box.x2), int(frame_width)- 1);
        face_box.y2 = std::min(int(face_box.y2), int(frame_height) - 1);
        v_face_box[i] = face_box;
        
    }

    /*------------------ face alignment ------------------*/
    std::cerr << "Face alignment: ";
    bar.reset();
    video_reader.set(cv::CAP_PROP_POS_FRAMES, 0);
    face_kit.CloseAllSessions();
    std::vector<lite::types::Landmarks> v_face_landmarks(total_frames);
    for (int i = 0; i < total_frames; i++) {
        bar.update();
        video_reader >> frame;
        cv::Mat face_img = frame(cv::Range(v_face_box[i].y1, v_face_box[i].y2), cv::Range(v_face_box[i].x1, v_face_box[i].x2));
        if (std::string(lmk_type) == "lmk19") {
        } else if (std::string(lmk_type) == "lmk29") {
        } else if (std::string(lmk_type) == "lmk68") {
            v_face_landmarks[i] = face_kit.Alignment(face_img, LMK68);
        } else if (std::string(lmk_type) == "lmk98") {
            v_face_landmarks[i] = face_kit.Alignment(face_img, LMK98);
        } else {
        }
        for (int j = 0; j < v_face_landmarks[i].points.size(); j++) {
            v_face_landmarks[i].points[j].x += v_face_box[i].x1;
            v_face_landmarks[i].points[j].y += v_face_box[i].y1;
        }
    }
    std::cerr << std::endl;

    if (store_visualization) {
        /*------------------ visualization ------------------*/
        std::cerr << "Visualization: ";
        cv::VideoWriter video_writer(vis_path, cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(frame_width, frame_height));
        bar.reset();
        video_reader.set(cv::CAP_PROP_POS_FRAMES, 0);
        for (int i = 0; i < total_frames; i++) {
            bar.update();
            video_reader >> frame;
            lite::utils::draw_boxes_inplace(frame, std::vector<lite::types::Boxf>{v_face_box[i]});
            lite::utils::draw_landmarks_inplace(frame, v_face_landmarks[i]);
            video_writer.write(frame);
        }
        std::cerr << std::endl;
        video_writer.release();
        video_reader.release();
    }

    /*------------------ write npy ------------------*/
    std::cerr << "Save npy to: " << npy_path << std::endl;
    // box.x1, box.y1, box.x2, box.y2, box.score, lmk0.x, lmk0.y, ..., lmkN.x, lmkN.y
    int number_values = 5 + v_face_landmarks[0].points.size() * 2;
    std::vector<float> data(total_frames * number_values);
    for (int i = 0; i < total_frames; i++) {
        data[number_values * i + 0] = v_face_box[i].x1;
        data[number_values * i + 1] = v_face_box[i].y1;
        data[number_values * i + 2] = v_face_box[i].x2;
        data[number_values * i + 3] = v_face_box[i].y2;
        data[number_values * i + 4] = v_face_box[i].score;
        for (int j = 0; j < v_face_landmarks[i].points.size(); j++) {
            data[number_values * i + 5 + j * 2 + 0] = v_face_landmarks[i].points[j].x;
            data[number_values * i + 5 + j * 2 + 1] = v_face_landmarks[i].points[j].y;
        }
    }
    npy::npy_data_ptr<float> d;
    d.data_ptr = data.data();
    d.shape = { static_cast<unsigned long>(total_frames), static_cast<unsigned long>(number_values) };
    npy::write_npy(npy_path, d);
    return 0;
}

static struct cmd_struct commands[] = {
    {"detect_and_align", DetectAndAlign},
};

int main(int argc, const char** argv) {
    struct argparse argparse;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_END(),
    };
    argparse_init(&argparse, options, usages, ARGPARSE_STOP_AT_NON_OPTION);
    argc = argparse_parse(&argparse, argc, argv);
    if (argc < 1) {
        argparse_usage(&argparse);
        return -1;
    }
    struct cmd_struct* cmd = NULL;
    for (int i = 0; i < ARRAY_SIZE(commands); i++) {
        if (!strcmp(commands[i].cmd, argv[0])) {
            cmd = &commands[i];
        }
    }
    if (cmd) {
        return cmd->fn(argc, argv);
    }
    return 0;
}
