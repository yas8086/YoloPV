// YoloPV C++ ONNX 推理示例
// 用法：
//   ./yolopv_infer                                  # 默认推理 ../../yolov8/datasets/images/test
//   ./yolopv_infer --model a.onnx --source img.jpg  # 自定义
//   ./yolopv_infer --cpu                            # 强制 CPU

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "inference.h"

namespace fs = std::filesystem;

struct Args {
    std::string model = "../../runs/detect/train/weights/best.onnx";
    std::string source = "../../yolov8/datasets/images/test";
    int imgsz = 640;
    bool cuda = true;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    static struct option long_opts[] = {
        {"model",  required_argument, 0, 'm'},
        {"source", required_argument, 0, 's'},
        {"imgsz",  required_argument, 0, 'i'},
        {"cpu",    no_argument,       0, 'c'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    int c, idx;
    while ((c = getopt_long(argc, argv, "m:s:i:ch", long_opts, &idx)) != -1) {
        switch (c) {
            case 'm': a.model = optarg; break;
            case 's': a.source = optarg; break;
            case 'i': a.imgsz = std::stoi(optarg); break;
            case 'c': a.cuda = false; break;
            case 'h':
            default:
                std::cout << "用法: " << argv[0]
                          << " [--model onnx] [--source img/dir] [--imgsz N] [--cpu]\n";
                exit(0);
        }
    }
    return a;
}

int main(int argc, char** argv)
{
    Args args = parse_args(argc, argv);

    if (!fs::exists(args.model)) {
        std::cerr << "[错误] 模型不存在：" << args.model
                  << "（请先运行 `python YoloPV.py export`）\n";
        return 1;
    }

    Inference inf(args.model, cv::Size(args.imgsz, args.imgsz), "", args.cuda);

    // 收集图像
    std::vector<std::string> imagePaths;
    fs::path src(args.source);
    if (fs::is_regular_file(src)) {
        imagePaths.push_back(args.source);
    } else {
        for (auto& e : fs::directory_iterator(src)) {
            auto p = e.path();
            if (p.extension() == ".jpg" || p.extension() == ".jpeg")
                imagePaths.push_back(p.string());
        }
    }
    std::sort(imagePaths.begin(), imagePaths.end());

    fs::create_directories("output");
    int total = 0;

    for (const auto& path : imagePaths) {
        cv::Mat frame = cv::imread(path);
        if (frame.empty()) {
            std::cerr << "[跳过] 无法读取：" << path << "\n";
            continue;
        }

        std::vector<Detection> output = inf.runInference(frame);
        std::cout << path << ": " << output.size() << " 个目标\n";

        for (const auto& det : output) {
            cv::rectangle(frame, det.box, det.color, 2);
            std::string label = det.className + ' '
                + std::to_string(det.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.6, 2, 0);
            cv::Rect textBox(det.box.x, det.box.y - 25, textSize.width + 10, textSize.height + 10);
            cv::rectangle(frame, textBox, det.color, cv::FILLED);
            cv::putText(frame, label, cv::Point(det.box.x + 5, det.box.y - 8),
                        cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(0, 0, 0), 2, 0);
            total++;
        }

        fs::path outPath = fs::path("output") / ("det_" + fs::path(path).filename().string());
        cv::imwrite(outPath.string(), frame);
    }

    std::cout << "\n完成：" << imagePaths.size() << " 张图，共 " << total
              << " 个目标，结果保存到 output/\n";
    return 0;
}
