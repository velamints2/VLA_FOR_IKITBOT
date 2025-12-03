/*
 * RealSense .bag文件帧提取工具 (C++)
 * 使用librealsense库提取RGB和Depth帧
 * 
 * 编译: g++ -std=c++11 extract_bag_frames.cpp -lrealsense2 -lopencv_core -lopencv_imgcodecs -o extract_bag_frames
 * 
 * 使用: ./extract_bag_frames <bag_file> <output_dir> [sample_rate]
 */

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

// 创建目录（递归）
bool create_directory(const std::string& path) {
    std::string cmd = "mkdir -p " + path;
    return system(cmd.c_str()) == 0;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " <bag_file> <output_dir> [sample_rate]\n";
        std::cerr << "示例: " << argv[0] << " input.bag output_frames 5\n";
        return 1;
    }

    std::string bag_file = argv[1];
    std::string output_dir = argv[2];
    int sample_rate = (argc >= 4) ? std::stoi(argv[3]) : 2;

    std::cout << "=========================================================\n";
    std::cout << "RealSense .bag 文件帧提取工具\n";
    std::cout << "=========================================================\n";
    std::cout << "输入文件: " << bag_file << "\n";
    std::cout << "输出目录: " << output_dir << "\n";
    std::cout << "采样率: 每 " << sample_rate << " 帧提取一帧\n";
    std::cout << "=========================================================\n\n";

    // 创建输出目录
    std::string rgb_dir = output_dir + "/rgb";
    std::string depth_dir = output_dir + "/depth";
    
    if (!create_directory(rgb_dir) || !create_directory(depth_dir)) {
        std::cerr << "错误: 无法创建输出目录\n";
        return 1;
    }

    try {
        // 配置RealSense管道
        rs2::config cfg;
        cfg.enable_device_from_file(bag_file);
        
        // 启动管道
        rs2::pipeline pipe;
        rs2::pipeline_profile profile = pipe.start(cfg);
        
        std::cout << "✓ 成功打开 .bag 文件\n";
        std::cout << "正在提取帧...\n\n";

        int frame_count = 0;
        int saved_count = 0;

        // 主循环
        while (true) {
            try {
                // 等待帧集
                rs2::frameset frames = pipe.wait_for_frames(1000);
                
                // 按采样率提取
                if (frame_count % sample_rate == 0) {
                    // 获取RGB帧
                    rs2::video_frame color_frame = frames.get_color_frame();
                    // 获取深度帧
                    rs2::depth_frame depth_frame = frames.get_depth_frame();

                    if (color_frame && depth_frame) {
                        // RGB帧转OpenCV Mat
                        cv::Mat rgb_image(cv::Size(color_frame.get_width(), 
                                                   color_frame.get_height()),
                                         CV_8UC3,
                                         (void*)color_frame.get_data(),
                                         cv::Mat::AUTO_STEP);
                        
                        // Depth帧转OpenCV Mat
                        cv::Mat depth_image(cv::Size(depth_frame.get_width(),
                                                     depth_frame.get_height()),
                                           CV_16UC1,
                                           (void*)depth_frame.get_data(),
                                           cv::Mat::AUTO_STEP);

                        // 生成文件名
                        std::stringstream rgb_filename, depth_filename;
                        rgb_filename << rgb_dir << "/frame_" 
                                    << std::setw(6) << std::setfill('0') 
                                    << saved_count << ".jpg";
                        depth_filename << depth_dir << "/frame_"
                                      << std::setw(6) << std::setfill('0')
                                      << saved_count << ".png";

                        // 保存图像
                        cv::cvtColor(rgb_image, rgb_image, cv::COLOR_RGB2BGR);
                        cv::imwrite(rgb_filename.str(), rgb_image);
                        cv::imwrite(depth_filename.str(), depth_image);

                        saved_count++;

                        if (saved_count % 100 == 0) {
                            std::cout << "已提取 " << saved_count << " 帧...\n";
                        }
                    }
                }

                frame_count++;

            } catch (const rs2::error& e) {
                // 到达文件末尾
                if (std::string(e.what()).find("Frame didn't arrive") != std::string::npos) {
                    break;
                }
                throw;
            }
        }

        pipe.stop();

        std::cout << "\n=========================================================\n";
        std::cout << "✓ 提取完成！\n";
        std::cout << "=========================================================\n";
        std::cout << "总帧数: " << frame_count << "\n";
        std::cout << "已保存: " << saved_count << " 帧\n";
        std::cout << "RGB帧: " << rgb_dir << "\n";
        std::cout << "Depth帧: " << depth_dir << "\n";
        std::cout << "=========================================================\n";

        return 0;

    } catch (const rs2::error& e) {
        std::cerr << "RealSense错误: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
}
