#include "ocr_engine.h"

#include "utils.h"

#include "plog/Log.h"
#include "plog/Record.h"

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/core/matx.inl.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <utility>

#include "json/json.hpp"

namespace
{

template <typename T>
T GetJValue(const nlohmann::json &json, const std::vector<std::string> &keys, const T &dft)
{
    try
    {
        const nlohmann::json *current = &json;
        for (const auto &key : keys)
        {
            if (current->contains(key))
            {
                current = &((*current)[key]);
            }
            else
            {
                PLOGW << "Failed to find key: " << key << ", use default value: " << dft;
                return dft;
            }
        }
        return current->get<T>();
    }
    catch (const std::exception &e)
    {
        PLOGW << "Failed to parsing json: " << e.what() << ", use default value: " << dft;
        return dft;
    }
}

// Merge horizontally-adjacent boxes that belong to the same text line.
// Used to fix over-segmentation on sparse/label-style text.
void MergeNearbyBoxes(std::vector<OCR::TextBox> &boxes)
{
    if (boxes.size() < 2)
        return;

    // compute average box height for thresholds
    float avg_height = 0.0f;
    for (const auto &box : boxes)
    {
        auto [t, b] = std::minmax({box.points[0].y, box.points[1].y,
                                   box.points[2].y, box.points[3].y});
        avg_height += static_cast<float>(b - t);
    }
    avg_height /= boxes.size();
    if (avg_height < 1.0f)
        return;

    const float y_gap = avg_height * 0.6f;  // same-line Y tolerance
    const float x_gap = avg_height * 2.0f;  // horizontal merge distance

    // group into lines by Y proximity
    std::vector<std::vector<size_t>> lines;
    std::vector<bool> grouped(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (grouped[i])
            continue;
        grouped[i] = true;

        int i_min_y = std::min({boxes[i].points[0].y, boxes[i].points[1].y,
                                boxes[i].points[2].y, boxes[i].points[3].y});
        int i_max_y = std::max({boxes[i].points[0].y, boxes[i].points[1].y,
                                boxes[i].points[2].y, boxes[i].points[3].y});

        std::vector<size_t> line{ i };

        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (grouped[j])
                continue;

            int j_min_y = std::min({boxes[j].points[0].y, boxes[j].points[1].y,
                                    boxes[j].points[2].y, boxes[j].points[3].y});
            int j_max_y = std::max({boxes[j].points[0].y, boxes[j].points[1].y,
                                    boxes[j].points[2].y, boxes[j].points[3].y});

            // Y ranges overlap or are close → same line
            if (std::abs(i_min_y - j_min_y) <= y_gap ||
                std::abs(i_max_y - j_max_y) <= y_gap)
            {
                line.push_back(j);
                grouped[j] = true;
            }
        }

        // sort line left-to-right
        std::sort(line.begin(), line.end(),
            [&boxes](size_t a, size_t b) {
                int ax = std::min({boxes[a].points[0].x, boxes[a].points[1].x,
                                   boxes[a].points[2].x, boxes[a].points[3].x});
                int bx = std::min({boxes[b].points[0].x, boxes[b].points[1].x,
                                   boxes[b].points[2].x, boxes[b].points[3].x});
                return ax < bx;
            });

        lines.push_back(std::move(line));
    }

    // merge close boxes within each line
    std::vector<OCR::TextBox> merged;
    for (auto &line : lines)
    {
        if (line.empty())
            continue;

        OCR::TextBox cur = boxes[line[0]];
        for (size_t k = 1; k < line.size(); ++k)
        {
            const auto &next = boxes[line[k]];

            int cur_max_x = std::max({cur.points[0].x, cur.points[1].x,
                                      cur.points[2].x, cur.points[3].x});
            int next_min_x = std::min({next.points[0].x, next.points[1].x,
                                       next.points[2].x, next.points[3].x});

            if (next_min_x - cur_max_x <= x_gap)
            {
                // merge: expand cur's bounding box
                int x_min = std::min({cur.points[0].x, cur.points[1].x,
                                      cur.points[2].x, cur.points[3].x,
                                      next.points[0].x, next.points[1].x,
                                      next.points[2].x, next.points[3].x});
                int x_max = std::max({cur.points[0].x, cur.points[1].x,
                                      cur.points[2].x, cur.points[3].x,
                                      next.points[0].x, next.points[1].x,
                                      next.points[2].x, next.points[3].x});
                int y_min = std::min({cur.points[0].y, cur.points[1].y,
                                      cur.points[2].y, cur.points[3].y,
                                      next.points[0].y, next.points[1].y,
                                      next.points[2].y, next.points[3].y});
                int y_max = std::max({cur.points[0].y, cur.points[1].y,
                                      cur.points[2].y, cur.points[3].y,
                                      next.points[0].y, next.points[1].y,
                                      next.points[2].y, next.points[3].y});
                cur.points = {
                    cv::Point{x_min, y_min},
                    cv::Point{x_max, y_min},
                    cv::Point{x_max, y_max},
                    cv::Point{x_min, y_max}
                };
                cur.score = std::max(cur.score, next.score);
            }
            else
            {
                merged.push_back(cur);
                cur = next;
            }
        }
        merged.push_back(cur);
    }

    boxes = std::move(merged);
}

}   // unnamed namespace

namespace OCR
{

OCREngine::OCREngine(const std::string &config_path)
{
    Initialize(config_path);
}

OCREngine::OCREngine(OCREngine &&other) noexcept
    : config_(std::exchange(other.config_, {}))
    , det_net_(std::move(other.det_net_))
    , cls_net_(std::move(other.cls_net_))
    , rec_net_(std::move(other.rec_net_))
{

}

OCREngine & OCREngine::operator = (OCREngine &&other) noexcept
{
    if (this != &other)
    {
        config_ = std::exchange(other.config_, {});
        det_net_ = std::move(other.det_net_);
        cls_net_ = std::move(other.cls_net_);
        rec_net_ = std::move(other.rec_net_);
    }
    return *this;
}

bool OCREngine::Initialize(const std::string &config_path)
{
    // load json
    nlohmann::json j{};
    try
    {
        std::ifstream config_file(config_path);
        j = nlohmann::json::parse(config_file, nullptr, true, true);
    }
    catch(const nlohmann::json::exception &e)
    {
        PLOGE << "Failed to read JSON config from " << config_path << ": " << e.what();
        return false;
    }

    // read config
    config_.is_save = GetJValue(j, {"save"}, false);

    DetConfig &det_config = config_.det_config;
    det_config.infer_threads = GetThreads(GetJValue(j, {"det", "infer_threads"}, 1));
    det_config.model_path = GetJValue(j, {"det", "model_path"}, std::string());
    det_config.padding = GetJValue(j, {"det","padding"}, 50);
    det_config.max_side_len = GetJValue(j, {"det","max_side_len"}, 1024);
    det_config.min_side_len = GetJValue(j, {"det","min_side_len"}, 320);
    det_config.box_thres = GetJValue(j, {"det", "box_thres"}, 0.4f);
    det_config.bitmap_thres = GetJValue(j, {"det", "bitmap_thres"}, 0.3f);
    det_config.unclip_ratio = GetJValue(j, {"det", "unclip_ratio"}, 1.6f);
    det_config.is_fp16 = GetJValue(j, {"det", "fp16"}, false);

    ClsConfig &cls_config = config_.cls_config;
    cls_config.infer_threads = GetThreads(GetJValue(j, {"cls", "infer_threads"}, 1));
    cls_config.reco_threads = GetThreads(GetJValue(j, {"cls", "reco_threads"}, 1));
    cls_config.model_path = GetJValue(j, {"cls", "model_path"}, std::string());
    cls_config.enable = GetJValue(j, {"cls", "enable"}, true);
    cls_config.most_angle = GetJValue(j, {"cls", "most_angle"}, true);
    cls_config.is_fp16 = GetJValue(j, {"cls", "fp16"}, false);

    RecConfig &rec_config = config_.rec_config;
    rec_config.infer_threads = GetThreads(GetJValue(j, {"rec", "infer_threads"}, 1));
    rec_config.reco_threads = GetThreads(GetJValue(j, {"rec", "reco_threads"}, 1));
    rec_config.model_path = GetJValue(j, {"rec", "model_path"}, std::string());
    rec_config.keys_path = GetJValue(j, {"rec", "keys_path"}, std::string());
    rec_config.is_fp16 = GetJValue(j, {"rec", "fp16"}, false);

    // show configs
    ShowConfig();

    // create nets
    det_net_ = std::make_unique<DBNet>();
    cls_net_ = std::make_unique<AngleNet>();
    rec_net_ = std::make_unique<CRNNNet>();

    if (!det_net_->Initialize(det_config) ||
        !cls_net_->Initialize(cls_config) ||
        !rec_net_->Initialize(rec_config))
    {
        det_net_.reset();
        cls_net_.reset();
        rec_net_.reset();
        return false;
    }

    return true;
}

std::vector<OCRResult> OCREngine::Run(const cv::Mat &image) const
{
    if (!det_net_ || !cls_net_ || !rec_net_)
    {
        PLOGW << "Return an empty result since ( "
            << (!det_net_ ? "det_net " : "")
            << (!cls_net_ ? "cls_net " : "")
            << (!rec_net_ ? "rec_net " : "")
            << ") == nullptr";
        return {};
    }

    // timers
    double det_time{}, cls_time{}, rec_time{}, total_time{};

    // 1. Text Detection
    total_time = det_time = static_cast<double>(cv::getTickCount());

    auto text_boxes = det_net_->Det(image);

    det_time = (cv::getTickCount() - det_time) / cv::getTickFrequency() * 1000.0;

    // merge over-segmented boxes on the same line
    MergeNearbyBoxes(text_boxes);

    // rotate and crop images
    std::vector<cv::Mat> text_images(text_boxes.size());
    for (size_t i = 0; i < text_boxes.size(); ++i)
    {
        text_images[i] = GetRotatedCropImage(image, text_boxes[i].points);
    }

    // 2. Handle Angle
    cls_time = static_cast<double>(cv::getTickCount());

    auto angles = cls_net_->Cls(text_images);

    cls_time = (cv::getTickCount() - cls_time) / cv::getTickFrequency() * 1000.0;

    // rotate images
    for (size_t i = 0; i < text_images.size(); ++i)
    {
        if (angles[i].is_rot)
            cv::rotate(text_images[i], text_images[i], cv::ROTATE_180);
    }

    // 3. Recognize Text
    rec_time = static_cast<double>(cv::getTickCount());

    auto text_lines = rec_net_->Rec(text_images);

    rec_time = (cv::getTickCount() - rec_time) / cv::getTickFrequency() * 1000.0;

    std::vector<OCRResult> results(text_lines.size());
    for (size_t i = 0; i < text_lines.size(); ++i)
    {
        results[i].line = text_lines[i];
        results[i].angle = angles[i];
        results[i].box = text_boxes[i];
    }

    // timer
    total_time = (cv::getTickCount() - total_time) / cv::getTickFrequency() * 1000.0;
    PLOGI.printf("det_time(%.2fms), cls_time(%.2fms), rec_time(%.2fms), total(%.2fms)",
        det_time, cls_time, rec_time, total_time);

    // save results for debugging
    SaveResults(image, text_boxes, text_images, results);

    return results;
}

void OCREngine::ShowConfig() const
{
    const DetConfig &det_config = config_.det_config;
    const ClsConfig &cls_config = config_.cls_config;
    const RecConfig &rec_config = config_.rec_config;

    PLOGD << "--------------- Configs ---------------";

    PLOGD << "Det config";
    PLOGD.printf("  infer_threads(%d) padding(%d) max_side_len(%d) min_side_len(%d) box_thres(%.2f) "
        "bitmap_thres(%.2f) unclip_ratio(%.2f) fp16(%d)",
        det_config.infer_threads, det_config.padding, det_config.max_side_len,
        det_config.min_side_len,
        det_config.box_thres, det_config.bitmap_thres, det_config.unclip_ratio, det_config.is_fp16);

    PLOGD << "Cls config";
    PLOGD.printf("  infer_threads(%d) reco_threads(%d) enable(%d) most_angle(%d) fp16(%d)",
        cls_config.infer_threads, cls_config.reco_threads, cls_config.enable, cls_config.most_angle, cls_config.is_fp16);

    PLOGD << "Rec config";
    PLOGD.printf("  infer_threads(%d) reco_threads(%d) fp16(%d)",
        rec_config.infer_threads, rec_config.reco_threads, rec_config.is_fp16);

    PLOGD << "---------------------------------------";
}

void OCREngine::SaveResults(const cv::Mat &image, std::vector<TextBox> &text_boxes,
    std::vector<cv::Mat> &text_images, std::vector<OCRResult> &results,
    const std::string folder_name) const
{
    if (config_.is_save)
    {
        // create results folder
        std::filesystem::create_directories(folder_name);

        // save det results
        cv::Mat det_image = image.clone();
        for (size_t i = 0; i < text_boxes.size(); ++i)
            cv::polylines(det_image, text_boxes[i].points, true, cv::Scalar(255.0, 0.0, 0.0), 2);
        cv::imwrite(folder_name + "/det.jpg", det_image);

        // print boxes
        for (size_t i = 0; i < results.size(); ++i)
        {
            auto &box = text_boxes[i];
            auto &angle = results[i].angle;
            PLOGD.printf("Box[%zu] (%d, %d) (%d, %d) (%d, %d) (%d, %d) score: %.2f | Rotate: %d, score: %.2f",
                i,
                box.points[0].x, box.points[0].y, box.points[1].x, box.points[1].y,
                box.points[2].x, box.points[2].y, box.points[3].x, box.points[3].y,
                box.score * 100.0f, angle.is_rot, angle.score * 100.0f
            );
        }

        // save rec inputs
        for (size_t i = 0; i < text_images.size(); ++i)
            cv::imwrite(folder_name + std::string("/text") + std::to_string(i) + std::string(".jpg"), text_images[i]);

        PLOGI << "Results saved to ./" << folder_name;
    }
}

}   // namespace OCR