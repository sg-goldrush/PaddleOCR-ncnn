#include "db_net.h"

#include "common.h"
#include "config.h"
#include "utils.h"

#include "plog/Log.h"
#include "plog/Record.h"

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/core/matx.inl.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include <mat.h>
#include <net.h>
#include <option.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace OCR
{

DBNet::DBNet(const DetConfig &config)
{
    Initialize(config);
}

DBNet::DBNet(DBNet &&other) noexcept
    : config_(std::exchange(other.config_, {}))
    , net_(std::move(other.net_))
{

}

DBNet & DBNet::operator = (DBNet &&other) noexcept
{
    if (this != &other)
    {
        config_ = std::exchange(other.config_, {});
        net_ = std::move(other.net_);
    }
    return *this;
}

bool DBNet::Initialize(const DetConfig &config)
{
    config_ = config;

    // create net
    net_ = std::make_unique<ncnn::Net>();
    net_->opt.num_threads = config_.infer_threads;
    net_->opt.use_fp16_packed = config_.is_fp16;
    net_->opt.use_fp16_storage = config_.is_fp16;
    net_->opt.use_fp16_arithmetic = config_.is_fp16;

    if (net_->load_param((config_.model_path + ".param").c_str()) ||
        net_->load_model((config_.model_path + ".bin").c_str()))
    {
        PLOGE << "Failed to load model " << config_.model_path;
        net_.reset();
        return false;
    }

    return true;
}

std::vector<TextBox> DBNet::Det(const cv::Mat &image) const
{
    // padding
    const int padding = config_.padding;
    cv::Mat pad_image;
    cv::copyMakeBorder(image, pad_image, padding, padding, padding, padding,
        cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(255.0, 255.0, 255.0));

    // resize
    const int img_rows = pad_image.rows, img_cols = pad_image.cols;
    const int max_side = std::max(img_rows, img_cols);
    const int min_side = std::min(img_rows, img_cols);

    // Compute ratio constrained by max_side_len
    const int target_max = std::min(config_.max_side_len + 2 * padding, max_side);
    float ratio = static_cast<float>(target_max) / max_side;

    // Ensure the short side is not compressed too much (but don't upscale)
    if (static_cast<int>(min_side * ratio) < config_.min_side_len)
    {
        ratio = std::min(1.0f, static_cast<float>(config_.min_side_len) / min_side);
        PLOGD.printf("min_side_len (%d) override ratio: %.4f -> %.4f",
            config_.min_side_len, static_cast<float>(target_max) / max_side, ratio);
    }

    int rsz_rows = std::max(static_cast<int>(img_rows * ratio) / target_stride_ * target_stride_, target_stride_);
    int rsz_cols = std::max(static_cast<int>(img_cols * ratio) / target_stride_ * target_stride_, target_stride_);
    float ratio_rows = static_cast<float>(rsz_rows) / img_rows;
    float ratio_cols = static_cast<float>(rsz_cols) / img_cols;

    PLOGD.printf("src_w(%d), src_h(%d), dst_w(%d), dst_h(%d), ratio_w(%f), ratio_h(%f)",
        img_cols, img_rows, rsz_cols, rsz_rows, ratio_cols, ratio_rows);

    ncnn::Mat blob = ncnn::Mat::from_pixels_resize(
        pad_image.data, ncnn::Mat::PIXEL_BGR, img_cols, img_rows, rsz_cols, rsz_rows);
    blob.substract_mean_normalize(mean_values_, norm_values_);

    // inference
    ncnn::Extractor ex = net_->create_extractor();
    ex.input("input", blob);
    ncnn::Mat out;
    ex.extract("output", out);

    // binarization
    const float denorm_values[1] = {255.0f};
    out.substract_mean_normalize(0, denorm_values);

    cv::Mat pred(out.h, out.w, CV_8UC1);
    out.to_pixels(pred.data, ncnn::Mat::PIXEL_GRAY);
    cv::Mat bitmap = pred > static_cast<uint8_t>(config_.bitmap_thres * 255.0f);

    // get boxes from bitmap
    auto text_boxes = FindBoxesFromBitmap(pred, bitmap, img_rows, img_cols, ratio_rows, ratio_cols);

    return text_boxes;
}

std::vector<TextBox> DBNet::FindBoxesFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap,
    const int img_rows, const int img_cols, const float ratio_rows, const float ratio_cols) const
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    size_t num_contours = std::min(contours.size(), max_candidates_);

    std::vector<TextBox> text_boxes;
    for (size_t i = 0; i < num_contours; ++i)
    {
        if (contours[i].size() <= 2)
            continue;

        cv::RotatedRect min_area_rect = cv::minAreaRect(contours[i]);

        int long_side;
        auto min_boxes = GetMinBoxes(min_area_rect, long_side);
        if (long_side < min_size_)
            continue;

        float box_score = BoxScoreFast(min_boxes, pred);
        if (box_score < config_.box_thres)
            continue;

        // unclip
        cv::RotatedRect unclip_rect = Unclip(min_boxes, config_.unclip_ratio);
        if (unclip_rect.size.height <= 1.0f || unclip_rect.size.width <= 1.0f)
            continue;

        min_boxes = GetMinBoxes(unclip_rect, long_side);
        if (long_side < min_size_ + 2)
            continue;

        std::vector<cv::Point> text_points;
        for (size_t j = 0; j < min_boxes.size(); ++j)
        {
            int x = std::clamp(static_cast<int>(min_boxes[j].x / ratio_cols) - config_.padding,
                0, img_cols - 2 * config_.padding - 1);
            int y = std::clamp(static_cast<int>(min_boxes[j].y / ratio_rows) - config_.padding,
                0, img_rows - 2 * config_.padding - 1);
            text_points.emplace_back(cv::Point{x, y});
        }
        text_boxes.emplace_back(TextBox{std::move(text_points), box_score});
    }
    // sort by reading order: top-to-bottom, then left-to-right
    std::sort(text_boxes.begin(), text_boxes.end(),
        [](const TextBox &a, const TextBox &b) {
            int ay = std::min({a.points[0].y, a.points[1].y, a.points[2].y, a.points[3].y});
            int by = std::min({b.points[0].y, b.points[1].y, b.points[2].y, b.points[3].y});
            int ty = ay - by;
            if (ty != 0) return ty < 0;
            int ax = std::min({a.points[0].x, a.points[1].x, a.points[2].x, a.points[3].x});
            int bx = std::min({b.points[0].x, b.points[1].x, b.points[2].x, b.points[3].x});
            return ax < bx;
        });

    return text_boxes;
}

}   // namespace OCR