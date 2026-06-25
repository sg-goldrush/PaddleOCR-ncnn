#include "ocr_bridge.h"
#include "ocr_engine.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstring>

OCR_HANDLE OCR_Init(const char* config_path)
{
    auto* engine = new (std::nothrow) OCR::OCREngine();
    if (!engine) return nullptr;

    if (!engine->Initialize(config_path))
    {
        delete engine;
        return nullptr;
    }
    return static_cast<OCR_HANDLE>(engine);
}

static void FillResult(const std::vector<OCR::OCRResult>& ocr_results,
                       OCR_Result* out)
{
    std::memset(out, 0, sizeof(OCR_Result));

    int n = static_cast<int>(ocr_results.size());
    if (n > OCR_MAX_BOXES) n = OCR_MAX_BOXES;
    out->count = n;

    for (int i = 0; i < n; ++i)
    {
        const auto& src = ocr_results[i];
        OCR_Line& dst = out->lines[i];

        // text
        int max_len = OCR_MAX_TEXT_LEN - 1;
        dst.text_len = static_cast<int>(src.line.text.size());
        if (dst.text_len > max_len) dst.text_len = max_len;
        std::memcpy(dst.text, src.line.text.c_str(), dst.text_len);
        dst.text[dst.text_len] = '\0';

        // character scores
        int slen = static_cast<int>(src.line.scores.size());
        if (slen > OCR_MAX_TEXT_LEN) slen = OCR_MAX_TEXT_LEN;
        for (int j = 0; j < slen; ++j)
            dst.text_scores[j] = src.line.scores[j];

        // angle
        dst.is_rotated = src.angle.is_rot ? 1 : 0;
        dst.cls_score  = src.angle.score;

        // 4-point box
        dst.box.x0 = src.box.points[0].x;
        dst.box.y0 = src.box.points[0].y;
        dst.box.x1 = src.box.points[1].x;
        dst.box.y1 = src.box.points[1].y;
        dst.box.x2 = src.box.points[2].x;
        dst.box.y2 = src.box.points[2].y;
        dst.box.x3 = src.box.points[3].x;
        dst.box.y3 = src.box.points[3].y;
        dst.box.score = src.box.score;
    }
}

OCR_Result OCR_DetectFile(OCR_HANDLE handle, const char* image_path)
{
    OCR_Result result{};
    if (!handle || !image_path) return result;

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR_BGR);
    if (image.empty()) return result;

    auto* engine = static_cast<OCR::OCREngine*>(handle);
    auto ocr_results = engine->Run(image);

    FillResult(ocr_results, &result);
    return result;
}

OCR_Result OCR_DetectMemory(OCR_HANDLE handle,
                            const uint8_t* data,
                            int width, int height)
{
    OCR_Result result{};
    if (!handle || !data || width <= 0 || height <= 0) return result;

    // Wrap external BGR pixel data (not copied — caller must keep it alive
    // until this function returns)
    cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(data));

    auto* engine = static_cast<OCR::OCREngine*>(handle);
    auto ocr_results = engine->Run(image.clone());  // clone to be safe

    FillResult(ocr_results, &result);
    return result;
}

void OCR_Destroy(OCR_HANDLE handle)
{
    delete static_cast<OCR::OCREngine*>(handle);
}
