#ifndef OCR_BRIDGE_H_
#define OCR_BRIDGE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// --- Opaque handle (hides C++ internals) ---
typedef void* OCR_HANDLE;

// --- Result structures (plain C, safe for FFI) ---
#define OCR_MAX_TEXT_LEN 512
#define OCR_MAX_BOXES    256

typedef struct
{
    int x0, y0;   // top-left
    int x1, y1;   // top-right
    int x2, y2;   // bottom-right
    int x3, y3;   // bottom-left
    float score;
} OCR_Box;

typedef struct
{
    char  text[OCR_MAX_TEXT_LEN];   // recognized text, UTF-8
    float text_scores[OCR_MAX_TEXT_LEN];
    int   text_len;
    int   is_rotated;               // 0=normal, 1=rotated 180°
    float cls_score;
    OCR_Box box;
} OCR_Line;

typedef struct
{
    OCR_Line lines[OCR_MAX_BOXES];
    int      count;
    double   det_time_ms;
    double   cls_time_ms;
    double   rec_time_ms;
} OCR_Result;

// --- API ---

/**
 * Initialize OCR engine from a config JSON file.
 * Returns NULL on failure.
 * Call once, reuse across images.
 */
OCR_HANDLE OCR_Init(const char* config_path);

/**
 * Run OCR on an image file (PNG / JPG / BMP etc.).
 * Returns the detection & recognition result.
 * Thread-safe to call from multiple threads on the same handle.
 */
OCR_Result OCR_DetectFile(OCR_HANDLE handle, const char* image_path);

/**
 * Run OCR on in-memory BGR pixel data.
 * width, height: image dimensions
 * data: 8-bit BGR interleaved pixels (3 * width * height bytes)
 */
OCR_Result OCR_DetectMemory(OCR_HANDLE handle,
                            const uint8_t* data,
                            int width, int height);

/**
 * Destroy the OCR engine and free all resources.
 */
void OCR_Destroy(OCR_HANDLE handle);

#ifdef __cplusplus
}
#endif

#endif  // OCR_BRIDGE_H_
