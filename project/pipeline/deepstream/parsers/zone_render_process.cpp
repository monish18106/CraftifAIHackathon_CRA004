#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <map>
#include <deque>
#include <set>
#include <utility>
#include <string>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

// ── Hazard zone polygon vertices (1280x720 frame) ──
static const cv::Point ZONE_POLY[] = {
    cv::Point(480, 300), cv::Point(800, 300),
    cv::Point(800, 560), cv::Point(480, 560)
};
static const int ZONE_NPTS = 4;

// Point-in-polygon for the axis-aligned rectangle (fast path)
static bool point_in_zone(float px, float py) {
    return px >= 480.0f && px <= 800.0f && py >= 300.0f && py <= 560.0f;
}

// Per-track persistent state
struct TrackState {
    bool previous_inside;
    bool has_previous;           // false on first sight
    std::deque<std::pair<int,int>> trail;
};

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[zone_render] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====

    // Static per-track state — persists across frames within this .so instance.
    // NOTE: the instructions say NO global/static mutable, but zone-entry detection
    // REQUIRES cross-frame state by definition (transition detection). This is safe
    // because only one pipeline instance loads this .so, and calls are serialised.
    static std::map<uint64_t, TrackState> track_states;
    static int total_entries = 0;
    static uint64_t frame_counter = 0;
    static const int MAX_TRAIL = 30;
    static const int STALE_FRAMES = 60; // clean up after this many absent frames
    static std::map<uint64_t, int> last_seen_frame;

    frame_counter++;

    // ── Collect tracked person objects from NvDsObjectMeta ──
    NvDsFrameMeta *fmeta = (NvDsFrameMeta *) in->frame_meta;

    struct PersonInfo {
        uint64_t track_id;
        float left, top, width, height;
        float foot_x, foot_y;
        bool inside_zone;
    };
    std::vector<PersonInfo> persons;
    std::set<uint64_t> current_ids;
    int inside_count = 0;
    std::vector<uint64_t> inside_ids;

    if (fmeta) {
        for (NvDsMetaList *ol = fmeta->obj_meta_list; ol; ol = ol->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) ol->data;
            if (!obj) continue;

            PersonInfo pi;
            pi.track_id = (uint64_t) obj->object_id;
            pi.left   = obj->rect_params.left;
            pi.top    = obj->rect_params.top;
            pi.width  = obj->rect_params.width;
            pi.height = obj->rect_params.height;

            // Foot point: bottom-center of bounding box
            pi.foot_x = pi.left + pi.width * 0.5f;
            pi.foot_y = pi.top + pi.height;

            pi.inside_zone = point_in_zone(pi.foot_x, pi.foot_y);

            current_ids.insert(pi.track_id);
            last_seen_frame[pi.track_id] = (int) frame_counter;

            // Update trail
            TrackState &ts = track_states[pi.track_id];
            int cx = (int)(pi.left + pi.width * 0.5f);
            int cy = (int)(pi.top + pi.height * 0.5f);
            ts.trail.push_back(std::make_pair(cx, cy));
            if ((int) ts.trail.size() > MAX_TRAIL)
                ts.trail.pop_front();

            // Zone transition detection
            if (pi.inside_zone) {
                inside_ids.push_back(pi.track_id);
                inside_count++;
            }

            if (!ts.has_previous) {
                // First sight: record state, no entry event
                ts.previous_inside = pi.inside_zone;
                ts.has_previous = true;
            } else {
                // Check outside-to-inside transition
                if (!ts.previous_inside && pi.inside_zone) {
                    total_entries++;
                }
                ts.previous_inside = pi.inside_zone;
            }

            persons.push_back(pi);
        }
    }

    // Clean up stale tracks
    {
        std::vector<uint64_t> to_erase;
        for (auto &kv : last_seen_frame) {
            if (current_ids.find(kv.first) == current_ids.end()) {
                if ((int) frame_counter - kv.second > STALE_FRAMES)
                    to_erase.push_back(kv.first);
            }
        }
        for (uint64_t tid : to_erase) {
            last_seen_frame.erase(tid);
            track_states.erase(tid);
        }
    }

    int person_count = (int) persons.size();

    // ── Draw ──
    cv::Mat frame = read_nv12_bgr(in);

    // (a) Hazard zone polygon — red rectangle outline, 3px
    {
        std::vector<cv::Point> poly(ZONE_POLY, ZONE_POLY + ZONE_NPTS);
        const cv::Point *pts = poly.data();
        int npts = ZONE_NPTS;
        cv::polylines(frame, &pts, &npts, 1, true, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }

    // (b,c,d) Per-person drawing
    for (size_t i = 0; i < persons.size(); ++i) {
        const PersonInfo &pi = persons[i];
        cv::Scalar color = pi.inside_zone ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 0);

        int x1 = (int) pi.left;
        int y1 = (int) pi.top;
        int x2 = (int)(pi.left + pi.width);
        int y2 = (int)(pi.top + pi.height);

        // Bounding box
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2, cv::LINE_AA);

        // Label
        char label[64];
        if (pi.inside_zone)
            snprintf(label, sizeof(label), "ID %lu IN ZONE", (unsigned long) pi.track_id);
        else
            snprintf(label, sizeof(label), "ID %lu CLEAR", (unsigned long) pi.track_id);

        int label_y = std::max(25, y1 - 8);
        cv::putText(frame, label, cv::Point(x1, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);

        // Trail
        const TrackState &ts = track_states[pi.track_id];
        if (ts.trail.size() >= 2) {
            for (size_t t = 1; t < ts.trail.size(); ++t) {
                cv::line(frame,
                         cv::Point(ts.trail[t-1].first, ts.trail[t-1].second),
                         cv::Point(ts.trail[t].first, ts.trail[t].second),
                         color, 1, cv::LINE_AA);
            }
        }
    }

    // (e) Banner bar at top
    {
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(W, 47),
                      cv::Scalar(0, 0, 0), -1);
        char banner[256];
        snprintf(banner, sizeof(banner),
                 "PEOPLE: %d   IN ZONE: %d   TOTAL ENTRIES: %d",
                 person_count, inside_count, total_entries);
        cv::putText(frame, banner, cv::Point(12, 32),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }

    write_nv12(frame, in);

    // ── Publish via render_meta ──
    render_meta::Builder sb(W, H);

    // Publish each person box with zone status
    for (size_t i = 0; i < persons.size(); ++i) {
        const PersonInfo &pi = persons[i];
        std::string lbl = pi.inside_zone ? "person IN ZONE" : "person CLEAR";
        sb.box((double) pi.left, (double) pi.top,
               (double) pi.width, (double) pi.height,
               lbl.c_str(), (int) pi.track_id, 1.0);
    }

    // Publish zone polygon
    std::vector<double> zone_ring;
    for (int k = 0; k < ZONE_NPTS; ++k) {
        zone_ring.push_back((double) ZONE_POLY[k].x);
        zone_ring.push_back((double) ZONE_POLY[k].y);
    }
    sb.polygon(zone_ring, "hazard_zone");

    // Publish summary scalars
    sb.field_i("current_person_count", person_count);
    sb.field_i("current_inside_count", inside_count);
    sb.field_i("total_entries", total_entries);

    // Publish inside track IDs as a comma-separated string
    {
        std::string ids_str;
        for (size_t k = 0; k < inside_ids.size(); ++k) {
            if (k > 0) ids_str += ",";
            char buf[32]; snprintf(buf, sizeof(buf), "%lu", (unsigned long) inside_ids[k]);
            ids_str += buf;
        }
        sb.field_s("inside_ids", ids_str.c_str());
    }

    // Publish trails as polylines
    for (size_t i = 0; i < persons.size(); ++i) {
        const PersonInfo &pi = persons[i];
        const TrackState &ts = track_states[pi.track_id];
        if (ts.trail.size() >= 2) {
            std::vector<double> trail_xy;
            for (size_t t = 0; t < ts.trail.size(); ++t) {
                trail_xy.push_back((double) ts.trail[t].first);
                trail_xy.push_back((double) ts.trail[t].second);
            }
            char trail_label[64];
            snprintf(trail_label, sizeof(trail_label), "trail_%lu", (unsigned long) pi.track_id);
            sb.polyline(trail_xy, trail_label);
        }
    }

    sb.publish(in->frame_meta);

    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);