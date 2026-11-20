# StreetMenu-VN: cách chạy pipeline

Mọi đường dẫn và phần cứng lấy từ `.env` (mẫu: `.env.example`). Mọi lệnh chạy từ thư mục repo
`code/`, với Python của môi trường chính.

## Một lần

```bash
cp .env.example .env          # SMV_DATA_DIR, SMV_MODELS_DIR, HF_TOKEN (cho mô hình gated)
make setup                    # env chính (main) + venv vLLM + venv OCR, trong thư mục code/
make verify-models            # chỉ metadata HF: repo, sha, giấy phép, dung lượng -> configs/models/VERIFIED.csv
```

## Mỗi lần có dữ liệu mới

```bash
make ingest                   # quét ảnh + nhãn, kiểm tra, tiền xử lý vào cache/, chia split
make stats agreement          # Bảng 1, đồng thuận người gán
make freeze IDS="P0 P1"       # đóng băng prompt sau khi chỉnh trên dev (bắt buộc trước split test)
scripts/run_tmux.sh pilot     # chạy pilot trong tmux (log: outputs/logs/pilot.log)
make status                   # mô hình nào xong/đang chạy/lỗi, còn bao nhiêu request, ổ đĩa
make score report             # chấm điểm + bảng LaTeX/Markdown + hình (reports/pilot/)
scripts/run_tmux.sh full      # G1-G6 (thêm ARGS=--with-optional để chạy cả mô hình/prompt tùy chọn)
make papers                   # tải tài liệu tham khảo + papers/models_table.tex
```

`PLAN=configs/plans/full.yaml make status score report` để đổi plan.

## Chạy gì, ở đâu

| Thành phần | Chạy bằng | Ghi chú |
|---|---|---|
| orchestrator, chấm điểm, báo cáo | môi trường chính (Python 3.10) | `taskset -c $SMV_CPU_AFFINITY`, 10 luồng |
| `vllm serve` | `.venvs/smv_vllm` | `CUDA_VISIBLE_DEVICES=$SMV_GPU_ID`, process group riêng |
| backend `hf` (transformers) | môi trường chính hoặc venv vLLM (`hf.python`) | batch nhiều ảnh, tự giảm batch khi OOM |
| OCR cổ điển | `.venvs/smv_ocr` | PaddleOCR (GPU), EasyOCR + VietOCR (CPU) |

Trình tự một plan: OCR → VLM (theo thứ tự trong plan) → LLM văn bản. Với mỗi mô hình: kiểm GPU trống →
tải trọng số vào `$SMV_MODELS_DIR/<key>` (revision ghim) → khởi động server → gửi đồng thời mọi request của
mọi prompt/biến thể thinking/seed → tắt server → chờ GPU nhả → ghi manifest → xóa trọng số
(`keep_weights: true` trong YAML hoặc `ARGS=--keep-weights` để giữ). Mô hình kế tiếp được tải trước ở nền
nếu ổ đĩa đủ cho cả hai.

## Khi có sự cố

- OOM khi khởi động hoặc preemption nhiều: tự giảm `max_num_seqs` → `max_model_len` → token ảnh; cấu hình
  cuối nằm trong manifest (`tune_final`, `tune_history`).
- Server chết giữa chừng: tự khởi động lại (tối đa `vllm.server_restarts` lần), gửi lại request dở.
- Mô hình không nạp được (kiến trúc không hỗ trợ, gated, OOM sau mọi nấc): ghi vào
  `outputs/<plan>/status.csv`, hàng đợi chạy tiếp mô hình sau.
- Ctrl-C / `kill`: tắt server sạch sẽ; chạy lại cùng lệnh thì bỏ qua request đã xong
  (kết quả lưu từng dòng trong `outputs/results/<model>/<prompt>.jsonl`).
- GPU đang có người dùng: dừng và in PID/người dùng, không giết tiến trình của ai.

## Kiểm thử

```bash
make test        # pytest: schema, ingest, chấm điểm, ca biên, đầu-cuối với server giả
make smoke       # đầu-cuối với server giả trên 16 thực đơn tổng hợp (không GPU, không tải)
make smoke-real  # một mô hình thật nhỏ (MODEL=qwen3vl_2b) trên thực đơn tổng hợp, GPU 0
make trial-all   # thử thật MỌI mô hình trên thực đơn tổng hợp (tải, chạy, kiểm, xóa)
```
