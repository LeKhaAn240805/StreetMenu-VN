# VietMenu Labeler: hướng dẫn và schema nhãn v3

Thư mục này gồm:

| File | Dùng để |
|---|---|
| `index.html` | Web gán nhãn (một file, chạy trên Chrome/Edge, không cần cài gì) |
| `heic2any.min.js` | Bộ giải mã ảnh `.HEIC` của iPhone (để cạnh `index.html`) |
| `schema_config.json` | Danh mục, thẻ, mô tả và ví dụ của mọi trường. Sửa file này để đổi nhãn mà không sửa code |
| `examples/*.json` | Ba nhãn mẫu hoàn chỉnh (30.jpg, IMG_1356.JPG, 35.jpg) |

---

## 1. Kết luận: schema trong file Word đã đủ để bắt đầu chưa?

**Đủ về khung, nhưng nên dùng bản v3 trong web này trước khi gán hàng loạt.** Schema v2 (ảnh → mục → món → offer, hai lớp raw/norm, 30 hiện tượng) được rút ra từ 19 ảnh S01–S19. Thư mục `Data` hiện có 76 ảnh; khoảng 57 ảnh mới có những hiện tượng mà v2 chưa có quy tắc. Nếu không chốt trước, ba người gán sẽ xử lý khác nhau và độ đồng thuận sẽ thấp.

### 1.1 Các thiếu sót tìm thấy trong ảnh mới và cách v3 xử lý

| # | Hiện tượng chưa có quy tắc trong v2 | Ví dụ trong `Data` | Cách xử lý trong v3 |
|---|---|---|---|
| 1 | **Menu chuỗi/nhà hàng chiếm gần một nửa ảnh (khoảng 35/76)** (KFC, Lotteria, McCafé, Sukiya, DQ, KOI, Saizeriya, Hotpot Story…) | 6.jpg, 7.jpg, 21–23.jpg, 26–29.jpg | Trường mới `venue_tier` (street_vendor / small_shop / chain / supermarket_foodcourt). Báo riêng kết quả "đường phố", và kiểm tra rò rỉ dữ liệu vì menu chuỗi có sẵn trên mạng. Tên "StreetMenu-VN" chỉ đúng nếu tách được tập con này |
| 2 | **Phụ thu cộng thêm** "+25.000", "Tự mix +2K", "Thêm bò +40.000" | 21.jpg, 24.jpg, temp_image_1BC1… | `price_kind = delta`, món là `is_add_on` |
| 3 | **Dãy giá không ghi nhãn size** "20k 25k 30k", "40 \| 59 \| 64" | IMG_1356, 19.jpg | size `#1`, `#2`, `#3` + `size_rank`; đánh dấu `variant_label_unknown` |
| 4 | **Khoảng giá không gán được** "45k-55k/dĩa", "30-35-37-40k/1 con", "35k-40k" | C98E…, IMG_1361 | `price_kind = range` với trường mới `price_min_vnd`, `price_max_vnd` |
| 5 | **Bậc số lượng** "1 miếng 35.000 / 2 miếng 70.000 / 3 miếng 104.000"; "(4) (6) (10)" | 6.jpg, 25.jpg | Trường mới `quantity` + `qty_unit` trên offer |
| 6 | **Hệ size khác S/M/L**: P/R/L, M/S/L/2X/Mega, Vừa/Lớn/Đại | 11.jpg, 23.jpg | `size` giữ nhãn gốc, `size_rank` để so sánh |
| 7 | **Thông tin khuyến mãi không phải giá** "Chỉ 88.000/người", "Tiết kiệm 23.000", "Gọi riêng 310.000", "Giảm sốc 38%" | 26.jpg, 21.jpg, 12.jpg, IMG_1471 | Không gán. Đưa vào thẻ bẫy `D_DISTRACTOR`: mô hình xuất ra là ảo giác |
| 8 | **Phiếu gọi món có số lượng khách ghi tay** ("2", "1" ở cột SL, "Bàn số 7") | IMG_1475, IMG_1464 | Không gán; thẻ `D_DISTRACTOR` |
| 9 | **Ô bảng gộp** "89k" phủ 5 loại lẩu | IMG_1464 | `price_source = merged_cell` |
| 10 | **Mã món** 4102, A20, 2309 | 12–14.jpg, 16.jpg | Trường mới `item_code` (không đưa vào tên) |
| 11 | **Lựa chọn không đổi giá** "Chọn 1 sốt", "Có thể chọn nước uống", mức đường/đá; set chọn theo bước | IMG_1341, 21.jpg, 28.jpg, 8.jpg | Trường mới `option_groups` (ghi chữ, không tạo offer) |
| 12 | **Biểu tượng và thời gian bán**: món chay, best seller, cay, "theo mùa", "Thứ 2 – Thứ 7" | IMG_1355, IMG_1422, temp_image_78CF… | Trường mới `attributes`, `availability` |
| 13 | **Ghi chú VAT** "chưa bao gồm 10% VAT" / "đã bao gồm VAT" | 20.jpg, 26.jpg, IMG_1355 | Trường mới cấp ảnh `tax_note`; không cộng thuế vào `price_vnd` |
| 14 | **Menu không có giá** | IMG_1414, IMG_1483, IMG_1482 | Trường mới `has_prices = false`; chỉ chấm tên |
| 15 | **Mặt hàng không phải đồ ăn** (thuốc lá, khăn lạnh) | 31.jpg, IMG_1464 | `category = không phải đồ ăn`, vẫn gán |
| 16 | **Ảnh trùng/gần trùng** | 12.jpg ≈ 13.jpg, 16.jpg ≈ 17.jpg | Trường mới `near_duplicate_of`; cùng `group_id`, chỉ một ảnh vào test |
| 17 | **Ảnh HEIC/WEBP** | IMG_1468–1473.HEIC | Web tự giải mã HEIC. Trước khi công bố bộ dữ liệu nên đổi hết sang JPG và xóa EXIF |

### 1.2 Gộp thẻ hiện tượng: từ khoảng 80 xuống 22 thẻ

Bộ thẻ v2 + các hiện tượng mới lên tới gần 80 mã, quá nhiều để ba người gán nhất quán và để báo cáo có ý nghĩa thống kê (nhiều thẻ chỉ có 1–2 ảnh). v3 gộp còn **22 thẻ**, mỗi nhóm ứng với một loại ngữ cảnh và một chỉ số trong bài:

| Nhóm (chỉ số) | Thẻ v3 | Gộp từ mã cũ |
|---|---|---|
| Tiền tệ / giá (PNA, IPA) | `P_IMPLICIT_K` | P_IMPLICIT_K (giữ riêng vì là hiện tượng ngữ cảnh tiền tệ cốt lõi) |
| | `P_FORMAT` | P_K_SUFFIX, P_WORD, P_FULL_DOTS, P_SUPERSCRIPT_000, P_INLINE_SIZE_PREFIX |
| | `P_INHERITED` | P_SECTION, P_GLOBAL, P_COLUMN, P_BRACKET, P_MERGED_CELL, L_SCOPE_AMBIG |
| | `P_MULTI_UNLABELED` | P_UNLABELED_TIERS, P_RANGE_SIZE, P_RANGE_UNKNOWN |
| | `P_UNIT_BUNDLE` | P_UNIT, P_BUNDLE, P_FROM |
| | `P_PROMO` | P_PROMO |
| | `P_HANDWRITTEN_OVERRIDE` | P_HANDWRITTEN_OVERRIDE |
| | `P_MISSING` | P_MISSING, P_NO_PRICES |
| Ngôn ngữ (NNA) | `N_NORMALIZE` | N_ABBR, N_SLANG_REGIONAL, N_MISSPELL, N_NO_DIACRITICS, N_SPELLING_VARIANTS (loại cụ thể ghi ở LEXICON) |
| | `N_SECTION_COMPOSE` | N_SECTION_COMPOSE |
| | `N_FOREIGN` | N_BILINGUAL, N_FOREIGN_DISH, N_BRAND_CREATIVE |
| Logic (Item F1, Variant F1) | `L_VARIANT` | L_VARIANT_TEMP, L_VARIANT_SIZE, P_PORTION, P_QTY_TIERS |
| | `L_COMBO_SET` | L_COMBO, L_SET_BUILDER, L_OPTION_GROUP |
| | `L_ADDON` | L_TOPPING, P_DELTA |
| | `L_ITEM_SPLIT` | N_SLASH_ALTERNATIVES, L_DUPLICATE |
| Không gian (SAA) | `S_DETACHED` | S_DETACHED_BADGE, S_MISALIGNED, S_GRID_ALIGN |
| | `S_LAYOUT` | S_MULTI_PANEL, S_FREE_LAYOUT, S_TABLE_FORM, S_PHOTO_HEAVY, S_ICON_LEGEND |
| Thị giác | `V_HANDWRITING` | V_HANDWRITING, V_HANDWRITTEN_FULL |
| | `V_DAMAGE` | V_OCCLUSION, V_BURN_DAMAGE, V_TRUNCATED |
| | `V_QUALITY` | V_GLARE, V_LOWLIGHT, V_BLUR, V_DECORATIVE_FONT |
| | `V_GEOMETRY` | V_PERSPECTIVE, S_ROTATED |
| Bẫy ảo giác (Hallucination) | `D_DISTRACTOR` | N_DESC_LINE, N_NUMBERED, V_GHOST, P_PROMO_INFO, V_ORDER_MARKS |

Không cần gán thẻ cho những gì đã có trường riêng hoặc tự tính được: V_DENSE (tự tính số offer > 60 khi xuất), X_* (trường `pii_types`), D_OOD (`split = ood`), VAT (`tax_note`).

Gợi ý viết bài: báo kết quả chính theo 4 mức CCL và 6 nhóm thẻ; chỉ tách thẻ con khi một thẻ có đủ ảnh (ví dụ ≥ 15 ảnh). Bảng ánh xạ trên cho phép quay lại mã cũ nếu reviewer hỏi.

### 1.3 Các trường mới trong v3 (so với v2)

- Cấp ảnh: `venue_tier`, `near_duplicate_of`, `tax_note`, `has_prices`, `source`, `pii_types`, `file_sha256`, `file_size`, `status`, `seconds_spent`.
- Cấp món: `item_code`, `option_groups`, `attributes`, `availability`, `bbox` (tùy chọn); `combo_components` thành danh sách `{name, qty, ref_price_vnd}`; `item_status` thêm `sold_out`; `ambig_reason` thêm `variant_label_unknown`.
- Cấp offer: `size_rank`, `quantity`, `qty_unit`, `price_min_vnd`, `price_max_vnd`; `price_kind` thêm `delta`; `price_source` thêm `merged_cell`.
- `ccl` của mức 3 và 4 được bổ sung các hiện tượng mới (bậc số lượng, delta → CCL-3; số lượng ghi tay trên phiếu gọi món → CCL-4).

Các quy tắc còn lại của file Word (5 nguyên tắc vàng, tách món/tách offer, 30 quy tắc quyết định, ví dụ S03/S05/S14/S16/S17/S18/S11/S09) giữ nguyên.

---

## 2. Cách dùng web

### Chạy
1. Mở `index.html` bằng **Chrome hoặc Edge** (bấm đúp file). Firefox/Safari vẫn chạy nhưng không ghi thẳng vào thư mục; nhãn được giữ trong trình duyệt, nhớ bấm **⬇ Xuất**.
2. Nhập **tên người gán** (mỗi người một tên khác nhau).
3. Bấm **📂 Mở thư mục ảnh**, chọn thư mục ảnh (ví dụ `D:\menus\images`), bấm **Cho phép** khi trình duyệt hỏi quyền chỉnh sửa. Lần sau chỉ cần bấm **↩ Mở lại**.

### Nhãn được lưu ở đâu
```
Data/
  1.jpg, 2.jpg, …
  _labels/
    schema_config.json        (tùy chọn: cấu hình chung của nhóm)
    An/1.jpg.json             (nhãn của An)
    Binh/1.jpg.json           (nhãn của Bình)
```
Mỗi người gán một thư mục con, nên nhiều người dùng chung một thư mục (Google Drive/OneDrive đồng bộ) không ghi đè nhau, và có sẵn cặp nhãn để tính độ đồng thuận.

### Khi thêm hoặc đổi ảnh
- Chép ảnh mới vào `Data` rồi bấm **🔄 Quét lại**: ảnh mới hiện nhãn "mới".
- Ảnh bị thay bằng file khác cùng tên: hiện nhãn "đổi?".
- Ảnh bị đổi tên: nhãn cũ hiện ở cuối danh sách là "nhãn mồ côi"; bấm **Tìm ảnh khớp theo nội dung** để ghép lại (so mã băm SHA-256 của ảnh).

### Làm một ảnh
1. Tab **Ảnh**: chọn Quy ước giá, CCL, thẻ, bối cảnh; vẽ vùng bỏ qua nếu menu bị cắt.
2. Tab **Mục & Món**: tạo mục, dán các dòng món vào **Nhập nhanh** (ví dụ `Trà lipton nóng/ đá 25 / 40`, `Mọng Khế Nho | 38/45 | M/L`, `Xôi gà xé 20k 25k 30k`, `Bò lá lốt 50k/phần`, `Set rau muống +32.000`). Web tự tách offer, quy đổi giá, nhận ra giá theo đơn vị và phụ thu. Sau đó sửa tên chuẩn hóa, nhóm món.
3. Tab **Kiểm tra**: sửa hết ⛔, xem ⚠ (giá bất thường, thiếu thẻ tương ứng với nhãn, offer trùng…).
4. Bấm **✓ Xong**, hoặc **⚑ Cần xem lại** nếu muốn người phân xử xem.

Mọi trường và thẻ đều có nút **?** hiện định nghĩa, ví dụ lấy từ chính ảnh trong `Data` và (với thẻ) các mã cũ đã gộp. Ô **Mô tả** ở thanh trên bật/tắt phần giải thích dưới mỗi trường. Tab **Hướng dẫn** có đủ nguyên tắc, bảng CCL, từ điển thẻ và từ điển trường.

Phím tắt: Ctrl+S lưu · Alt+→/← ảnh sau/trước · Alt+N thêm món · lăn chuột phóng to · F vừa khung · Esc hủy vẽ.

### Phân xử
Người phân xử mở cùng ảnh, bấm **⧉ Lấy nhãn người khác** để lấy nhãn của A hoặc B làm bản nháp, sửa rồi lưu dưới tên mình (ví dụ `C` hoặc `gold`).

### Xuất
**⬇ Xuất** tải về hai file: `vietmenu_labels_*.jsonl` (mỗi dòng một ảnh, đủ schema) và `vietmenu_offers_*.csv` (mỗi dòng một offer, mở được bằng Excel). Có thể chọn chỉ nhãn của mình hoặc của tất cả người gán.

### Gửi cho người khác
Gửi cả thư mục `labeler` (nén zip) cho từng người. Nếu muốn một đường link: kéo thư mục `labeler` vào Netlify Drop (app.netlify.com/drop) hoặc đưa lên GitHub Pages. Lưu ý: vì lý do bảo mật, không trang web nào tự đọc được ổ đĩa của người khác qua link. Mỗi người vẫn bấm **Mở thư mục** một lần và chọn thư mục ảnh trên máy mình; trình duyệt sẽ nhớ cho lần sau. Ảnh không bị tải lên đâu cả.

### Đổi danh mục / thẻ
Trong tab **Hướng dẫn**, bấm **Ghi file cấu hình mẫu** để tạo `_labels/schema_config.json`. Sửa danh mục, mô tả, ví dụ trong file đó rồi tải lại trang. Cả nhóm dùng chung file này nếu dùng chung thư mục.
