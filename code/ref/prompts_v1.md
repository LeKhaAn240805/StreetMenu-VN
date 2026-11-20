# Prompt v1 (trích nguyên văn Mục 6.2–6.4 file Word, ngày 01/11/2026)

Nguồn: Huong_dan_TAIMMA2027_StreetMenuVN_ban2.docx. Đây là văn bản gốc để tách thành prompts/*.txt. Không sửa nội dung prompt khi tách; chỉ sửa trên tập dev và ghi CHANGELOG.

```text
6.2 Schema đầu ra cho mô hình
Schema cho mô hình gọn hơn schema nhãn: bỏ các trường chỉ người gán mới biết (price_source, evidence, legibility). Khóa giữ tiếng Anh ở mọi phiên bản prompt để bộ chấm không phải xử lý hai ngôn ngữ; phần mô tả giá trị có thể dịch. Bản JSON Schema cho giải mã có ràng buộc ở Phụ lục B.
{
"items": [
{
"section": "heading the item is listed under, or null",
"raw_name": "item name exactly as written",
"name": "item name",
"description": "description text, or null",
"is_topping": false,
"is_combo": false,
"offers": [
{
"size": "size, or null",
"temp": "nóng | đá | null",
"portion": "portion, or null",
"raw_price": "price exactly as written, or null",
"price_vnd": 25000,
"old_price_vnd": null,
"unit": "unit such as 100 g, or null"
}
]
}
]
}
6.3 Họ prompt và câu hỏi mỗi prompt trả lời
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Mã Nội dung Trả lời câu hỏi Pilot / mở rộng
 --------------- -------------------------------------------------------------------- ---------------------------------------------------------------------------------- -----------------
 P0 Chỉ yêu cầu trích xuất theo schema Tri thức nội tại thuần (RQ2) Pilot
 P1 P0 + bối cảnh và 10 quy tắc chung, không có danh sách từ cụ thể Prompt ngữ cảnh chính của bài (RQ1) Pilot
 P2 P1 + ghi chú từ vựng và quy ước rút từ tập dev Cận trên khi được mớm tri thức; nút thắt là kiến thức hay thị giác (RQ2) Pilot nếu kịp
 P1-VI P1 viết bằng tiếng Việt Ngôn ngữ prompt có đổi kết quả không Mở rộng
 P1a, P1b, P1c Ba cách diễn đạt tương đương của P1 Độ nhạy diễn đạt (RQ5) Mở rộng
 P-T Chỉ chép lại chữ theo dòng Năng lực tri giác (RQ3) Pilot nếu kịp
 P-2S Bước 1 chép chữ, bước 2 cùng mô hình trích xuất có ảnh + bản chép Tách đọc và hiểu có giúp không (RQ3) Mở rộng
 P-OCR+ Như P-2S nhưng bản chép do OCR ngoài (PaddleOCR-VL hoặc PaddleOCR) Bổ sung OCR có giúp VLM không (RQ3) Mở rộng
 P-Oracle Không có ảnh; đưa dòng chữ do người chép (tên và giá cùng dòng) Năng lực chuẩn hóa ngôn ngữ và tiền tệ khi đã loại tri giác và bố cục (RQ3) Mở rộng
 P-NL Bước 1 liệt kê tự do, bước 2 chuyển sang JSON Ràng buộc định dạng có làm giảm chất lượng không Mở rộng
 P-Conf P1 + trường confidence cho mỗi offer Hiệu chuẩn độ tin cậy, đường rủi ro–độ phủ (RQ5) Mở rộng
 P-Adv P1 nhưng thay câu bối cảnh bằng một quy ước giá sai Mô hình bám ngữ cảnh sai đến đâu (tham khảo WorldCuisines); diễn giải thận trọng Tùy chọn
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
6.4 Toàn văn prompt
Mọi prompt dưới đây dùng chung một system prompt. Ký hiệu [image] là vị trí chèn ảnh theo chat template của mô hình; {SCHEMA} là schema ở Mục 6.2. Giữ nguyên văn khi chạy, chỉ sửa trên tập dev và ghi phiên bản.
System prompt
EN: You extract structured data from photos of menus. Reply with one JSON object only.
VI: Bạn trích xuất dữ liệu có cấu trúc từ ảnh thực đơn. Chỉ trả lời bằng một đối tượng JSON.
P0 (tối thiểu)
[image]
Extract every menu item and its price(s) from this image.
Return one JSON object that follows this schema exactly:
{SCHEMA}
[ảnh]
Trích xuất mọi món trong ảnh thực đơn này cùng giá của chúng.
Trả về một đối tượng JSON theo đúng schema sau:
{SCHEMA}
P1 (ngữ cảnh), bản tiếng Anh
[image]
Context: the photo shows a menu, price board or sign from a Vietnamese street-food
stall, café, drink shop or small restaurant. It may be printed or handwritten,
partly corrected by hand, photographed at an angle, or partly covered.
Instructions:
1. Read the whole image first. Extract every item offered for sale, including
toppings and add-ons. Do not output shop names, slogans, addresses, phone
numbers, opening hours, Wi-Fi details, bank or QR information, or description
lines as items. Put a description line under its item in "description".
2. raw_name and raw_price: copy the text exactly as written, including spelling
mistakes and missing accents.
3. name: the full standard Vietnamese name with correct diacritics. Expand
abbreviations and fix obvious misspellings only when you are confident. If the
heading the item is listed under is part of its identity (a flavour listed
under a drink-type heading), include the heading in the name. If unsure,
copy raw_name.
4. price_vnd: an integer number of Vietnamese dong. "15k", "15K", "15 ngàn",
"15.000" and "15.000đ" all mean 15000. A bare number such as "15" usually
means thousands of dong when the other prices on the menu are written the
same way.
5. Shared prices: a price written once may apply to several items, for example a
heading that says all items cost the same, a section heading with a price,
column headers for sizes, or a bracket or note covering several lines. Apply
it to every item it covers.
6. Options: if an item has several sizes, hot/iced options, portions or several
prices, create one entry in "offers" for each option. "nóng/đá 25 / 40" means
hot 25000 and iced 40000.
7. Corrections and promotions: if a price is crossed out and a new one is given,
price_vnd is the new price and old_price_vnd the crossed-out one. A
handwritten number written over a printed price replaces it.
8. Unit prices and bundles: if the price is per weight or per quantity, put that
in "unit".
9. Layout: a price may be far from its item, inside a badge on a photo, in a
separate column, or linked by a line. Pair each price with the item it
visually belongs to.
10. Report only what you can see. If a price is unreadable or you cannot tell
which item it belongs to, set price_vnd to null instead of guessing. Do not
invent items.
Return one JSON object that follows this schema exactly:
{SCHEMA}
P1 (ngữ cảnh), bản tiếng Việt
[ảnh]
Bối cảnh: ảnh chụp thực đơn, bảng giá hoặc biển hiệu của một quán ăn đường phố,
quán cà phê, quán nước hoặc quán ăn nhỏ ở Việt Nam. Thực đơn có thể in sẵn hoặc
viết tay, bị sửa tay một phần, chụp nghiêng, hoặc bị che một phần.
Hướng dẫn:
1. Đọc toàn bộ ảnh trước. Trích xuất mọi món được bán, kể cả topping và món
thêm. Không xuất tên quán, khẩu hiệu, địa chỉ, số điện thoại, giờ mở cửa,
thông tin Wi-Fi, ngân hàng, mã QR, hay dòng mô tả thành món. Dòng mô tả đưa
vào trường "description" của món tương ứng.
2. raw_name và raw_price: chép đúng như trên ảnh, giữ cả lỗi chính tả và chữ
thiếu dấu.
3. name: tên tiếng Việt chuẩn, đầy đủ, có dấu. Chỉ mở rộng viết tắt và sửa lỗi
chính tả rõ ràng khi chắc chắn. Nếu tiêu đề mục là một phần tên món (một vị
nằm dưới tiêu đề loại đồ uống), ghép tiêu đề vào tên. Không chắc thì chép
lại raw_name.
4. price_vnd: số nguyên, đơn vị đồng. "15k", "15K", "15 ngàn", "15.000",
"15.000đ" đều là 15000. Số trần như "15" thường là nghìn đồng nếu các giá
khác trên thực đơn cũng ghi như vậy.
5. Giá chung: một giá ghi một lần có thể áp cho nhiều món, ví dụ tiêu đề ghi mọi
món cùng giá, tiêu đề mục có giá, tiêu đề cột theo size, hoặc dấu ngoặc,
ghi chú phủ nhiều dòng. Áp giá đó cho mọi món nó bao phủ.
6. Lựa chọn: món có nhiều size, nóng/đá, khẩu phần hoặc nhiều giá thì tạo một
phần tử trong "offers" cho mỗi lựa chọn. "nóng/đá 25 / 40" là nóng 25000 và
đá 40000.
7. Sửa giá và khuyến mãi: giá bị gạch và có giá mới thì price_vnd là giá mới,
old_price_vnd là giá bị gạch. Số viết tay đè lên giá in thì thay giá in.
8. Giá theo đơn vị và theo gói: giá tính theo khối lượng hoặc số lượng thì ghi
vào "unit".
9. Bố cục: giá có thể nằm xa tên món, trong huy hiệu trên ảnh, ở cột riêng, hoặc
nối bằng đường kẻ. Ghép mỗi giá với món mà nó thuộc về theo hình ảnh.
10. Chỉ ghi những gì nhìn thấy. Giá không đọc được hoặc không biết thuộc món
nào thì để price_vnd là null, không đoán. Không bịa món.
Trả về một đối tượng JSON theo đúng schema sau:
{SCHEMA}
P2: chèn khối sau vào giữa hướng dẫn 10 và dòng "Return one JSON object…"
Reference notes for Vietnamese menus (use only when relevant; never add items
that are not in the image):
- Names and spellings: "cf", "cafe" = cà phê; "thơm" = dứa; "sapoche",
"sampoche" = sa-pô-chê; "yaourt" = sữa chua; "trà tắc" = trà quất; "đậu hủ",
"đậu hũ", "tàu hủ" = tàu hũ; "olong" = ô long; "cút" = trứng cút;
"bánh plan" = bánh flan; "ba rọi" = ba chỉ; "1 chục" = 10 pieces.
- Under a heading such as "Sinh tố", "Nước ép" or "Sữa chua", a bare flavour
name means that drink: "Dâu" under "Sinh tố" is "sinh tố dâu".
- Size conventions: "M | L", "nhỏ – lớn", "ly nhỏ – ly lớn". On a drinks board a
pair such as "8k-15k" usually means small and large.
- "đồng giá 15K" means every item in its scope costs 15000.
Lưu ý về P2
Khối ghi chú chỉ dùng mục từ của tập dev (19 ảnh mẫu). Khi viết bài, nói rõ P2 không còn là zero-shot thuần vì đã tiêm tri thức. Nếu P2 không hơn P1, đó vẫn là kết quả có giá trị (CROPE cũng thấy thêm ngữ cảnh không luôn giúp).
P-T (chép chữ, đo tri giác)
[image]
Transcribe all text visible in this image, line by line, in reading order (top
to bottom; finish one column or panel before moving to the next). Copy exactly as
written, including prices, spelling mistakes, missing accents and symbols. Do not
translate, correct, merge or explain anything. Write "?" for characters you
cannot read.
Return one JSON object: {"lines": ["first line", "second line", "..."]}
P-2S và P-OCR+ (bước 2)
[image]
Below is a line-by-line transcription of the menu in the image, made in an
earlier step. It may contain mistakes; when the transcription and the image
disagree, trust the image.
<transcription>
{LINES}
</transcription>
{P1 context, instructions 1-10 and schema, unchanged}
Với P-OCR+, thay câu đầu bằng "Below is the output of an OCR system for this image. It may contain mistakes…" và sắp dòng OCR theo thứ tự trên–trái xuống dưới–phải (ViTextVQA cho thấy thứ tự token OCR ảnh hưởng mạnh).
P-Oracle (không có ảnh)
The lines below were transcribed by a person from a photo of a Vietnamese
menu, in reading order. Each line that contains both an item and its price is
written on one line.
<lines>
{GOLD_LINES}
</lines>
{P1 instructions 1-8 and 10, then the schema}
Bản dòng chữ oracle sinh tự động từ nhãn: mỗi món một dòng "raw_name raw_price", tiêu đề mục giữ nguyên dòng. Vì đã ghép sẵn tên với giá, P-Oracle đo riêng ngữ cảnh ngôn ngữ, tiền tệ và logic; không đo không gian.
P-NL (hai bước tự do rồi JSON)
[image]
List every item on this menu with its price or prices, one item per line, in
the form: item name — options and prices. Copy names and prices as written.
Convert the list below into JSON following the rules and schema.
<list>
{STAGE1_OUTPUT}
</list>
{P1 instructions 2-8 and 10, then the schema}
P-Conf (thêm vào cuối hướng dẫn của P1)
In every offer, add "confidence": a number between 0 and 1 saying how sure you
are that price_vnd is the correct current price for this item and option.
Thay cho confidence tự khai, có thể lấy log-xác suất trung bình của các token giá từ vLLM (logprobs). Báo cả hai nếu có thời gian; confidence tự khai của VLM thường hiệu chuẩn kém.
Ba cách diễn đạt P1a, P1b, P1c
- P1a: đảo thứ tự quy tắc (bố cục lên đầu, tiền tệ xuống cuối), giữ nguyên nội dung.
- P1b: viết lại thành gạch đầu dòng ngắn, bỏ phần bối cảnh mô tả.
- P1c: viết dạng đoạn văn liền mạch.
- Sinh một lần, cho một người khác kiểm tra tương đương nghĩa, rồi đóng băng. Báo trung bình, nhỏ nhất, lớn nhất của offer F1.
```
