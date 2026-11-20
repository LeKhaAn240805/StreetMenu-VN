#!/usr/bin/env python3
"""Build prompts/*.txt from the verbatim text of ref/prompts_v1.md (sections 6.2-6.4).

Run once; the generated .txt files are the source of truth afterwards (hashes go to FROZEN.json).
Changes against prompts_v1.md are listed in prompts/CHANGELOG.md.
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "prompts"

SYSTEM_EN = "You extract structured data from photos of menus. Reply with one JSON object only."
SYSTEM_VI = "Bạn trích xuất dữ liệu có cấu trúc từ ảnh thực đơn. Chỉ trả lời bằng một đối tượng JSON."

SCHEMA = """{
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
          "price_max_vnd": null,
          "old_price_vnd": null,
          "unit": "unit such as 100 g, or null"
        }
      ]
    }
  ]
}"""

CONTEXT_EN = """Context: the photo shows a menu, price board or sign from a Vietnamese street-food
stall, café, drink shop or small restaurant. It may be printed or handwritten,
partly corrected by hand, photographed at an angle, or partly covered."""

RULES_EN = {
    1: """1. Read the whole image first. Extract every item offered for sale, including
toppings and add-ons. Do not output shop names, slogans, addresses, phone
numbers, opening hours, Wi-Fi details, bank or QR information, or description
lines as items. Put a description line under its item in "description". A heading
that groups several items (for example "SODA: 22K" above a list of flavours) is
not an item: list each flavour or dish under it as its own item.""",
    2: """2. raw_name and raw_price: copy the text exactly as written, including spelling
mistakes and missing accents. If a translation in another language is printed next
to the name, leave it out of raw_name and name.""",
    3: """3. name: the full standard Vietnamese name with correct diacritics. Expand
abbreviations and fix obvious misspellings only when you are confident. If the
heading the item is listed under is part of its identity (a flavour listed
under a drink-type heading), include the heading in the name. If unsure,
copy raw_name.""",
    4: """4. price_vnd: an integer number of Vietnamese dong. "15k", "15K", "15 ngàn",
"15.000" and "15.000đ" all mean 15000. A bare number such as "15" usually
means thousands of dong when the other prices on the menu are written the
same way. "12k5" means 12500, "1tr9" means 1900000 and "Free" means 0.""",
    5: """5. Shared prices: a price written once may apply to several items, for example a
heading that says all items cost the same, a section heading with a price,
column headers for sizes, or a bracket or note covering several lines. Apply
it to every item it covers.""",
    6: """6. Options: if an item has several sizes, hot/iced options, portions or several
prices, create one entry in "offers" for each option. "nóng/đá 25 / 40" means
hot 25000 and iced 40000. Several prices written in a row with no label ("20k 25k
30k") are one offer each, in reading order, with size null. A surcharge such as
"+5k" for a bigger size, on an item that has a clear base price, makes the price
of that option the full price (base + 5000), with raw_price "+5k". Only a separate
add-on or topping line ("Trân châu +5k") keeps the increment as its price_vnd,
with is_topping true.""",
    7: """7. Corrections and promotions: if a price is crossed out and a new one is given,
price_vnd is the new price and old_price_vnd the crossed-out one. A
handwritten number written over a printed price replaces it, even when the new
price is higher; the replaced price is old_price_vnd.""",
    # v1.1: range sentence added (DE_XUAT_CHINH_SUA_WORD.md B1) because the schema now has price_max_vnd
    8: """8. Unit prices and bundles: if the price is per weight or per quantity, put that
in "unit". If a price is a range such as 45k-55k, put the lower bound in
price_vnd and the upper bound in price_max_vnd.""",
    9: """9. Layout: a price may be far from its item, inside a badge on a photo, in a
separate column, or linked by a line. Pair each price with the item it
visually belongs to.""",
    10: """10. Report only what you can see. If a price is unreadable or you cannot tell
which item it belongs to, set price_vnd to null instead of guessing. If the menu
shows a word instead of a number ("tùy loại", "thời giá", "liên hệ"), or no price
at all for an item, set price_vnd to null and copy the words into raw_price. Do
not invent items.""",
}

RETURN_EN = "Return one JSON object that follows this schema exactly:\n{SCHEMA}"

CONTEXT_VI = """Bối cảnh: ảnh chụp thực đơn, bảng giá hoặc biển hiệu của một quán ăn đường phố,
quán cà phê, quán nước hoặc quán ăn nhỏ ở Việt Nam. Thực đơn có thể in sẵn hoặc
viết tay, bị sửa tay một phần, chụp nghiêng, hoặc bị che một phần."""

RULES_VI = """Hướng dẫn:
1. Đọc toàn bộ ảnh trước. Trích xuất mọi món được bán, kể cả topping và món
thêm. Không xuất tên quán, khẩu hiệu, địa chỉ, số điện thoại, giờ mở cửa,
thông tin Wi-Fi, ngân hàng, mã QR, hay dòng mô tả thành món. Dòng mô tả đưa
vào trường "description" của món tương ứng. Tiêu đề gom nhiều món (ví dụ "SODA:
22K" đứng trên danh sách các vị) không phải là một món: liệt kê từng vị hay từng
món bên dưới thành món riêng.
2. raw_name và raw_price: chép đúng như trên ảnh, giữ cả lỗi chính tả và chữ
thiếu dấu. Nếu cạnh tên có bản dịch sang ngôn ngữ khác thì không đưa vào raw_name
và name.
3. name: tên tiếng Việt chuẩn, đầy đủ, có dấu. Chỉ mở rộng viết tắt và sửa lỗi
chính tả rõ ràng khi chắc chắn. Nếu tiêu đề mục là một phần tên món (một vị
nằm dưới tiêu đề loại đồ uống), ghép tiêu đề vào tên. Không chắc thì chép
lại raw_name.
4. price_vnd: số nguyên, đơn vị đồng. "15k", "15K", "15 ngàn", "15.000",
"15.000đ" đều là 15000. Số trần như "15" thường là nghìn đồng nếu các giá
khác trên thực đơn cũng ghi như vậy. "12k5" là 12500, "1tr9" là 1900000, "Free" là 0.
5. Giá chung: một giá ghi một lần có thể áp cho nhiều món, ví dụ tiêu đề ghi mọi
món cùng giá, tiêu đề mục có giá, tiêu đề cột theo size, hoặc dấu ngoặc,
ghi chú phủ nhiều dòng. Áp giá đó cho mọi món nó bao phủ.
6. Lựa chọn: món có nhiều size, nóng/đá, khẩu phần hoặc nhiều giá thì tạo một
phần tử trong "offers" cho mỗi lựa chọn. "nóng/đá 25 / 40" là nóng 25000 và
đá 40000. Nhiều giá viết liền nhau không có nhãn ("20k 25k 30k") là mỗi giá một
lựa chọn, theo thứ tự đọc, size để null. Phụ thu như "+5k" cho size lớn, trên món
đã có giá gốc rõ, thì giá của lựa chọn đó là giá đầy đủ (giá gốc + 5000), raw_price
"+5k". Chỉ dòng món thêm hoặc topping riêng ("Trân châu +5k") mới giữ số cộng thêm
làm price_vnd, với is_topping true.
7. Sửa giá và khuyến mãi: giá bị gạch và có giá mới thì price_vnd là giá mới,
old_price_vnd là giá bị gạch. Số viết tay đè lên giá in thì thay giá in, kể cả khi giá mới cao
hơn; giá bị thay là old_price_vnd.
8. Giá theo đơn vị và theo gói: giá tính theo khối lượng hoặc số lượng thì ghi
vào "unit". Giá là một khoảng như 45k-55k thì ghi cận dưới vào price_vnd và
cận trên vào price_max_vnd.
9. Bố cục: giá có thể nằm xa tên món, trong huy hiệu trên ảnh, ở cột riêng, hoặc
nối bằng đường kẻ. Ghép mỗi giá với món mà nó thuộc về theo hình ảnh.
10. Chỉ ghi những gì nhìn thấy. Giá không đọc được hoặc không biết thuộc món
nào thì để price_vnd là null, không đoán. Nếu thực đơn ghi chữ thay cho số ("tùy
loại", "thời giá", "liên hệ"), hoặc món không có giá, thì để price_vnd là null và
chép chữ vào raw_price. Không bịa món.
Trả về một đối tượng JSON theo đúng schema sau:
{SCHEMA}"""

P2_NOTES = """Reference notes for Vietnamese menus (use only when relevant; never add items
that are not in the image):
- Names and spellings: "cf", "cafe" = cà phê; "thơm" = dứa; "sapoche",
"sampoche" = sa-pô-chê; "yaourt" = sữa chua; "trà tắc" = trà quất; "đậu hủ",
"đậu hũ", "tàu hủ" = tàu hũ; "olong" = ô long; "cút" = trứng cút;
"bánh plan" = bánh flan; "ba rọi" = ba chỉ; "1 chục" = 10 pieces.
- Under a heading such as "Sinh tố", "Nước ép" or "Sữa chua", a bare flavour
name means that drink: "Dâu" under "Sinh tố" is "sinh tố dâu".
- Size conventions: "M | L", "nhỏ – lớn", "ly nhỏ – ly lớn". On a drinks board a
pair such as "8k-15k" usually means small and large.
- "đồng giá 15K" means every item in its scope costs 15000."""

# v1.4 (16/11/2026): same rules as P1, other output layout. In the main run the indented JSON with every key spelled out
# costs about 140 output tokens per offer, so every menu above ~60 offers hits the 8,192-token output limit.
COMPACT_SCHEMA = ('{"items":[{"section":"heading the item is listed under","raw_name":"item name exactly as written",'
                  '"name":"item name","description":"description text","is_topping":"true for a topping or add-on",'
                  '"is_combo":"true for a combo or set","offers":[{"size":"size","temp":"nóng | đá","portion":"portion",'
                  '"raw_price":"price exactly as written","price_vnd":25000,'
                  '"price_max_vnd":"upper bound of a price range","old_price_vnd":"crossed-out or replaced price",'
                  '"unit":"unit such as 100 g"}]}]}')
RETURN_COMPACT = """Return one JSON object on a single line: no line breaks, no indentation, no spaces after ":" or ",".
Leave out every key whose value would be null or false, with one exception: every item keeps "raw_name",
"name" and at least one offer, and every offer keeps "price_vnd" (null when the menu shows no readable price
for it). The keys are those of this schema:
""" + COMPACT_SCHEMA

CONF = """In every offer, add "confidence": a number between 0 and 1 saying how sure you
are that price_vnd is the correct current price for this item and option."""


def rules(nums) -> str:
    return "\n".join(RULES_EN[n] for n in nums)


def p1_body(extra_after_10: str = "") -> str:
    parts = [CONTEXT_EN, "Instructions:", rules(range(1, 11))]
    if extra_after_10:
        parts.append(extra_after_10)
    parts.append(RETURN_EN)
    return "\n".join(parts)


PROMPTS: dict[str, str] = {
    "P0": "[image]\nExtract every menu item and its price(s) from this image.\n" + RETURN_EN,
    "P0-VI": "[image]\nTrích xuất mọi món trong ảnh thực đơn này cùng giá của chúng.\n"
             "Trả về một đối tượng JSON theo đúng schema sau:\n{SCHEMA}",
    "P1": "[image]\n" + p1_body(),
    "P1-VI": "[image]\n" + CONTEXT_VI + "\n" + RULES_VI,
    "P1-C": "[image]\n" + "\n".join([CONTEXT_EN, "Instructions:", rules(range(1, 11)), RETURN_COMPACT]),
    "P2": "[image]\n" + p1_body(P2_NOTES),
    "P-Conf": "[image]\n" + p1_body(CONF),
    "P-T": """[image]
Transcribe all text visible in this image, line by line, in reading order (top
to bottom; finish one column or panel before moving to the next). Copy exactly as
written, including prices, spelling mistakes, missing accents and symbols. Do not
translate, correct, merge or explain anything. Write "?" for characters you
cannot read.
Return one JSON object: {"lines": ["first line", "second line", "..."]}""",
    "P-2S": """[image]
Below is a line-by-line transcription of the menu in the image, made in an
earlier step. It may contain mistakes; when the transcription and the image
disagree, trust the image.
<transcription>
{LINES}
</transcription>
""" + p1_body(),
    "P-OCR+": """[image]
Below is the output of an OCR system for this image. It may contain mistakes;
when the transcription and the image disagree, trust the image.
<transcription>
{LINES}
</transcription>
""" + p1_body(),
    "P-Oracle": """The lines below were transcribed by a person from a photo of a Vietnamese
menu, in reading order. Each line that contains both an item and its price is
written on one line.
<lines>
{GOLD_LINES}
</lines>
Instructions:
""" + rules([1, 2, 3, 4, 5, 6, 7, 8, 10]) + "\n" + RETURN_EN,
    "P-NL.s1": """[image]
List every item on this menu with its price or prices, one item per line, in
the form: item name — options and prices. Copy names and prices as written.""",
    "P-NL": """Convert the list below into JSON following the rules and schema.
<list>
{STAGE1_OUTPUT}
</list>
Instructions:
""" + rules([2, 3, 4, 5, 6, 7, 8, 10]) + "\n" + RETURN_EN,
    # ---- drafts: must be checked for equivalence by a second person before freezing ----
    "P1a": "[image]\n" + CONTEXT_EN + "\nInstructions:\n" + "\n".join(
        f"{i}." + RULES_EN[n].split(".", 1)[1] for i, n in enumerate([9, 1, 2, 3, 5, 6, 7, 8, 10, 4], start=1)
    ) + "\n" + RETURN_EN,
    "P1b": """[image]
Rules:
- Extract every item for sale, including toppings and add-ons. No shop names, slogans, addresses, phone numbers, opening hours, Wi-Fi, bank or QR details. Description lines go in "description" of their item. A heading that groups items ("SODA: 22K" above a list of flavours) is not an item: list each flavour or dish as its own item.
- raw_name, raw_price: copy exactly as written, mistakes and missing accents included; a printed translation in another language stays out of raw_name and name.
- name: full standard Vietnamese name with diacritics; expand abbreviations and fix misspellings only when confident; add the section heading when it is part of the item's identity; if unsure, copy raw_name.
- price_vnd: integer dong. "15k", "15K", "15 ngàn", "15.000", "15.000đ" = 15000. A bare "15" usually means 15000 when other prices are written the same way. "12k5" = 12500, "1tr9" = 1900000, "Free" = 0.
- Shared prices (same-price heading, priced section heading, size column headers, bracket or note over several lines) apply to every item they cover.
- One "offers" entry per option (size, hot/iced, portion, several prices). "nóng/đá 25 / 40" = hot 25000, iced 40000. Unlabeled prices in a row ("20k 25k 30k") = one offer each in reading order, size null. "+5k" for a bigger size on an item with a clear base price: price_vnd = base + 5000, raw_price "+5k"; only a separate add-on or topping line keeps the increment as price_vnd, with is_topping true.
- Crossed-out price with a new one: price_vnd = new, old_price_vnd = crossed-out. A handwritten number over a printed price replaces it, even if higher (old price -> old_price_vnd).
- Per-weight or per-quantity prices: put the unit in "unit". Range such as 45k-55k: lower bound in price_vnd, upper bound in price_max_vnd.
- Pair each price with the item it visually belongs to, even when it is far away, in a badge, in a separate column or linked by a line.
- Report only what you can see; unreadable or unassignable price -> price_vnd null; a word instead of a number ("tùy loại", "thời giá", "liên hệ") or no price -> price_vnd null with the words in raw_price. Do not invent items.
""" + RETURN_EN,
    "P1c": """[image]
This photo shows a menu, price board or sign from a Vietnamese street-food stall, café, drink shop or small restaurant, which may be printed or handwritten, partly corrected by hand, photographed at an angle or partly covered. Read the whole image first and extract every item offered for sale, toppings and add-ons included, but never shop names, slogans, addresses, phone numbers, opening hours, Wi-Fi details, bank or QR information, and never description lines, which belong in the "description" field of their item, and a heading that groups several items ("SODA: 22K" above a list of flavours) is not itself an item, each flavour or dish under it being listed on its own. Copy raw_name and raw_price exactly as written, keeping spelling mistakes and missing accents but leaving out any printed translation into another language, while name is the full standard Vietnamese name with correct diacritics, expanding abbreviations and fixing obvious misspellings only when you are confident, adding the section heading when it is part of the item's identity (a flavour under a drink-type heading) and copying raw_name when unsure. price_vnd is an integer number of dong: "15k", "15K", "15 ngàn", "15.000" and "15.000đ" all mean 15000, and a bare "15" usually means thousands when the other prices are written the same way ("12k5" is 12500, "1tr9" is 1900000, "Free" is 0). A price written once, such as a same-price heading, a priced section heading, size column headers or a bracket or note over several lines, applies to every item it covers. When an item has several sizes, hot/iced options, portions or prices, give one "offers" entry per option, so "nóng/đá 25 / 40" means hot 25000 and iced 40000, unlabeled prices in a row ("20k 25k 30k") are one offer each in reading order with size null, and a surcharge such as "+5k" for a bigger size on an item with a clear base price makes that option's price_vnd the full price (base + 5000, raw_price "+5k") while only a separate add-on or topping line keeps the increment as price_vnd with is_topping true. If a price is crossed out and replaced, price_vnd is the new price and old_price_vnd the crossed-out one, and a handwritten number over a printed price replaces it even when it is higher, the replaced price being old_price_vnd. Per-weight or per-quantity prices keep their unit in "unit", and a range such as 45k-55k puts the lower bound in price_vnd and the upper bound in price_max_vnd. A price may sit far from its item, inside a badge, in a separate column or at the end of a connecting line, so pair each price with the item it visually belongs to. Report only what you can see: if a price is unreadable or cannot be assigned, set price_vnd to null instead of guessing, a word instead of a number ("tùy loại", "thời giá", "liên hệ") or a missing price also gives price_vnd null with the words in raw_price, and do not invent items.
""" + RETURN_EN,
}

DRAFTS: set[str] = set()   # P1a/b/c were checked against P1 rule by rule on 15/11/2026 (same content, other form): final


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "system_en.txt").write_text(SYSTEM_EN + "\n", encoding="utf-8")
    (OUT / "system_vi.txt").write_text(SYSTEM_VI + "\n", encoding="utf-8")
    (OUT / "schema.txt").write_text(SCHEMA + "\n", encoding="utf-8")
    for pid, text in PROMPTS.items():
        (OUT / f"{pid}.txt").write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {len(PROMPTS)} prompts to {OUT}")


if __name__ == "__main__":
    main()
