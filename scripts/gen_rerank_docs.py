#!/usr/bin/env python3
# ABOUTME: Sinh scripts/hello.json - bộ doc cho bench reranker, phân bố độ dài mô phỏng chunk RAG
# ABOUTME: Chạy: python3 scripts/gen_rerank_docs.py --out scripts/hello.json --verify-url http://127.0.0.1:9012

"""Vì sao phải sinh chứ không viết tay, và vì sao phải quan tâm ĐỘ DÀI.

Thứ quyết định độ trễ của reranker không phải nội dung mà là **số token sau khi
pad**. Backend python của TEI không có flash-attn nên pad mọi sequence trong
một batch lên bằng sequence dài nhất; đo được: cùng 33.2k token, doc dài ngắn
lẫn lộn tốn 998ms còn doc đều nhau chỉ 518ms (1.93x). Xem bench/RERANK.md.

Hệ quả: một bộ doc mà mọi đoạn dài bằng nhau - đúng thứ `rerank_probe.make_docs`
sinh ra - sẽ báo số ĐẸP HƠN SỰ THẬT gần hai lần. Bộ doc dùng để ra ngân sách
cho production bắt buộc phải RÁP ĐỘ DÀI giống dữ liệu thật.

Vì sao ghép từ mẫu câu chứ không xoay vòng vài đoạn có sẵn
----------------------------------------------------------
`make_docs` xoay vòng qua 8 câu mồi, nên sinh k>8 là doc i và doc i+8 trùng
khít. Trùng thì tokenizer và cache của model gặp lại đúng chuỗi cũ, và bộ eval
chất lượng thành vô nghĩa. Ở đây mỗi câu ráp từ khuôn + nhiều bộ điền nên số
câu khác nhau lên tới hàng nghìn; kiểm trùng ở cuối là chốt chặn.

PHÂN BỐ MẶC ĐỊNH LÀ GIẢ ĐỊNH, KHÔNG PHẢI SỐ ĐO
----------------------------------------------
median 300 token, p95 700, chặn ở [40, 900] - dáng lognormal thường thấy của
chunk RAG. Khi có histogram thật từ Milvus thì đổi --median/--p95 rồi sinh lại;
mọi con số bench trước đó phải đo lại, vì chúng là hàm của phân bố này.
"""

import argparse
import json
import math
import random
import sys
import urllib.request
from pathlib import Path

# 4 đoạn smoke test, nay ghi ra FILE RIÊNG (scripts/smoke.json).
#
# Trước đây chúng nằm đầu hello.json để giữ phép thử tính đúng, và đó là một
# lỗi đo: `load_file` lấy k đoạn ĐẦU, nên ở top-k 10 thì 4/10 doc là hàng
# 26-28 token, kéo trung bình xuống 172 token/doc trong khi top-k 100 là 356.
# Bảng vì thế đẹp hơn sự thật ở k nhỏ và dốc hơn sự thật khi k tăng. Tách ra
# thì mỗi file làm đúng một việc.
QUERY = "Thủ đô của Việt Nam là thành phố nào?"
SMOKE = [
    "Con mèo đang nằm ngủ trên chiếc ghế sofa màu xanh ở phòng khách.",
    "Hà Nội là thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam.",
    "Thành phố Hồ Chí Minh là trung tâm kinh tế lớn nhất của cả nước.",
    "Paris là thủ đô của nước Pháp, nổi tiếng với tháp Eiffel.",
]

# Đo trên chính tokenizer của model: 120 từ -> 133 token phần doc. Chỉ để ƯỚC
# số từ cần sinh; số token THẬT lấy bằng --verify-url chứ không tin hằng số này.
TOKENS_PER_WORD = 1.11

FILL = {
    "vb": ["Luật Doanh nghiệp", "Nghị định 01/2021/NĐ-CP", "Thông tư 105/2020/TT-BTC",
           "Luật Quản lý thuế", "Bộ luật Lao động", "Luật Đất đai",
           "Nghị quyết của Hội đồng nhân dân tỉnh", "quy chế nội bộ của đơn vị"],
    "ai": ["chủ hộ kinh doanh", "người đại diện theo pháp luật", "người nộp thuế",
           "đơn vị sử dụng lao động", "cơ quan đăng ký kinh doanh cấp huyện",
           "tổ chức hành nghề công chứng", "người sử dụng đất", "bên nhận chuyển nhượng"],
    "lam": ["nộp hồ sơ trực tiếp hoặc qua cổng dịch vụ công",
            "kê khai đầy đủ thông tin theo mẫu quy định",
            "lưu giữ chứng từ gốc tại trụ sở chính",
            "thông báo bằng văn bản cho cơ quan có thẩm quyền",
            "công bố thông tin trên trang thông tin điện tử",
            "hoàn tất nghĩa vụ tài chính trước khi nhận kết quả",
            "cập nhật thay đổi vào cơ sở dữ liệu quốc gia"],
    "sn": ["ba", "năm", "bảy", "mười", "mười lăm", "hai mươi", "ba mươi", "sáu mươi"],
    "moc": ["nhận đủ hồ sơ hợp lệ", "phát sinh thay đổi", "kết thúc kỳ tính thuế",
            "ký kết hợp đồng", "được cấp giấy chứng nhận", "có quyết định của cơ quan thuế"],
    "benh": ["tăng huyết áp", "đái tháo đường týp hai", "viêm dạ dày mạn tính",
             "thiếu máu thiếu sắt", "rối loạn lo âu", "thoái hóa khớp gối"],
    "yte": ["duy trì chế độ ăn giảm muối và tăng rau xanh",
            "tái khám định kỳ mỗi ba tháng một lần",
            "tuân thủ liều thuốc do bác sĩ chỉ định",
            "theo dõi chỉ số tại nhà và ghi vào sổ",
            "hạn chế rượu bia và ngừng hút thuốc lá"],
    "mon": ["phở bò", "bún chả", "cơm tấm sườn nướng", "bánh xèo miền Tây",
            "mì Quảng", "canh chua cá lóc", "gỏi cuốn tôm thịt"],
    "nau": ["ninh xương ống nhiều giờ cho nước dùng vừa trong vừa ngọt",
            "ướp thịt với hành tím và nước mắm ngon khoảng một giờ",
            "chiên ở lửa vừa để vỏ giòn mà bên trong không bị khô",
            "nêm nếm lại lần cuối trước khi tắt bếp",
            "dùng kèm rau sống và nước chấm pha loãng"],
    "noi": ["Vịnh Hạ Long", "phố cổ Hội An", "cao nguyên đá Đồng Văn",
            "đồng bằng sông Cửu Long", "đảo Phú Quốc", "thành phố Đà Lạt"],
    "dl": ["thu hút đông khách vào mùa cao điểm từ tháng sáu đến tháng tám",
           "có hệ thống lưu trú trải rộng từ bình dân tới cao cấp",
           "được nhiều tạp chí quốc tế xếp vào danh sách nên tới một lần",
           "kết nối thuận tiện bằng đường bộ và đường hàng không"],
    "cn": ["dịch vụ suy luận mô hình ngôn ngữ", "hệ thống tìm kiếm ngữ nghĩa",
           "cụm cơ sở dữ liệu vector", "hàng đợi xử lý bất đồng bộ",
           "lớp bộ nhớ đệm phân tán"],
    "kt": ["giảm độ trễ đuôi ở phân vị thứ chín mươi chín",
           "tăng thông lượng mà không nới thêm phần cứng",
           "cắt chi phí vận hành trên mỗi nghìn yêu cầu",
           "giữ tỷ lệ lỗi dưới ngưỡng cam kết trong giờ cao điểm"],
    "tt": ["Nam Bộ", "Bắc Bộ", "Trung Trung Bộ", "Tây Nguyên", "vùng núi phía Bắc"],
    "tho": ["có mưa rào và dông vài nơi", "trời nắng gián đoạn, gió nhẹ",
            "sáng sớm có sương mù nhẹ", "nhiệt độ giảm sâu về đêm và sáng sớm"],
}

TEMPLATES = {
    "hanhchinh": [
        "Theo {vb}, {ai} phải {lam} trong thời hạn {sn} ngày làm việc kể từ ngày {moc}.",
        "Hồ sơ do {ai} nộp được tiếp nhận và trả kết quả trong {sn} ngày làm việc kể từ ngày {moc}.",
        "Trường hợp hồ sơ chưa hợp lệ, cơ quan tiếp nhận hướng dẫn {ai} bổ sung một lần duy nhất bằng văn bản.",
        "{vb} quy định rõ {ai} có trách nhiệm {lam} và chịu trách nhiệm về tính chính xác của thông tin đã kê khai.",
        "Sau khi {moc}, {ai} tiếp tục {lam} theo hướng dẫn của cơ quan quản lý chuyên ngành.",
    ],
    "yte": [
        "Người bệnh {benh} nên {yte} để hạn chế biến chứng về lâu dài.",
        "Trong theo dõi {benh}, bác sĩ thường khuyên người bệnh {yte}.",
        "Phác đồ điều trị {benh} cần được cá thể hóa, kết hợp thay đổi lối sống và {yte}.",
        "Tỷ lệ mắc {benh} trong cộng đồng có xu hướng tăng ở nhóm tuổi trung niên.",
    ],
    "amthuc": [
        "Muốn nấu {mon} ngon thì phải {nau}.",
        "{mon} là món quen thuộc trong bữa sáng của nhiều gia đình Việt Nam.",
        "Bí quyết của {mon} nằm ở khâu chuẩn bị nguyên liệu và {nau}.",
        "Mỗi vùng miền lại có cách biến tấu {mon} theo khẩu vị riêng.",
    ],
    "dulich": [
        "{noi} {dl}.",
        "Du khách tới {noi} thường dành trọn hai ngày để đi hết các điểm chính.",
        "{noi} là một trong những điểm đến {dl}.",
    ],
    "congnghe": [
        "Đội kỹ thuật đang tối ưu {cn} nhằm {kt}.",
        "Việc tách {cn} thành thành phần độc lập giúp {kt}.",
        "Báo cáo tuần ghi nhận {cn} đã {kt} sau đợt điều chỉnh cấu hình.",
    ],
    "thoitiet": [
        "Khu vực {tt} ngày mai {tho}, nhiệt độ cao nhất phổ biến ba mươi hai độ.",
        "Dự báo trong hai ngày tới, {tt} {tho}.",
        "Người dân {tt} được khuyến cáo theo dõi bản tin dự báo trước khi ra khơi.",
    ],
}


def lognormal_tokens(rng, median, p95, lo, hi):
    """Rút độ dài mục tiêu theo lognormal khớp (median, p95), chặn hai đầu.

    Lognormal chứ không phải đều: chunk RAG thật lệch phải - rất nhiều đoạn
    trung bình, một cái đuôi dài. Chính cái đuôi đó tạo ra phần pad đắt tiền,
    nên phân bố đều sẽ giấu mất đúng thứ cần đo.
    """
    sigma = math.log(p95 / median) / 1.6448536269514722   # z(0.95)
    return int(min(hi, max(lo, round(median * math.exp(sigma * rng.gauss(0, 1))))))


def make_doc(rng, target_words):
    """Ghép câu trong CÙNG một chủ đề tới khi đủ dài - đọc vào còn ra một đoạn
    mạch lạc, chứ không phải rổ câu ghép ngẫu nhiên."""
    topic = rng.choice(list(TEMPLATES))
    out, seen = [], set()
    while sum(len(s.split()) for s in out) < target_words:
        tpl = rng.choice(TEMPLATES[topic])
        sent = tpl.format(**{k: rng.choice(v) for k, v in FILL.items()})
        if sent in seen and len(seen) < 400:
            continue
        seen.add(sent)
        out.append(sent)
    return " ".join(" ".join(out).split()[:target_words])


def token_lengths(url, query, texts):
    """Độ dài token THẬT, hỏi chính tokenizer của server. Hằng số ước lượng
    trong file này chỉ để sinh; con số đưa vào báo cáo phải là con số này."""
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def one(t):
        req = urllib.request.Request(url + "/tokenize",
                                     data=json.dumps({"inputs": t}).encode(),
                                     headers={"Content-Type": "application/json"})
        return len(json.load(op.open(req, timeout=30))[0])
    return [one(query + " " + t) for t in texts]


def pct(xs, p):
    s = sorted(xs)
    return s[min(max(1, math.ceil(p / 100 * len(s))), len(s)) - 1]


def main():
    ap = argparse.ArgumentParser(description="generate a reranker bench document set")
    ap.add_argument("--out", default="scripts/hello.json")
    ap.add_argument("--smoke-out", default="scripts/smoke.json",
                    help="4-doc smoke-test file; --smoke-out '' to skip")
    ap.add_argument("--n", type=int, default=128,
                    help="doc count; must be >= the largest top-k you plan to bench (default covers 128)")
    ap.add_argument("--median", type=int, default=300, help="tokens/doc at the median")
    ap.add_argument("--p95", type=int, default=700, help="tokens/doc at the 95th percentile")
    ap.add_argument("--min-tokens", type=int, default=40)
    ap.add_argument("--max-tokenthor-model-servings", type=int, default=900)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--verify-url", help="e.g. http://127.0.0.1:9012 - count real tokens via /tokenize")
    args = ap.parse_args()


    rng = random.Random(args.seed)
    texts = []
    while len(texts) < args.n:
        tok = lognormal_tokens(rng, args.median, args.p95, args.min_tokens, args.max_tokens)
        texts.append(make_doc(rng, max(8, round(tok / TOKENS_PER_WORD))))

    # Xáo trộn để MỌI lát top-k đều cùng phân bố độ dài. Không xáo thì thứ tự
    # sinh ra là thứ tự rút mẫu, và bench lấy k đoạn đầu sẽ đo một mẫu con
    # không đại diện - đúng cái lỗi vừa tách smoke test ra để tránh.
    rng.shuffle(texts)

    # Trùng đoạn là hỏng cả bộ: model gặp lại đúng chuỗi cũ và bộ eval chất
    # lượng mất nghĩa. Chốt chặn ở đây chứ không để phát hiện lúc đọc bảng.
    dup = len(texts) - len(set(texts))
    if dup:
        raise SystemExit(f"{dup} duplicate docs - widen the FILL pools and regenerate")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"query": QUERY, "texts": texts, "raw_scores": False},
        ensure_ascii=False, indent=2), encoding="utf-8")

    if args.smoke_out:
        Path(args.smoke_out).write_text(json.dumps(
            {"query": QUERY, "texts": SMOKE, "raw_scores": False},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.smoke_out}: {len(SMOKE)} smoke-test docs", file=sys.stderr)

    words = [len(t.split()) for t in texts]
    print(f"wrote {args.out}: {len(texts)} docs, {Path(args.out).stat().st_size / 1024:.0f} KB",
          file=sys.stderr)
    print(f"  words/doc  min {min(words)} | p50 {pct(words, 50)} | "
          f"p95 {pct(words, 95)} | max {max(words)}", file=sys.stderr)
    if args.verify_url:
        toks = token_lengths(args.verify_url, QUERY, texts)
        print(f"  tokens/pair min {min(toks)} | p50 {pct(toks, 50)} | "
              f"p95 {pct(toks, 95)} | max {max(toks)} | total {sum(toks)}", file=sys.stderr)


if __name__ == "__main__":
    main()
