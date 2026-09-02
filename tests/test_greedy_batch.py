# ABOUTME: Kiểm greedy_search_batch cho ra ĐÚNG token như greedy_search_step batch-1
# ABOUTME: Dùng decoder/joiner giả, chạy được khi server tắt và không cần weights

import numpy as np
import pytest

from serving.streaming_search import (
    SearchState,
    greedy_search_batch,
    greedy_search_step,
    init_search_state,
)

VOCAB = 2000
DIM = 512
BLANK = 0
CTX = 2


class FakeModel:
    """decoder/joiner giả, tất định theo context và encoder frame.

    Không dùng weights thật vì test này kiểm LOGIC GOM BATCH, không kiểm model.
    Điều cần chứng minh: gom B stream vào một lần gọi cho ra cùng token với gọi
    riêng từng stream - tức không có chỗ nào lệch index giữa states và
    encoder_out. Lệch index là bug tệ nhất của hàm này vì transcript vẫn trông
    hợp lệ, chỉ là lẫn giữa các user.
    """

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.emb = rng.standard_normal((VOCAB, DIM)).astype(np.float32)
        self.proj = rng.standard_normal((DIM, VOCAB)).astype(np.float32) * 0.05
        self.n_dec = 0
        self.n_join = 0

    def decoder_batch(self, ctx):
        self.n_dec += 1
        ctx = np.asarray(ctx, dtype=np.int64)
        return (self.emb[ctx[:, 0]] + self.emb[ctx[:, 1]]).astype(np.float32)

    def joiner_batch(self, enc, dec):
        self.n_join += 1
        return np.tanh(enc + dec) @ self.proj

    # đường batch-1, gọi lại chính hai hàm trên
    def decoder_one(self, context):
        return self.decoder_batch([list(context)])

    def joiner_one(self, enc_frame, dec_out):
        return self.joiner_batch(enc_frame, dec_out)


def make_encoder_out(b, t, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((b, t, DIM)).astype(np.float32) * 2.0


@pytest.mark.parametrize("b,t", [(1, 8), (2, 8), (8, 8), (3, 16)])
def test_batch_khop_batch_mot(b, t):
    enc = make_encoder_out(b, t, seed=b * 100 + t)

    ref_model = FakeModel()
    ref_states = [
        init_search_state(ref_model.decoder_one, BLANK, CTX) for _ in range(b)
    ]
    for i in range(b):
        greedy_search_step(
            enc[i], ref_states[i], ref_model.decoder_one, ref_model.joiner_one, BLANK, CTX
        )

    bat_model = FakeModel()
    bat_states = [
        init_search_state(bat_model.decoder_one, BLANK, CTX) for _ in range(b)
    ]
    greedy_search_batch(
        enc, bat_states, bat_model.decoder_batch, bat_model.joiner_batch, BLANK, CTX
    )

    for i in range(b):
        assert bat_states[i].hyp == ref_states[i].hyp, f"stream {i} lệch token"
        np.testing.assert_allclose(
            bat_states[i].decoder_out, ref_states[i].decoder_out, rtol=1e-5, atol=1e-5
        )


def test_giam_so_lan_goi_joiner():
    """Lý do tồn tại của hàm batch: số lần invoke joiner giảm đúng B lần."""
    b, t = 8, 8
    enc = make_encoder_out(b, t, seed=7)

    ref = FakeModel()
    for i in range(b):
        st = init_search_state(ref.decoder_one, BLANK, CTX)
        greedy_search_step(enc[i], st, ref.decoder_one, ref.joiner_one, BLANK, CTX)

    bat = FakeModel()
    sts = [init_search_state(bat.decoder_one, BLANK, CTX) for _ in range(b)]
    greedy_search_batch(enc, sts, bat.decoder_batch, bat.joiner_batch, BLANK, CTX)

    assert bat.n_join == t, f"joiner phải gọi đúng {t} lần, thực tế {bat.n_join}"
    assert ref.n_join == b * t
    assert bat.n_join * b == ref.n_join


def test_giu_state_qua_nhieu_buoc():
    """Gọi nối tiếp nhiều bước encoder phải giống hệt một lần gọi dài."""
    b, t = 4, 8
    enc = make_encoder_out(b, 2 * t, seed=42)

    one = FakeModel()
    sts_one = [init_search_state(one.decoder_one, BLANK, CTX) for _ in range(b)]
    greedy_search_batch(
        enc, sts_one, one.decoder_batch, one.joiner_batch, BLANK, CTX
    )

    two = FakeModel()
    sts_two = [init_search_state(two.decoder_one, BLANK, CTX) for _ in range(b)]
    greedy_search_batch(
        enc[:, :t], sts_two, two.decoder_batch, two.joiner_batch, BLANK, CTX
    )
    greedy_search_batch(
        enc[:, t:], sts_two, two.decoder_batch, two.joiner_batch, BLANK, CTX
    )

    for i in range(b):
        assert sts_two[i].hyp == sts_one[i].hyp, f"stream {i} lệch khi chia hai bước"


def test_thu_tu_states_phai_khop_encoder_out():
    """Đảo thứ tự states mà không đảo encoder_out thì kết quả PHẢI khác.

    Test này bảo vệ chống một sửa đổi tưởng vô hại trong tương lai: nếu ai đó
    sắp xếp lại states bên trong hàm (gom theo độ dài chẳng hạn) mà quên sắp xếp
    encoder_out theo, hypothesis sẽ lẫn giữa các user và transcript vẫn trông
    hợp lệ. Nếu test này bắt đầu PASS mà không ai cố ý sửa gì, hàm đã hỏng.
    """
    b, t = 4, 8
    enc = make_encoder_out(b, t, seed=11)

    m1 = FakeModel()
    s1 = [init_search_state(m1.decoder_one, BLANK, CTX) for _ in range(b)]
    greedy_search_batch(enc, s1, m1.decoder_batch, m1.joiner_batch, BLANK, CTX)

    m2 = FakeModel()
    s2 = [init_search_state(m2.decoder_one, BLANK, CTX) for _ in range(b)]
    greedy_search_batch(
        enc[::-1], s2, m2.decoder_batch, m2.joiner_batch, BLANK, CTX
    )

    assert [s.hyp for s in s1] != [s.hyp for s in s2]


def test_so_state_lech_thi_bao_loi():
    enc = make_encoder_out(4, 8, seed=1)
    m = FakeModel()
    sts = [init_search_state(m.decoder_one, BLANK, CTX) for _ in range(3)]
    with pytest.raises(ValueError, match="stream"):
        greedy_search_batch(enc, sts, m.decoder_batch, m.joiner_batch, BLANK, CTX)


def test_batch_rong():
    m = FakeModel()
    assert greedy_search_batch(
        np.zeros((0, 8, DIM), np.float32), [], m.decoder_batch, m.joiner_batch
    ) == []


def test_khung_toan_blank_khong_goi_decoder():
    """encoder_out = 0 cho logit gần đều; ép blank bằng cách chặn argmax.

    Chunk im lặng chiếm phần lớn lưu lượng thật, nên đường đi khi cả batch cùng
    phát blank phải rẻ: không gọi decoder lần nào.
    """
    b, t = 4, 8
    enc = np.zeros((b, t, DIM), dtype=np.float32)

    class AlwaysBlank(FakeModel):
        def joiner_batch(self, e, d):
            self.n_join += 1
            out = np.full((e.shape[0], VOCAB), -1.0, dtype=np.float32)
            out[:, BLANK] = 1.0
            return out

    m = AlwaysBlank()
    sts = [init_search_state(m.decoder_one, BLANK, CTX) for _ in range(b)]
    n_dec_sau_init = m.n_dec
    greedy_search_batch(enc, sts, m.decoder_batch, m.joiner_batch, BLANK, CTX)

    assert m.n_dec == n_dec_sau_init, "khung blank không được gọi decoder"
    assert all(s.hyp == [BLANK] * CTX for s in sts)


def test_decoder_out_giu_shape_1_c():
    """SearchState.decoder_out phải luôn là (1, C) để lần sau gom batch được."""
    b, t = 3, 8
    enc = make_encoder_out(b, t, seed=5)
    m = FakeModel()
    sts = [init_search_state(m.decoder_one, BLANK, CTX) for _ in range(b)]
    greedy_search_batch(enc, sts, m.decoder_batch, m.joiner_batch, BLANK, CTX)
    for s in sts:
        assert s.decoder_out.shape == (1, DIM)


def test_khong_chia_se_bo_nho_giua_cac_stream():
    """decoder_out của mỗi stream phải là bản copy riêng, không phải view chung.

    Nếu quên .copy() khi tách dec_out thì mọi stream trỏ vào cùng một buffer:
    lần gọi sau ghi đè và tất cả cùng đổi theo một stream.
    """
    b, t = 3, 8
    enc = make_encoder_out(b, t, seed=9)
    m = FakeModel()
    sts = [init_search_state(m.decoder_one, BLANK, CTX) for _ in range(b)]
    greedy_search_batch(enc, sts, m.decoder_batch, m.joiner_batch, BLANK, CTX)

    truoc = sts[1].decoder_out.copy()
    sts[0].decoder_out[:] = 999.0
    np.testing.assert_array_equal(sts[1].decoder_out, truoc)


def test_state_khong_mang_sang_stream_khac():
    """Chạy batch B rồi chạy riêng từng stream phải ra cùng kết quả tiếp theo."""
    b, t = 4, 8
    enc1 = make_encoder_out(b, t, seed=21)
    enc2 = make_encoder_out(b, t, seed=22)

    m = FakeModel()
    sts = [init_search_state(m.decoder_one, BLANK, CTX) for _ in range(b)]
    greedy_search_batch(enc1, sts, m.decoder_batch, m.joiner_batch, BLANK, CTX)
    hyp_giua = [list(s.hyp) for s in sts]
    greedy_search_batch(enc2, sts, m.decoder_batch, m.joiner_batch, BLANK, CTX)

    m2 = FakeModel()
    for i in range(b):
        st = SearchState(hyp=list(hyp_giua[i]), decoder_out=None)
        st.decoder_out = m2.decoder_one(st.hyp[-CTX:])
        greedy_search_step(
            enc2[i], st, m2.decoder_one, m2.joiner_one, BLANK, CTX
        )
        assert st.hyp == sts[i].hyp, f"stream {i} lệch ở bước thứ hai"
