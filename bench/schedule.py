# ABOUTME: Lịch gửi chunk theo mốc TUYỆT ĐỐI - tách riêng để test không cần tritonclient
# ABOUTME: Dùng chung cho mọi worker stream, mọi mức CCU


def send_deadlines(t_start, n, chunk_s):
    """Mốc gửi của n chunk, đều nhau chunk_s giây kể từ t_start.

    Tuyệt đối chứ không phải sleep(chunk_s) cộng dồn. Cộng dồn thì overhead của
    chính client (serialize tensor, GC, tranh CPU với server vì bench chạy ngay
    trên Thor) trôi vào mốc gửi, và drift đo được sẽ là drift của client chứ
    không phải của server - hỏng đúng thứ bài bench sinh ra để đo.
    """
    return [t_start + i * chunk_s for i in range(n)]
