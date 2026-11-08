# kgrun — chạy script Python trên GPU Kaggle từ VS Code

**Trạng thái:** Hoàn thành toàn bộ (M0, M1, M2, M3) · **Ngày:** 2026-11-08 · **Phạm vi:** chỉ Kaggle, dùng API chính thức

## 1. Mục tiêu

Viết code trong VS Code như bình thường, rồi chạy:

```bash
kgrun run train.py --epochs 30      # đẩy lên Kaggle, chạy trên GPU, stream log về terminal
kgrun pull                          # kéo weights/CSV về máy
```

Trải nghiệm gần với `ssh server "python train.py"`, nhưng không có shell tương tác.

**Thành công khi:** từ một thư mục dự án bất kỳ, `kgrun run check_gpu.py` in ra tên GPU Kaggle
(T4/P100) trong terminal local, và `kgrun pull` tải được file output về `runs/<run-id>/`.

**Không làm:**
- SSH/tunnel (ngrok, cloudflared): có rủi ro vi phạm điều khoản, có thể bị khoá tài khoản.
- Colab.
- Notebook tương tác.
- Hàng đợi nhiều job.
- Giao diện web.

## 2. Vì sao không dùng thẳng tool có sẵn

| Lựa chọn | Đánh giá |
|---|---|
| `kaggle` CLI chính thức (`kernels push/status/output/logs`) | Đủ cơ chế, nhưng mỗi lần chạy phải tự viết `kernel-metadata.json`, chỉ đẩy được **1 file** code, và output phải tự kéo về. → **kgrun bọc lớp mỏng quanh CLI này.** |
| [`kgz`](https://pypi.org/project/kgz/) (MIT, beta, 0.1.4, 04/2026) | Nối WebSocket vào một notebook **đang mở trên trình duyệt**. Tương tác tốt, nhưng phải giữ notebook mở, mới có 1 người bảo trì, và cơ chế chưa rõ có hợp ToS không. Có thể thử như phương án B. |
| SSH qua tunnel | Loại (xem Không làm). |

## 3. Sự thật về nền tảng (đã tra, kèm độ tin cậy)

| Điều | Giá trị | Nguồn / độ tin cậy |
|---|---|---|
| Chọn GPU khi push | `kaggle kernels push --accelerator NvidiaTeslaT4` hoặc field `machine_shape` + `enable_gpu: true` trong `kernel-metadata.json` | Docs chính thức `docs/kernels.md`, `kernels_metadata.md`. Chắc chắn |
| Giới hạn thời gian | `kernels push -t/--timeout <giây>` | Docs chính thức. Chắc chắn |
| Gắn dữ liệu | `dataset_sources: ["user/slug"]`, `model_sources`, `kernel_sources` | Docs chính thức. Chắc chắn |
| Upload dữ liệu | `kaggle datasets create/version -p DIR -r zip\|tar`, `-m` bắt buộc khi version | Docs chính thức. Chắc chắn |
| Log trực tiếp | `kaggle kernels logs --follow` (`-f`) dùng SSE stream. Đã kiểm chứng ở M0: stream log thời gian thực mượt mà cho kernel kiểu `script`. | Đã kiểm chứng thực tế ở M0. Chắc chắn |
| Quota | ~30 h GPU/tuần (động, reset thứ Bảy 00:00 UTC), tối đa 12 h/phiên GPU | Nguồn thứ ba; Kaggle không công bố con số. Gần đúng |
| GPU free | 2× T4 (2×16 GB) khi chọn `NvidiaTeslaT4`. `NvidiaTeslaP100` đã bị Kaggle gắn nhãn retired (cảnh báo khi push). | Đã kiểm chứng thực tế ở M0. Chắc chắn |
| Dùng GPU | Tài khoản phải xác minh số điện thoại. Đã kiểm chứng tài khoản chạy tốt GPU. | Đã kiểm chứng thực tế ở M0 |
| Xác thực | `credentials.json` hiện có trong `~/.kaggle/` được CLI v2.2.5 nhận diện và hoạt động trơn tru (user: `huytu2025`). | Đã kiểm chứng thực tế ở M0 |
| Huỷ job đang chạy | **Không có** lệnh cancel/stop trong CLI v2.2.5 (chỉ có `delete`). Dùng `--timeout` làm cầu chì hoặc huỷ trên web | Đã kiểm chứng CLI help ở M0. Chắc chắn |
| Giới hạn code_file | API `SaveKernel` giới hạn cứng đúng **1 MiB (1,048,576 bytes)** cho `code_file`. Vì nhúng base64 tăng ~33% dung lượng, file `tar.gz` đóng gói source code phải **≤ 750 KB**. | Đã probe thực tế ở M0. Chắc chắn |

Máy đã cài `kaggle` CLI (v2.2.5). Đã xác thực thành công qua `~/.kaggle/credentials.json` (tài khoản `huytu2025`).

## 4. Thiết kế

### 4.1 Cấu hình theo dự án: `kgrun.toml` ở thư mục gốc dự án

```toml
kernel     = "xuantu2002/fish-train"     # slug kernel; mỗi lần run = một version mới
accelerator = "NvidiaTeslaT4"            # hoặc "NvidiaTeslaP100"
timeout_h  = 11                          # cầu chì < 12 h
internet   = true                        # cần nếu pip install thêm gói
datasets   = ["xuantu2002/fish-ffe"]     # dữ liệu đã upload sẵn
include    = ["*.py", "src/**", "configs/**", "requirements-kaggle.txt"]
exclude    = ["venv/**", "data/**", "runs/**", "**/__pycache__/**"]
pip        = "requirements-kaggle.txt"   # tuỳ chọn; chỉ gói mà image Kaggle thiếu
```

Tạo bằng `kgrun init`, mặc định lấy username từ CLI. Không có file cấu hình toàn cục: mọi
cấu hình đều nằm trong file của dự án.

### 4.2 Luồng `kgrun run <script.py> [args...]`

1. **Đóng gói code.** Gom các file khớp `include` mà không khớp `exclude` thành tar.gz rồi
   mã hoá base64. Nếu file tar.gz vượt ngưỡng **750 KB** (để đảm bảo file bootstrap chứa base64
   không vượt trần 1 MiB của Kaggle API) thì dừng và báo lỗi, gợi ý đưa file lớn vào dataset.
2. **Sinh bootstrap.** Tạo một file `kgrun_entry.py` duy nhất, nhúng sẵn chuỗi base64 ở bước 1.
   Khi chạy trên Kaggle, file này:
   - in **stamp tái lập**: thời điểm, tên GPU (`nvidia-smi`), phiên bản torch/CUDA/timm,
     lệnh và args chính xác, mã băm của gói code; đồng thời ghi stamp vào `/kaggle/working/kgrun_stamp.json`;
   - giải nén code vào `/kaggle/tmp/src` để output không bị lẫn source;
   - nếu cấu hình có `pip` thì chạy `pip install -r ...`;
   - đặt các biến môi trường: `KGRUN_OUT=/kaggle/working` và `KGRUN_DATA=<đường dẫn mount
     của dataset đầu tiên>`. Đường dẫn mount được **tìm bằng glob** dưới `/kaggle/input`,
     vì cấu trúc thư mục mount có thể đổi;
   - chạy `python -u script.py args` bằng `subprocess`, đẩy thẳng stdout/stderr ra ngoài,
     và trả về đúng exit code của script.
3. **Sinh metadata.** Tạo `kernel-metadata.json` gồm: `id`, `code_file=kgrun_entry.py`,
   `kernel_type=script`, `is_private=true`, `enable_gpu`, `machine_shape`, `enable_internet`
   và `dataset_sources`.
4. **Đẩy lên.** Chạy `kaggle kernels push -p <staging> --accelerator ... -t ...`. Thư mục
   staging nằm trong thư mục tạm của hệ điều hành, không làm bẩn dự án.
5. **Theo dõi.** Mặc định gọi `kaggle kernels logs <kernel> --follow`. Nếu M0 cho thấy lệnh
   này không dùng được thì chuyển sang fallback: cứ 30 giây gọi `kernels status` một lần, và
   in toàn bộ log khi job kết thúc. Nhấn Ctrl+C chỉ ngắt việc theo dõi, **job trên Kaggle vẫn
   chạy**. kgrun sẽ in rõ câu này cho người dùng.
6. **Kết thúc.** In trạng thái cuối và thời gian GPU đã dùng. Nếu có `--pull` thì tự kéo output về.

Code trong dự án chỉ cần đổi một chỗ: ghi output vào `os.environ.get("KGRUN_OUT", "outputs")`
và đọc dữ liệu từ `KGRUN_DATA`. Nhờ đó cùng một script chạy được cả trên máy, trên server lẫn trên Kaggle.

### 4.3 Các lệnh

| Lệnh | Việc |
|---|---|
| `kgrun init` | Tạo `kgrun.toml`; kiểm tra CLI đã cài và đã xác thực |
| `kgrun run SCRIPT [ARGS] [--gpu t4\|p100\|none] [--no-follow] [--pull]` | Như mục 4.2 |
| `kgrun logs [-f]` | Xem log của lần chạy gần nhất |
| `kgrun status` | Trạng thái kernel |
| `kgrun pull [--pattern '*.pth']` | `kaggle kernels output` về `runs/<YYYY-MM-DD_HHMMSS>/`; **không ghi đè** thư mục đã có |
| `kgrun data push DIR [-m msg]` | Lần đầu thì `datasets create`, các lần sau thì `datasets version`, dùng `-r tar`; chờ `datasets status` báo `ready` |

`kgrun` ghi thông tin từng lần chạy (run-id, version, args, thời điểm) vào `.kgrun/history.jsonl` trong dự án.

### 4.4 Cài đặt và cấu trúc mã

- Gồm một gói `kgrun/` (~300 dòng): `cli.py` (argparse), `pack.py`, `bootstrap_template.py`.
- Phụ thuộc: `kaggle>=2.2.3` và stdlib (`tomllib`, `tarfile`, `base64`, `subprocess`).
- Cài toàn cục bằng `pipx install -e D:\Desktop\GPU`. Lệnh `kgrun` chạy được ở mọi dự án,
  cả PowerShell lẫn Git Bash.
- kgrun **gọi CLI `kaggle` qua subprocess**, không import SDK nội bộ, để giảm rủi ro khi
  API thay đổi.

## 5. Giới hạn cần biết trước

- Mỗi lần chạy phải chờ khởi động máy ảo, khoảng 1–3 phút. Không hợp kiểu sửa một dòng rồi chạy lại ngay.
- Không có shell tương tác, không debug được giữa chừng. Nên chạy thử trên CPU ở máy trước.
- Mỗi kernel slug chỉ có một job chạy tại một thời điểm. Muốn chạy song song phải dùng slug khác.
  Số phiên GPU được chạy đồng thời **chưa xác minh**.
- Output trong `/kaggle/working` bị giới hạn dung lượng (khoảng 20 GB, chưa xác minh). Nếu
  hết 12 giờ giữa chừng thì phải tự lưu checkpoint thường xuyên.
- Image Kaggle có sẵn torch/timm, nhưng phiên bản không do mình chọn. Stamp ghi lại phiên
  bản thực tế, phục vụ yêu cầu tái lập.

## 6. Kế hoạch thực hiện

| Mốc | Việc | Kiểm chứng |
|---|---|---|
| **M0** Kiểm tra nền tảng | `pip install kaggle`; xác thực (`credentials.json` hiện có, hoặc tạo token mới); xác minh số điện thoại; push tay một `check_gpu.py` | **Đã hoàn thành:** Kiểm chứng Kaggle CLI v2.2.5, cấp 2× Tesla T4, stream SSE thời gian thực, giới hạn code_file đúng 1 MiB (tar.gz ≤ 750 KB). |
| **M1** `run` + `pull` | Đóng gói code, sinh bootstrap và metadata, push, theo dõi log | **Đã hoàn thành:** Triển khai gói `kgrun/` hoàn chỉnh. Đã kiểm thử thực tế trên `examples/sample_project/` (train.py, utils.py, config.py), stream log mượt mà, kéo output `kgrun_stamp.json` và kết quả model về `runs/<timestamp>/`. |
| **M2** `data push` | Tạo/version dataset; `KGRUN_DATA` trỏ đúng chỗ | **Đã hoàn thành:** Triển khai lệnh `kgrun data push`. Đã upload thành công 162 ảnh từ `data/` lên dataset `huytu2025/fish-data`. Script `check_dataset.py` chạy trên GPU nhận đúng mount `/kaggle/input/datasets/huytu2025/fish-data`, đọc ảnh mẫu kích thước 225x222 và kéo kết quả về máy. |
| **M3** Dùng thử thật | Chạy một job train ngắn (≤10 phút GPU) của dự án fish | **Đã hoàn thành:** Chạy thành công `train_fish.py` trên 2× Tesla T4 với dataset FFE (`huytu2025/fish-freshness-images`). Quá trình huấn luyện hội tụ (train_loss giảm 1.0094 → 0.0392), live logs stream mượt mà, tự động kéo checkpoint `best_fish_model.pth` (44.8 MB) và `fish_metrics.csv` về `runs/<timestamp>/`. |

Ước lượng: M0 khoảng 30 phút (phần lớn là thao tác tay), M1 và M2 khoảng một buổi.

## 7. Kết quả giải đáp các câu hỏi kỹ thuật (Đã xác minh ở M0)

1. **`kaggle kernels logs --follow` (`-f`)**: Hoạt động hoàn hảo với kernel `script`. Nhận log SSE thời gian thực và tự động thoát khi job hoàn thành.
2. **Giới hạn kích thước `code_file`**: Giới hạn chính xác của endpoint `SaveKernel` là **1 MiB (1,048,576 bytes)**. Gói code nén `tar.gz` trước khi encode base64 vào bootstrap file phải **≤ 750 KB**. Nếu vượt quá ngưỡng này, `kgrun` sẽ chặn sớm và gợi ý đưa thư mục lớn vào Kaggle Dataset.
3. **Lệnh huỷ job**: Kaggle CLI không hỗ trợ lệnh cancel/stop (`kaggle kernels` chỉ có `delete`). Biện pháp kiểm soát thời gian là cờ `-t/--timeout` (tính bằng giây) và huỷ thủ công trên giao diện web Kaggle.
4. **Xác thực CLI**: File `~/.kaggle/credentials.json` có sẵn hoạt động trực tiếp với Kaggle CLI v2.2.5, không cần tạo lại token hay cấu hình thêm.
5. **Định dạng file JSON**: Khi sinh `kernel-metadata.json`, **bắt buộc dùng UTF-8 không có BOM**, nếu có BOM thì parser của Kaggle CLI sẽ văng lỗi `Unexpected UTF-8 BOM`.
6. **Bộ gia tốc GPU**: `--accelerator NvidiaTeslaT4` cấp 2× Tesla T4 (mỗi card 15 GB VRAM). `NvidiaTeslaP100` hiện đã bị Kaggle gắn nhãn retired và có cảnh báo khi sử dụng.
7. **Cấu trúc mount Dataset**: Kaggle mount dataset theo đường dẫn lồng `/kaggle/input/datasets/<username>/<dataset-name>/`. Bootstrap của `kgrun` sử dụng thuật toán duyệt cây thư mục tự động tìm đến tầng chứa dữ liệu thực tế và gán vào `KGRUN_DATA`.
8. **Độ dài tiêu đề Kernel**: Kaggle API bắt buộc `title` trong metadata phải dài ít nhất **5 ký tự** (`Title must be at least five characters`). `kgrun` đã được bổ sung cơ chế tự động đệm hậu tố nếu tên thư mục gốc dưới 5 ký tự.

## Nguồn

- Kaggle CLI docs: [kernels.md](https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md),
  [kernels_metadata.md](https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md),
  [datasets.md](https://github.com/Kaggle/kaggle-api/blob/main/docs/datasets.md)
- Live logs: [kaggle-cli PR #999](https://github.com/Kaggle/kaggle-cli/pull/999) ·
  [releases](https://github.com/Kaggle/kaggle-cli/releases)
- Token mới: [kaggle-api PR #863](https://github.com/Kaggle/kaggle-api/pull/863)
- Quota: [gpuperhour.com](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- Phương án B: [kgz trên PyPI](https://pypi.org/project/kgz/)
