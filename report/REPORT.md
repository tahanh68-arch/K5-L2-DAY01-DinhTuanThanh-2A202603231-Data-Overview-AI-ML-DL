# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:**  T4 GPU

**Python / PyTorch / Ultralytics:** `3.13.16` / `2.11.0+cu130` / `8.4.145`.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`.

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`. Ô so sánh detection trên ảnh `kitchen` dùng các ngưỡng `0.20`, `0.35`, `0.60` theo notebook.

**Thay đổi so với notebook nguồn:** không thay đổi checkpoint, phiên bản Ultralytics hoặc ngưỡng của các tác vụ trong kết quả được cung cấp.

**Dữ liệu và bằng chứng:** notebook lớp `v1.0.3`; ba ảnh COCO công khai `traffic` (210273), `kitchen` (397133), `dining` (166918). Log tải ảnh và checkpoint đều báo kiểm tra SHA-256 `PASS`. Phần 1 dùng `traffic`; phần 2–3 dùng `kitchen`. Ô validation cuối phần tạo evidence báo: `PASS: đủ 3 JSON, 3 PNG, attribution và instance_id duy nhất.`

### Từ cần nhớ

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model dự đoán; chưa phải nhãn đã được xác nhận đúng. |
| Ground truth | Nhãn do con người tạo và kiểm tra theo guideline. |
| Taxonomy | Danh sách lớp và ánh xạ mã lớp đi kèm dữ liệu/checkpoint. |
| Checkpoint | File model `.pt` đã được huấn luyện. |
| Guideline | Quy tắc gán nhãn để nhiều người xử lý nhất quán. |
| Annotator | Người gán nhãn theo guideline. |
| Reviewer / QC | Người kiểm tra nhãn và yêu cầu sửa nếu cần. |
| Escalation | Đưa ca khó hoặc chưa có quy tắc lên Lab Coach/người phụ trách để được quyết định. |
| Model score | Giá trị dùng xếp hạng hoặc lọc prediction; không phải điểm chất lượng nhãn người. |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `day1_lab_outputs/visuals/classification_top5.png`.

**Ảnh sử dụng:** `sample_id = traffic`, chỉ lấy record `rank = 1`.

1. **Record hạng 1:** `class_id = 468`; `class_name = cab`; `rank = 1`; `score = 0.510915`; `taxonomy_name = ImageNet-1K`. Ảnh có kích thước `640 × 428` pixel, COCO image ID `210273`. Record được lưu trong output của ô classification và khớp biểu đồ top-5.

2. **Phạm vi của record:** record nói về toàn bộ bức ảnh, vì tác vụ image classification xếp hạng lớp cho ảnh đầu vào và không trả về tọa độ của một vật thể cụ thể. Trong ảnh có nhiều xe; record `cab` không xác định một chiếc xe nào là taxi.

3. **Nguồn của class list:** tên lớp và mã lớp đến từ taxonomy ImageNet-1K đi kèm checkpoint `yolo11n-cls.pt`; model không tự nghĩ ra một danh sách lớp mới khi nhìn ảnh. Ảnh nguồn thuộc COCO nhưng model phân loại vẫn dùng taxonomy ImageNet-1K.

4. **Lý do ghi đủ ID, tên và taxonomy:** `class_id` là mã trong một hệ lớp cụ thể, `class_name` giúp con người đọc, còn `taxonomy_name` xác định hệ lớp để giải nghĩa mã đó. Ví dụ ID `468` trong record này phải được hiểu trong ImageNet-1K; không dùng mã đó để tra lớp của checkpoint COCO-80.

5. **Guideline cho ảnh có nhiều chủ đề:** cần quy định chọn lớp theo đối tượng/chủ đề chính nào, tiêu chí ưu tiên khi có nhiều đối tượng và cách xử lý ảnh không đủ rõ để chọn một lớp. Ảnh `traffic` có nhiều xe buýt và ô tô, nên không thể lấy số lượng vật thể hoặc top-1 của model làm quy tắc chọn lớp nếu guideline chưa quy định. Ca không chọn được một lớp nhất quán cần escalation.

6. **Vì sao score không phải ground truth:** `0.510915` chỉ là model score của lớp `cab`. Nó không xác nhận rằng ảnh đã được người kiểm tra gán nhãn đúng là `cab`, cũng không phải xác suất đúng đã được bảo đảm cho trường hợp này. Quan sát ảnh cho thấy một cảnh giao thông có nhiều loại xe; cần đối chiếu taxonomy và guideline trước khi xác nhận nhãn.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `day1_lab_outputs/visuals/detection_predictions.png`.

**Ảnh sử dụng:** `sample_id = kitchen`, COCO image ID `397133`, kích thước `640 × 427` pixel. JSON có `11` prediction của ảnh này ở ngưỡng `0.35`; tổng `53` prediction trong JSON là của cả ba ảnh.

1. **Một record được chọn:** `class_id = 0`; `class_name = person`; `score = 0.912625`; `bbox_xyxy = [385.33, 69.24, 498.92, 348.92]`; `bbox_width = 113.58`; `bbox_height = 279.68`; `taxonomy_name = COCO-80`; `coordinate_unit = pixel`; `bbox_format = xyxy`; `score_threshold = 0.35`.

2. **Giải thích box:** gốc tọa độ nằm ở góc trên trái của ảnh; x tăng sang phải và y tăng xuống dưới. Box bao người đứng ở phía phải ảnh, từ góc trên trái `(385.33, 69.24)` đến góc dưới phải `(498.92, 348.92)`, rộng `113.58` pixel và cao `279.68` pixel theo JSON. Các tọa độ và kích thước được làm tròn độc lập từ giá trị gốc, nên lấy hai tọa độ đã làm tròn trừ nhau có thể lệch `0.01` pixel so với trường kích thước; giữ nguyên số liệu trong record.

3. **So sánh threshold trên cùng ảnh `kitchen`:**

   | Ngưỡng | Số prediction trong output ô so sánh |
   | --- | ---: |
   | `0.20` | 17 |
   | `0.35` | 11 |
   | `0.60` | 6 |

   So sánh `0.20` với `0.60`, ngưỡng thấp hơn giữ tổng số prediction nhiều hơn `11`. Đây là số prediction model trả về, không phải số vật thể thật đã được con người xác nhận.

4. **Việc reviewer phải làm khi hạ ngưỡng:** kiểm tra thêm các hộp có score thấp, đối chiếu lớp với vật thể trong ảnh, tìm hộp trùng hoặc sai vị trí và loại prediction không có đủ bằng chứng. Reviewer cũng phải nhìn ảnh để kiểm tra vật thể bị bỏ sót; không chỉ duyệt những hộp model đã vẽ. Threshold lọc prediction không phải quy tắc bỏ vật thể khi tạo ground truth.

5. **Một quy tắc box đề xuất:** với guideline chọn gán nhãn phần nhìn thấy, vẽ hình chữ nhật nhỏ nhất bao kín phần nhìn thấy của từng đối tượng cần gán nhãn, giữ tọa độ trong ảnh và không tự kéo hộp ra ngoài khung để đoán phần bị cắt. Mỗi đối tượng có hộp riêng. Quy tắc này cần được người phụ trách xác nhận trong guideline trước khi áp dụng; nếu dự án yêu cầu hộp bao toàn bộ vật thể kể cả phần bị che thì phải dùng quy tắc tương ứng của dự án.

6. **Trường hợp cắt mép và quyết định xử lý:** ở mép trái ảnh có một phần người/tay xuất hiện gần bàn. Model tạo record `person`, `score = 0.610931`, box `[0.12, 263.18, 61.55, 310.78]`. Box bắt đầu sát mép trái và ảnh không cho thấy toàn bộ người. Nếu guideline đã quy định nhãn cho người chỉ lộ một phần và cách vẽ box phần nhìn thấy thì áp dụng quy tắc đó. Nếu chưa quy định mức nhìn thấy tối thiểu hoặc có gán nhãn riêng phần tay hay không, cần escalation lên Lab Coach, không tự suy ra cơ thể ngoài ảnh hay coi prediction `person` là đáp án chuẩn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `day1_lab_outputs/visuals/segmentation_prediction.png`.

**Ảnh sử dụng:** `sample_id = kitchen`. JSON có `11` instance của ảnh này ở ngưỡng `0.35`; tổng `49` instance trong JSON là của cả ba ảnh.

1. **Một record được chọn:** `instance_id = kitchen-001`; `class_id = 0`; `class_name = person`; `score = 0.899318`; `polygon_point_count = 348`; `taxonomy_name = COCO-80`. Năm điểm đầu của `polygon_xy` theo pixel là:

   ```json
   [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
   ```

   Đây là một phần của polygon gồm 348 điểm, không phải toàn bộ mask. Record có box `[385.45, 66.44, 498.02, 348.58]` và mô tả người đứng phía phải ảnh.

2. **Thông tin polygon thêm so với box:** box chỉ cho biết phạm vi hình chữ nhật bao đối tượng, còn polygon mô tả biên và hình dạng của vùng đối tượng. Ở người đứng phía phải, polygon có thể bám theo đầu, tay, thân và chân, giúp phân biệt vùng người với nền quanh người hoặc giữa hai chân; box không thể biểu diễn chi tiết đó. Polygon vẫn là prediction nên cần kiểm tra bằng mắt.

3. **Ý nghĩa của `instance_id`:** ID giúp phân biệt từng đối tượng trong bộ output lab. `kitchen-001` và `kitchen-009` đều có `class_id = 0`, `class_name = person`, nhưng là hai instance riêng. ID này không phải mã lớp, vì hai instance có thể cùng lớp; cũng không phải tracking ID, vì notebook không theo dõi một người qua chuỗi khung hình. Khi chạy lại, không được mặc định thứ tự ID vẫn nhận diện cùng một đối tượng nếu chưa đối chiếu vị trí và ảnh.

4. **Một quy tắc vẽ mask đề xuất:** với guideline gán nhãn phần nhìn thấy, đặt polygon theo biên nhìn thấy của từng đối tượng, loại nền và phần thuộc đối tượng khác khỏi mask, không tự tô thêm vùng bị che; các instance cùng lớp vẫn có mask riêng. Nếu công cụ và định dạng có hỗ trợ lỗ/vùng rời thì xử lý theo guideline của dự án; không tự chọn cách nối qua vùng nền khi chưa có quy tắc.

5. **Biên khó và guideline/escalation:** trên ảnh `kitchen`, bàn tiền cảnh tiếp xúc và bị che bởi các bát/khay cùng đồ vật đặt trên bàn. Record `kitchen-005` được model gọi là `dining table`, `score = 0.578416`, với `598` điểm polygon. Cần kiểm tra vùng tiếp giáp ở mép các đồ vật và xác định mask của bàn bao phần nào. Nếu guideline đã quy định mask phần nhìn thấy và phạm vi lớp bàn/quầy thì áp dụng để tách vùng của các instance. Nếu chưa rõ quầy bếp có thuộc lớp `dining table` hay không, hoặc có vẽ phần mặt bàn bị đồ vật che hay không, cần escalation; không biến mask của model thành quy tắc chuẩn.

## 4. Vòng đời và kiểm tra chất lượng

Lab này chỉ xem prediction. Vòng đời production để nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Trong ví dụ single-label, một lớp từ taxonomy quy định cho toàn ảnh. | Ảnh `traffic` có nhiều loại xe nhưng top-1 là `cab`, score `0.510915`; prediction không xác định chiếc xe nào và chưa đủ để xác nhận nhãn toàn ảnh. | Đọc tiêu chí chọn chủ đề chính; chọn một lớp theo guideline hoặc escalation nếu không đủ thông tin. | Kiểm tra nhãn có phù hợp toàn ảnh và quy tắc ưu tiên không; đối chiếu ID/tên/taxonomy, không duyệt chỉ vì top-1 có score cao nhất. |
| Phát hiện vật thể | Một lớp và một bounding box cho mỗi đối tượng cần gán nhãn theo guideline. | `kitchen`: record `person` score `0.610931`, box `[0.12, 263.18, 61.55, 310.78]` nằm sát mép trái, chỉ thấy một phần người/tay. | Áp dụng quy tắc đối tượng cắt mép và phần nhìn thấy; không đoán phần ngoài ảnh; hỏi Lab Coach khi thiếu quy tắc. | Kiểm tra chính sách có gán nhãn đối tượng này không, box có bám đúng phần cần gán không, có bỏ sót/nhầm lớp/trùng hộp không; yêu cầu sửa nếu không khớp. |
| Instance segmentation | Một lớp và một mask/polygon riêng cho từng instance theo phạm vi gán nhãn đã quy định. | `kitchen-005` được dự đoán `dining table`, score `0.578416`, polygon `598` điểm; bàn/quầy bị đồ vật tiền cảnh che, khó xác định phạm vi lớp và vùng biên. | Tách biên bàn với đồ vật theo guideline; không tự điền vùng bị che; escalation nếu taxonomy hoặc quy tắc biên chưa rõ. | So mask với ảnh, tìm vùng nền/đồ vật khác bị tô lẫn, vùng thiếu và instance bị gộp; kiểm tra lớp và yêu cầu rework. |

Các tình huống trên là điểm mơ hồ cần con người đối chiếu, không phải kết luận model sai dựa trên một bộ ground truth có sẵn. Trong lab không huấn luyện mô hình và không tạo bộ ground truth hoàn chỉnh.

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu:** chỉ dùng ba ảnh COCO công khai đã cố định và giữ attribution đi kèm; không đưa ảnh cá nhân, dữ liệu khách hàng, dữ liệu nội bộ hoặc thông tin định danh học viên vào báo cáo/output công khai; họ tên và MSSV chỉ đặt trong tên repository theo yêu cầu lớp.

2. **Khi gặp dữ liệu ngoài phạm vi:** dừng xử lý và không tải lên Colab/GitHub; báo Lab Coach hoặc giảng viên qua kênh hỗ trợ chính thức, mô tả vấn đề mà không gửi kèm dữ liệu nhạy cảm khi chưa được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Đã kiểm tra đủ các tệp trong bộ nộp tại thư mục `report/` trên máy. Khi đưa lên GitHub, giữ nguyên cấu trúc này và mở lại từng tệp để xác nhận upload thành công.

Kiểm tra bộ tệp nộp trong thư mục `report/`:

- ✅ `REPORT.md` (đã điền và đặt đúng vị trí trong bộ nộp)
- ✅ `day1_lab_outputs/classification_predictions.json`
- ✅ `day1_lab_outputs/detection_predictions.json`
- ✅ `day1_lab_outputs/segmentation_predictions.json`
- ✅ `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- ✅ `day1_lab_outputs/visuals/classification_top5.png`
- ✅ `day1_lab_outputs/visuals/detection_predictions.png`
- ✅ `day1_lab_outputs/visuals/segmentation_prediction.png`
- ✅ Ô validation cuối phần tạo evidence trong notebook báo `PASS`
- ✅ Báo cáo không chứa họ tên, MSSV hoặc dữ liệu nhạy cảm của học viên; các JSON và PNG đã đối chiếu dùng ảnh mẫu công khai của bài lab
- ✅ Đã nộp **link repository** trên LMS
