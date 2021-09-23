### Code convention của engine dịch hiện tại:
- Sử dụng python với indent là 4 space, không dùng tab.
- Cố gắng hạn chế sử dụng lệnh import * tới mức tối đa. Ưu tiên *import cả file vào làm namespace* > *import các function* > _import *_ (vd em dùng hàm foo trong file bar, thì **import bar as b; b.foo()** > **from bar import foo; foo()** > __from bar import *, foo()__ )
- Tên class đặt theo CamelCase (e.g *MyCustomClass*)
- Tên hàm, file và các biến bên trong theo snake_case (e.g *def my_function()*, *my_variable*)
- Tên các biến global và hằng số constant đặt bằng ALL_CAP (e.g *DEFAULT_VALUE*)
- String sử dụng ""
- Các file dạng đọc được cố hết sức quy chuẩn về unicode NFKC và encoding utf-8

### Format comment:
Các hàm sẽ comment theo mẫu:

"""**Giải thích mục đích của hàm**

Args:

  **tên argument**: Argument đại biểu cho cái gì, *shape* và *type* của nó

Returns: (Hàm void không cần phần này; mỗi dòng một giá trị nếu trả ra tuple.)
  
  Giá trị đầu ra của hàm, ý nghĩa, *shape* và *type*.
"""
# MultilingualIMT-UET-KC4.0
