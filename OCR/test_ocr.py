# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Run OCR inference on a sample image 



# results = ocr.ocr(frame, cls=False, rec=False)(input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

# # Visualize the results and save the JSON results
# for res in result:
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")