# Document Visual Q&A（DOC-VQA）

Document Visual Q&A, mainly for the image content of the question and answer, DOC-VQA is a type of VQA task, DOC-VQA mainly asks questions about the textual content of text images.

The DOC-VQA algorithm in PP-Structure is developed based on PaddleNLP natural language processing algorithm library.

The main features are as follows:

- Integrated LayoutXLM model and PT-OCR prediction engine.
- Support Semantic Entity Recognition (SER) and Relation Extraction (RE) tasks based on multi-modal methods. Based on SER task, text recognition and classification in images can be completed. Based on THE RE task, we can extract the relation of the text content in the image, such as judge the problem pair.

- Support custom training for SER and RE tasks.

- Support OCR+SER end-to-end system prediction and evaluation.

- Support OCR+SER+RE end-to-end system prediction.

**Note**: This project is based on the open source implementation of  [LayoutXLM](https://arxiv.org/pdf/2104.08836.pdf) on Paddle 2.2, and at the same time, after in-depth polishing by the flying Paddle team and the Industrial and **Commercial Bank of China** in the scene of real estate certificate, jointly open source.

## convert

```bash
cd ${PaddleOCR2Pytorch_PATH}/converter
# model_state.pth will saved in original paddlepaddle model directory, then you can cancel the model_state.pdparams

# layoutxlm ser
python3.7 ./layoutxlm_ser_converter.py --model_name_or_path your_org_PP-Layout_v1.0_ser_pretrained_dir

#layoutxlm re
python3.7 ./layoutxlm_re_converter.py --re_model_name_or_path your_org_PP-Layout_v1.0_re_pretrained_dir
```

## usage

```bash
cd ${PaddleOCR2Pytorch_PATH}/ptstructure/vqa
# either paddlepaddle model or pytorch model can be loaded.

# layoutxlm ser
python3.7 infer_ser_e2e.py --det_model_path ptocr_general_model_ch_ptocr_v2_det_infer.pth --rec_model_path ptocr_general_model_ch_ptocr_v2_rec_infer.pth --model_name_or_path your_PP-Layout_v1.0_ser_pretrained_dir --output_dir ./output/ser_e2e --infer_imgs ./images/input/zh_val_0.jpg

#layoutxlm ser_re
python3.7 infer_ser_re_e2e.py --det_model_path ptocr_general_model_ch_ptocr_v2_det_infer.pth --rec_model_path ptocr_general_model_ch_ptocr_v2_rec_infer.pth --model_name_or_path your_PP-Layout_v1.0_ser_pretrained_dir --re_model_name_or_path your_PP-Layout_v1.0_re_pretrained_dir --output_dir ./output/ser_re_e2e/ --infer_imgs ./images/input/zh_val_40.jpg
```

## Demonstration

### ser

![](./output/ser_e2e/zh_val_0_ser.jpg)

![](./output/ser_e2e/zh_val_42_ser.jpg)

### ser_re

![](./output/ser_re_e2e/zh_val_21_re.jpg)

![](./output/ser_re_e2e/zh_val_40_re.jpg)

## References

- [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/pdf/2104.08836.pdf)
- [microsoft/unilm/layoutxlm](https://github.com/microsoft/unilm/tree/master/layoutxlm)
- [XFUND dataset](https://github.com/doc-analysis/XFUND)