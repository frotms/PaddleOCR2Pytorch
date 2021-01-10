# coding: utf-8
# https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python
import os, sys
import numpy as np
from paddle import fluid


from paddle.inference import Config
from paddle.inference import create_predictor

class PaddlePaddleOCR:
    def __init__(self, cfg):
        self.cfg = cfg
        self.predictor = self.init_predictor(cfg)
        print('model is inited')

    def init_predictor(self, cfg):
        model_dir = cfg['model_dir']
        params_file = cfg['params_file']
        use_gpu = cfg['use_gpu']

        config = Config(model_dir, params_file)

        # config.enable_memory_optim()

        if use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            # If not specific mkldnn, you can set the blas thread.
            # The thread num should not be greater than the number of cores in the CPU.
            config.set_cpu_math_library_num_threads(4)
            config.enable_mkldnn()
        config.disable_glog_info()

        predictor = create_predictor(config)
        return predictor

    def run(self, image_data):
        inp = np.expand_dims(image_data, 0)
        input_names = self.predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.reshape(inp.shape)
            input_tensor.copy_from_cpu(inp.copy())

        # do the inference
        self.predictor.run()

        results = []
        # get out data from output tensor
        output_names = self.predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = self.predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)

        return results


if __name__ == '__main__':
    print('begin...')

    USE_GPU = False
    # root_dir = './ch_ppocr_server_v2.0_det_infer'
    # root_dir = 'C:/workspace/repo/OCR/ai_groupOCR/ppocr_models/v2/ch_ppocr_mobile_v2.0_det_infer'
    # root_dir = 'C:/workspace/repo/OCR/ai_groupOCR/ppocr_models/v2/ch_ppocr_mobile_v2.0_cls_infer'
    root_dir = 'C:/workspace/repo/OCR/ai_groupOCR/ppocr_models/v2/ch_ppocr_server_v2.0_rec_infer'
    MODEL_DIR = 'inference.pdmodel'
    PARAMS_FILE = 'inference.pdiparams'

    os_cvd = '-1' if USE_GPU==False else '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    h = 32
    w = 320
    c = 3
    np.random.seed(666)
    # inp = np.random.rand(c, h, w).astype(np.float32)
    inp = np.random.randn(c, h, w).astype(np.float32)
    print('==> ',np.sum(inp), np.mean(inp), np.max(inp), np.min(inp))
    # print(inp, inp.shape, inp.dtype, '\n')

    import cv2

    image = cv2.imread('C:/workspace/repo/OCR/ai_groupOCR/ai_OCR_Recognizer/Snipaste.jpg')
    image = cv2.resize(image, (w, h))
    mean = 0.5
    std = 0.5
    scale = 1. / 255
    norm_img = (image * scale - mean) / std
    inp = norm_img
    inp = inp.transpose(2, 0, 1)
    print(np.sum(inp), np.mean(inp), np.max(inp), np.min(inp))

    #
    # import cv2
    # image = cv2.imread('6.jpg')
    # image = cv2.resize(image, (320, 448))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # scale = 1. / 255
    # norm_img = (image * scale - mean) / std
    # inp = norm_img
    # inp = inp.transpose(2,0,1)



    inp = inp.astype(np.float32)
    # print(inp, inp.shape, inp.dtype, '\n')

    model_file_path = os.path.abspath(os.path.join(root_dir, MODEL_DIR))
    params_file_path = os.path.abspath(os.path.join(root_dir, PARAMS_FILE))
    if not os.path.exists(model_file_path):
        print("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        print("not find params file path {}".format(params_file_path))
        sys.exit(0)

    cfg = {}
    cfg['model_dir'] = model_file_path
    cfg['params_file'] = params_file_path
    cfg['use_gpu'] = USE_GPU

    paddleOCR = PaddlePaddleOCR(cfg)
    results = paddleOCR.run(inp)
    print(results[0].shape)
    print(np.sum(results), np.mean(results), np.max(results), np.min(results))
    print('done!')




