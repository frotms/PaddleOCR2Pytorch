
import os, sys
import numpy as np
import cv2
import onnxruntime as rt
class OnnxInfer:
    def __init__(self, onnx_path):
        self.model_path = os.path.abspath(os.path.expanduser(onnx_path))
        if not os.path.exists(self.model_path):
            raise FileNotFoundError('{} is not existed.'.format(self.model_path))

        self.sess = rt.InferenceSession(self.model_path)
        self.input_name = self.get_input_name(self.sess)
        self.output_name = self.get_output_name(self.sess)
        print('{} is loaded.'.format(self.model_path))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def run(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        out = self.sess.run(self.output_name, input_feed=input_feed)
        return out

if __name__ == '__main__':
    print('begin..')
    np.random.seed(666)
    input_size = (1, 3, 32, 320)
    onnx_path = 'ch_ptocr_server_v2.0_rec_infer_optim.onnx'

    inp = np.random.randn(*input_size).astype(np.float32)
    print('input: ', inp.shape)

    onnxmodel = OnnxInfer(onnx_path)
    onnx_res = onnxmodel.run(inp.copy())
    print('the length of onnx inference is {}'.format(len(onnx_res)))
    out = onnx_res[0]
    print('out shape: {}'.format(out.shape))
    print('done.')
