from .camera import Camera
import traitlets
import jetson.utils


class CSICamera(Camera):

    capture_device = traitlets.Integer(default_value=0)
    capture_fps = traitlets.Integer(default_value=30)
    capture_width = traitlets.Integer(default_value=640)
    capture_height = traitlets.Integer(default_value=480)

    def __init__(self, *args, **kwargs):
        super(CSICamera, self).__init__(*args, **kwargs)

        self.input = jetson.utils.videoSource(
            f'csi://{self.capture_device}',
            [f'--input-width={self.width}', f'--input-height={self.height}'])

    def _read(self):
        cuda_img = self.input.Capture()
        dummy = jetson.utils.cudaAllocMapped(width=cuda_img.width,
                                             height=cuda_img.height,
                                             format='bgr8')
        jetson.utils.cudaConvertColor(cuda_img, dummy)
        jetson.utils.cudaDeviceSynchronize()
        return jetson.utils.cudaToNumpy(dummy)
