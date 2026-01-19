from s4_debugging_and_logging.exercise_files import vae_mnist_working
import torch

class Test_Data_Shape:

    
    def test_get_data_shapes(self):
        train_loader, test_loader = vae_mnist_working.get_data("datasets", batch_size=32)

        # grab one batch
        x, y = next(iter(train_loader))

        # minimal assertions
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        # MNIST = (N, 1, 28, 28)
        assert x.shape[1:] == (1, 28, 28)
        assert y.ndim == 1
        assert len(x) == len(y)
