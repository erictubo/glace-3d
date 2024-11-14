
import argparse
import logging
from distutils.util import strtobool
from pathlib import Path

from encoder_trainer import TrainerEncoder


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Fine-tuning of a pre-trained encoder network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='path to pre-trained encoder network')
    
    parser.add_argument('data_path', type=Path,
                        help='path to the dataset folder')
    
    
    parser.add_argument('dataset_names', type=str, nargs='+',
                        help='names of the datasets to use')

    parser.add_argument('val_dataset_name', type=str,
                        help='name of the dataset to use for validation')
    
    # TODO: head_paths dictionary for coords loss over multiple datasets
    

    parser.add_argument('loss_function', type=str, choices=['combined', 'separate'],
                        help='loss function')

    parser.add_argument('--use_coords', type=str, choices=['loss', 'track', 'none'],
                        help='use coordinates loss as')
    
    parser.add_argument('--median', type=_strtobool, default=False,
                        help='use median for coordinates loss')
    
    parser.add_argument('--coords_scale', type=float, default=1/50,
                        help='scale for coordinates loss')
    
    parser.add_argument('--use_cosine', type=str, choices=['loss', 'track', 'none'],
                        help='use cosine loss as')
    
    parser.add_argument('--cosine_weights', type=float, nargs='+',
                        help='weights for the cosine loss: separate: 2, combined: 3')
    
    parser.add_argument('--use_magnitude', type=str, choices=['loss', 'track', 'none'],
                        help='use magnitude loss as')

    
    # parser.add_argument('experiment_name', type=str,
    #                     help='name of the experiment')
    
    # parser.add_argument('output_path', type=Path,
    #                     help='path to the output folder')
    
    
    # parser.add_argument('--clip_norm', type=float, default=1.0,
    #                     help='clip norm')

    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='number of epochs')
    
    parser.add_argument('--max_iterations', type=int, default=2000,
                        help='maximum number of iterations')
    
    parser.add_argument('--gradient_accumulation_samples', type=int, default=20,
                        help='number of samples to accumulate gradients')
    
    parser.add_argument('--validation_frequency', type=int, default=5,
                        help='validation frequency')
    
    parser.add_argument('--iter_val_limit', type=int, default=20,
                        help='number of samples for each validation')
    
    parser.add_argument('--epoch_val_limit', type=int, default=80,
                        help='number of samples for epoch validation')
    



    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='use half precision')
    
    parser.add_argument('--image_height', type=int, default=480,
                        help='image height')
    
    parser.add_argument('--aug_rotation', type=int, default=40,
                        help='augmentation rotation')
    
    parser.add_argument('--aug_scale_min', type=float, default=240/480,
                        help='augmentation scale min')
    
    parser.add_argument('--aug_scale_max', type=float, default=960/480,
                        help='augmentation scale max')
    
    options = parser.parse_args()
    
    trainer = TrainerEncoder(options)
    trainer.train()