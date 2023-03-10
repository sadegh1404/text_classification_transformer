from omegaconf import OmegaConf

from text_classification.trainer import Trainer

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to yaml config file', default='configs/sentiment.yaml')

    args = parser.parse_args()

    config_path = args.config
    config = OmegaConf.load(config_path)

    trainer = Trainer(config)
    trainer.train()
