import argparse, os, datetime, yaml
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint
from lightning.pytorch.cli import (
    LightningCLI,
    LightningArgumentParser,
    SaveConfigCallback,
)
from ecdm.callbacks import (
    CUDACallback,
)  # noqa: F401


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Implement to add extra arguments to the parser or link arguments.

        Args:
            parser: The parser object to which arguments can be added

        """
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint_callback")
        parser.set_defaults(
            {
                "checkpoint_callback.dirpath": None,
                "checkpoint_callback.filename": "{epoch:06}",
                "checkpoint_callback.verbose": True,
                "checkpoint_callback.save_last": True,
                "checkpoint_callback.save_top_k": 20,
                "checkpoint_callback.monitor": "val/rec_loss",
                "checkpoint_callback.save_weights_only": False,
            }
        )
        parser.add_lightning_class_args(OnExceptionCheckpoint, "exception_callback")
        parser.set_defaults(
            {
                "exception_callback.dirpath": "logs",
                "exception_callback.filename": "exc_last",
            }
        )

        parser.add_lightning_class_args(CUDACallback, "cuda_callback")
        parser.add_argument(
            "-l",
            "--logdir",
            type=str,
            default="logs",
            help="directory for logging dat shit",
        )
        parser.add_argument(
            "--scale_lr",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="scale base-lr by ngpu * batch_size * n_accumulate",
        )
        parser.add_argument(
            "--vscode_debug",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="scale base-lr by ngpu * batch_size * n_accumulate",
        )

    def before_instantiate_classes(self) -> None:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        name = (
            "_"
            + self.config[self.config.subcommand]
            .config[-1]
            .relative.split(".")[0]
            .split("/")[-1]
        )
        if self.config[self.subcommand].vscode_debug:
            nowname = "debug_" + now + name
        else:
            nowname = now + name

        if self.subcommand == "fit":
            base_lr = self.config[self.subcommand].model.init_args.learning_rate
            accumulate_grad_batches = self.config[
                self.subcommand
            ].trainer.accumulate_grad_batches
            ngpu = len(
                self.config[self.subcommand].trainer.devices.strip(",").split(",")
            )
            bs = self.config[self.subcommand].data.init_args.batch_size
            learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        logdir = os.path.join(self.config[self.subcommand].logdir, nowname)
        self.config[self.subcommand].trainer.logger.init_args.save_dir = logdir
        ckpt_path = os.path.join(logdir, "checkpoints")
        self.save_config_kwargs = {
            "config_filename": "config.yaml",
        }
        self.config[self.subcommand].checkpoint_callback.dirpath = ckpt_path

        self.config[self.subcommand].exception_callback.dirpath = ckpt_path

        lightning_config = OmegaConf.create()
        for config_file in self.config[self.subcommand].config:
            with open(config_file) as f:
                light_config = yaml.load(f, Loader=yaml.FullLoader)
            lightning_config = OmegaConf.merge(lightning_config, light_config)

        model_config = OmegaConf.create(lightning_config["model"])
        data_config = OmegaConf.create(lightning_config["data"])
        config = OmegaConf.merge(model_config, data_config)

        lightning_config.pop("model")
        lightning_config.pop("data")

        # self.config[self.subcommand].setup_callback.config=config
        # self.config[self.subcommand].setup_callback.lightning_config=lightning_config

        pass


def cli_main():
    cli = MyLightningCLI(
        save_config_callback=SaveConfigCallback,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    cli_main()
