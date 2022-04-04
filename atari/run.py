# import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# dotenv.load_dotenv(override=True)

# import importer


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # if config.get("only_test"):
    #     return test(config)
    # # Train model
    # return train(config)
    if config.get("alg.name") == "dqn":
        return
    else:
        return train(config)


if __name__ == "__main__":
    main()
