import logging
import coloredlogs
import os


# todo remove color when logging in file
def get_logger(name='', save_dir=None, distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger(name)
    coloredlogs.install(level='DEBUG', logger=logger)
    # logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")

    # ch = logging.StreamHandler(stream=sys.stdout)
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
