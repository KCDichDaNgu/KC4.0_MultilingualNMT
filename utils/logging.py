from __future__ import absolute_import

import os
import logging

def init_logger(model_dir, log_file=None, rotate=False):

    logging.basicConfig(level=logging.DEBUG,
                            format='[%(asctime)s %(levelname)s] %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=os.path.join(model_dir, log_file),
                            filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', '%a, %d %b %Y %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    return logging