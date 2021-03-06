import time
import logging.config
import cv2
import os
from differnet.differnet_util import DiffernetUtil
import time
from conf.settings import LOGGING, DIFFERNET_CONF

# import logging
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)

# https://www.django-rest-framework.org/api-guide/testing/#apiclient


def test_train():
    """Test training
    use nvidia-smi -i 5 -l 5 to monitor GPU
    -i GPU device number -l interval to refresh
    """
    # load cutomized conf
    conf = DIFFERNET_CONF
    logger.info(f"working folder: {conf.get('differnet_work_dir')}")

    differnetutil = DiffernetUtil(conf, "black1")

    # train the model
    differnetutil.train_model()


def test_model():
    """Test training"""
    # load cutomized conf
    conf = DIFFERNET_CONF
    logger.info(f"working folder: {conf.get('differnet_work_dir')}")

    differnetutil = DiffernetUtil(conf, "black1")

    t1 = time.process_time()

    # test trained model
    differnetutil.test_model()

    t2 = time.process_time()
    elapsed_time = t2 - t1
    logger.info(f"elapsed time: {elapsed_time}")


def test_detect():
    """Test Detection"""
    # load cutomized conf
    conf = DIFFERNET_CONF
    logger.info(f"working folder: {conf.get('differnet_work_dir')}")

    t0 = time.process_time()
    differnetutil = DiffernetUtil(conf, "black1")
    differnetutil.load_model()
    t1 = time.process_time()

    elapsed_time = t1 - t0
    logger.info(f"Model load elapsed time: {elapsed_time}")

    img = cv2.imread(
        os.path.join(
            differnetutil.test_dir, "defect", "Camera0_202009142018586_product.png"
        ),
        cv2.IMREAD_UNCHANGED,
    )
    t1 = time.process_time()
    ret = differnetutil.detect(img, 10)
    # calculate time
    t2 = time.process_time()
    elapsed_time = t2 - t1
    logger.info(f"Detection elapsed time: {elapsed_time}")
    assert ret == True

    img = cv2.imread(
        os.path.join(
            differnetutil.test_dir, "good", "Camera0_202009142018133_product.png"
        ),
        cv2.IMREAD_UNCHANGED,
    )
    t2 = time.process_time()
    ret = differnetutil.detect(img, 10)
    t3 = time.process_time()
    elapsed_time = t3 - t2
    logger.info(f"Detection elapsed time: {elapsed_time}")
    assert ret == False


if __name__ == "__main__":
    test_train()
    test_model()
    test_detect()