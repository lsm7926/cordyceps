import os
import json
import time
import schedule

'''
from edge import edge_detection, sobel_filter, canny_edge_detector
from circle import circle_detection
from houghcircle import houghcircle
from removebg import removebg
from tunebg import tunebg
from center import center
'''
import preprocess
from config import cfg
from logging.handlers import RotatingFileHandler
import logging
logger = logging.getLogger()
import utils


class Main():
    def job(self):
        logger.info('job')
    
    
    def process(self):
        files = utils.get_files_fullpath(cfg['raw']['path'])
        for idx, file in enumerate(files):
            filename = os.path.basename(file)
            removeexif = preprocess.removeexif(file)
            threshold = preprocess.threshold(removeexif)
            removebg = preprocess.removebg(threshold)
            mask = preprocess.mask(removebg)
            centroid = preprocess.center(mask)
            cropped_removeexif, cropped_mask = preprocess.crop(removeexif,mask,cfg['conv']['crop'],centroid)
            segments, res = preprocess.maskslic(cropped_removeexif,cropped_mask)
            preprocess.saveslic(res, segments, idx)
            print('{} / {} {} Done.'.format(idx + 1, len(files), filename))


    def __init__(self):
        logger.info('init')
        self.process()
        
        '''
        schedule.every(1).minutes.do(self.job)
        while True:
            schedule.run_pending()
        '''


if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s %(message)s', "%Y-%m-%d %H:%M:%S")
    
    handler = RotatingFileHandler(os.path.join(cfg['base']['path'],'logger.log'), mode='a', maxBytes=3*1024*1024, backupCount=2)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('main')
    Main()
    logger.info('quit')
