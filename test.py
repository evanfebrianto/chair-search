from util import Downloader, ProductCatalogue
from config import config as cfg
import json

pc = ProductCatalogue(project_id=cfg.PROJECT_ID, location_id=cfg.LOCATION_ID)
product_mapping = pc.get_product_mapping()

downloader = Downloader(project_id=cfg.PROJECT_ID, bucket_name=cfg.BUCKET_NAME)
bucket= {'scraped_images/ÖGLA/test_1.png': 'ÖGLA', 'scraped_images/IDOLF/test_1.png': 'IDOLF', 'scraped_images/NORRARYD/test_1.png': 'NORRARYD', 'scraped_images/HARRY/test_1.png': 'HARRY'}
downloader.download_blob_manager(list(product_mapping.values()))
with open('blob_images.json', 'w') as f:
    json.dump(downloader.blob_images.copy(), f)


# # SuperFastPython.com
# # example of using the manager to create a shared queue
# from time import sleep
# from random import random
# from multiprocessing import Process
# from multiprocessing import Manager
 
# # task executed in a new process
# def task(number, shared_queue):
#     # generate a number between 0 and 1
#     value = random()
#     # block for a fraction of a second
#     sleep(value)
#     # add the value to the queue
#     shared_queue.put((number, value))
 
# # protect the entry point
# if __name__ == '__main__':
#     # create the manager
#     with Manager() as manager:
#         # create the shared queue
#         shared_queue = manager.Queue()
#         # create many child processes
#         n_tasks = 50
#         processes = [Process(target=task, args=(i, shared_queue)) for i in range(n_tasks)]
#         # start all processes
#         for process in processes:
#             process.start()
#         # read data from the queue
#         for _ in range(n_tasks):
#             # get an item from the queue
#             item = shared_queue.get()
#             # report the item
#             print(f'> got {item}')