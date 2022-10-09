from util import Downloader
from config import config as cfg

downloader = Downloader(project_id=cfg.PROJECT_ID, bucket_name=cfg.BUCKET_NAME)
bucket= {'scraped_images/ÖGLA/test_1.png': 'ÖGLA', 'scraped_images/IDOLF/test_1.png': 'IDOLF', 'scraped_images/NORRARYD/test_1.png': 'NORRARYD', 'scraped_images/HARRY/test_1.png': 'HARRY'}
# blob_images = downloader.get_blob_images(list(bucket.keys()))
# print(blob_images)

blob_images = downloader.get_blob_images_parallel(list(bucket.keys()))
print(blob_images)