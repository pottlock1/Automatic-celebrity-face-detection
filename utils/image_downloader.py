import os
from selenium import webdriver
import requests
import time
import pickle

class CollectCelebImages:
    def __init__(self, celeb1, no_of_images = 20):
        self.celebs = [celeb1]
        self.no_of_images = no_of_images

    def download(self):
        for celeb in self.celebs:
            self.link_extractor(search_string=celeb, no_of_images=self.no_of_images)
            self.imagedownloader(celeb)


    def link_extractor(self, search_string: str, no_of_images, sleep_time=2, wd=webdriver):
        # opeing google chrome using selenium webdriver and searching our query...
        wd = webdriver.Chrome(executable_path='chromedriver.exe')
        search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"
        wd.get(search_url.format(q=search_string))
        time.sleep(5)

        # getting thumbnail_images...
        thumbnail_result = wd.find_elements_by_css_selector('img.Q4LuWd')
        print(f'{len(thumbnail_result)} images are found!!')

        links = []
        counter = 0
        while len(links) < no_of_images:
            img = thumbnail_result[counter]
            img.click()
            time.sleep(sleep_time)
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            print(f'{len(actual_images)} candidate actual images are found for image {counter} !!')

            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    links.append(actual_image.get_attribute('src'))
                else:
                    pass
            print(f'no of links extracted = {len(links)}')
            counter += 1

        f = open('train_dumps\image_links_{}.pickle'.format(search_string.replace(' ', '')), 'ab')
        pickle.dump(links, f)
        f.close()
        print('{} image links have been saved in pickle format!!'.format(len(links)))
        wd.quit()


    def imagedownloader(self, query):
        if not os.path.exists(os.path.join('celeb_images', query)):
            os.mkdir(os.path.join('celeb_images', query))

        links = pickle.loads(open('train_dumps\image_links_{}.pickle'.format(query.replace(' ', '')), 'rb').read())
        counter = 0
        for i, link in enumerate(links):
            # try:
            response = requests.get(link)
            f = open(os.path.join('celeb_images', query, 'img_{}.jpg'.format(str(i+1))), 'wb')
            f.write(response.content)
            f.close()
            counter += 1
            print('Total downloaded imges = {}'.format(counter))

            # except:
            #     print('Cannot download image {}'.format(i))

        print('{} images of {} have been downloaded!!'.format(len(links), query))


