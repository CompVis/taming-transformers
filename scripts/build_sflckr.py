from tqdm import tqdm
import requests
import os
import argparse
import time
from flickrapi import FlickrAPI

KEY = 'deb1fdcc4a1992e92e5b16f8f04ba8e6'
SECRET = '30ebb04328962696'

SIZES = ["url_l"]  # Large 1024 (1024 × 732) | ["url_o", "url_k", "url_h", "url_l", "url_c"] in order of preference


def get_photos(image_tag):
    extras = ','.join(SIZES)
    flickr = FlickrAPI(KEY, SECRET)
    photos = flickr.walk(text=image_tag,  # it will search by image title and image tags
                         extras=extras,  # get the urls for each size we want
                         privacy_filter=1,  # search only for public photos
                         per_page=50,
                         sort='relevance')  # we want what we are looking for to appear first
    return photos


def get_url(photo):
    for i in range(len(SIZES)):  # makes sure the loop is done in the order we want
        url = photo.get(SIZES[i])
        if url:  # if url is None try with the next size
            return url


def get_urls(image_tag, max):
    photos = get_photos(image_tag)
    counter = 0
    urls = []

    for photo in photos:
        if counter < max:
            url = get_url(photo)  # get preffered size url
            if url:
                urls.append(url)
                counter += 1
            # if no url for the desired sizes then try with the next photo
        else:
            break

    return urls


def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def download_images(urls, path):
    create_folder(path)  # makes sure path exists

    for url in urls:
        image_name = url.split("/")[-1]
        image_path = os.path.join(path, image_name)

        if not os.path.isfile(image_path):  # ignore if already downloaded
            response = requests.get(url, stream=True)

            with open(image_path, 'wb') as outfile:
                outfile.write(response.content)


def download(size, tag_path, output_path):
    with open(tag_path, 'r') as fhandle:
        all_tags = fhandle.readlines()
        images_per_tag = size // len(all_tags)
        for tag in tqdm(all_tags):
            tag = tag.replace('\n', '')
            print('Getting urls for', tag)
            urls = get_urls(tag, images_per_tag)
            print('Downloading images for', tag)
            path = os.path.join(output_path, tag)
            download_images(urls, path)


def argparse_setup():
    # Init parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Arguments
    parser.add_argument("--tag_fpath", type=str, help="File path to pick the image tag names from.")
    parser.add_argument("--output_path", type=str, default="./downloaded",
                        help="Output directory for the downloaded images.")
    parser.add_argument("--n_images", type=int, default=10000, help="Total number of images to download.")
    # Return args object
    return parser.parse_args()


if __name__ == '__main__':
    args = argparse_setup()
    start_time = time.time()
    download(size=args.n_images, tag_path=args.tag_fpath, output_path=args.output_path)
    print('Took', round(time.time() - start_time, 2), 'seconds')
