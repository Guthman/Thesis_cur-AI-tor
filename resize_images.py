from concurrent.futures import ProcessPoolExecutor


def resize(image_):
    # TODO: determine proper image size for training the model, test function in notebook, finish code here
    return


def main():

    with ProcessPoolExecutor(max_workers=12) as executor:
        for image in executor.map(resize, image_paths):
            resize(image)


if __name__ == '__main__':
    main()
