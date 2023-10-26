import json

import cv2
import matplotlib.pyplot as plt
from path import Path

from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree

with open('../data/config.json') as f:
    sample_config = json.load(f)

with open('../data/words_alpha.txt') as f:
    word_list = [w.strip().upper() for w in f.readlines()]
prefix_tree = PrefixTree(word_list)

for decoder in ['best_path', 'word_beam_search']:
    for img_filename in Path('../data').files('*.png'):
        print(f'Reading file {img_filename} with decoder {decoder}')

        # read text
        img = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
        scale = sample_config[img_filename.basename()]['scale'] if img_filename.basename() in sample_config else 1
        margin = sample_config[img_filename.basename()]['margin'] if img_filename.basename() in sample_config else 0
        read_lines = read_page(img,
                               detector_config=DetectorConfig(scale=scale, margin=margin),
                               line_clustering_config=LineClusteringConfig(min_words_per_line=2),
                               reader_config=ReaderConfig(decoder=decoder, prefix_tree=prefix_tree))

        # output text
        sentences = []
        # for read_line in read_lines:
        #     print(' '.join(read_word.text for read_word in read_line))

        for read_line in read_lines:
            sentence = ' '.join(read_word.text for read_word in read_line)
            sentences.append(sentence)
            print(sentence)
            
        print()

        file_path = "C:\paper project\HTRPipeline-master\scripts\sentence.txt"

        # Open the file in write mode, create it if it doesn't exist, and write the sentences to it
        with open(file_path, 'w') as file:
            file.write('\n'.join(sentences))

        # plot image with detections and texts as overlay
        plt.figure(f'Image: {img_filename} Decoder: {decoder}')
        plt.imshow(img, cmap='gray')
        for i, read_line in enumerate(read_lines):
            for read_word in read_line:
                aabb = read_word.aabb
                xs = [aabb.xmin, aabb.xmin, aabb.xmax, aabb.xmax, aabb.xmin]
                ys = [aabb.ymin, aabb.ymax, aabb.ymax, aabb.ymin, aabb.ymin]
                plt.plot(xs, ys, c='r' if i % 2 else 'b')
                plt.text(aabb.xmin, aabb.ymin - 2, read_word.text)

plt.show()