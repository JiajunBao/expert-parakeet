from utils import create_input_files

if __name__ == '__main__':
    create_input_files(csv_folder='./data',
                       output_folder='./checkpoints',
                       sentence_limit=15,
                       word_limit=20,
                       min_word_count=5)

