from src.utils import create_input_files

if __name__ == '__main__':
    create_input_files(csv_folder='./src/data',
                       output_folder='./src/checkpoints',
                       sentence_limit=15,
                       word_limit=20,
                       min_word_count=5)

