# To make the lyrics data more usable, we remove empty lines
def remove_empty_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Remove empty lines
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    # Write the non-empty lines to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(non_empty_lines))

# Lowercase the text to make the vocab size smaller. For the purpose of this basic LLM we don't care (its also trained on a Mac)
def preprocess_lyrics(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Remove empty lines and lowercase the text
    processed_lines = [line.strip().lower() for line in lines if line.strip()]
    
    # Write the processed lines to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(processed_lines))

# Specify the paths to the input and output files
input_file = 'all_tswift_lyrics.txt'
output_file = 'all_tswift_lyrics_cleaned.txt'

# Call the function to remove empty lines and save to a new file
remove_empty_lines(input_file, output_file)
preprocess_lyrics(input_file, output_file)
