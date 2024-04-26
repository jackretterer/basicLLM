# Lowercase the text to make the vocab size smaller. Also remove empty lines. For the purpose of this basic LLM we don't care (its also trained on a Mac)
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
preprocess_lyrics(input_file, output_file)
