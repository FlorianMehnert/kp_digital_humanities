import random


def load_newspaper_text(filename):
    """Load preprocessed newspaper text file"""
    try:
        with open(filename, 'r') as f:
            article_text = f.read()
        return article_text.strip()  # Remove leading/trailing whitespace
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None


def limit_string_length(text, max_length):
    if len(text) > max_length:
        return text[:max_length].rstrip() + '...'
    else:
        return text


def delete_random_words(article_text, num_words_to_delete):
    """Delete random words from the article text"""

    if num_words_to_delete > len(article_text.split()):
        raise ValueError("Cannot delete more words than exist in the article")

    print(len(article_text.split()))
    deleted_words = []

    # Delete word at random index until done
    i = 0
    while len(deleted_words) < num_words_to_delete:
        if len(deleted_words) > 1:
            print(deleted_words[-1])
        word_index = random.randint(0, len(article_text.split()) - 1)
        word = article_text.split()[word_index]
        if word not in deleted_words:
            deleted_words.append(word)

    gap_article_text = ""
    for i, word in enumerate(article_text.split()):
        if i % 6 == 5:  # Add linebreak after every 10 words
            gap_article_text += word + "\n"
        elif word in deleted_words:
            gap_article_text += "______" + " "
        else:
            gap_article_text += word + " "

    return gap_article_text.strip()


# Example usage:
if __name__ == "__main__":
    filename = 'newspaper.txt'  # Replace with your preprocessed newspaper text file
    num_words_to_delete = 5  # Number of random words to delete

    article_text = load_newspaper_text(filename)
    if article_text:
        print("Loaded Article Text:")
        print(article_text)

        gap_article_text = delete_random_words(article_text, num_words_to_delete)
        print(f"\nDeleted {num_words_to_delete} Random Words:\n{gap_article_text}")
